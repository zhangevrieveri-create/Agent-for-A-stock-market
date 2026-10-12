"""PPO-Clip 训练（算法1 主训练循环）。

for k = 1..K:
    1-2. 状态特征提取，策略网络/价值网络前向传播
    3.   课程噪声注入与动作采样：a_t = μ_t + (σ_t + σ_noise)⊙ε_t，A_t = clip(a_t,0,1)/Σ
    4.   若 |D| ≥ B：计算 GAE，E 轮从 D 中采样批量 B 更新 L_total = L_policy + 0.5·L_value
    5.   若 k mod 100 = 0：自适应裁剪参数 ε_t 与自适应学习率 η_t
"""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..backtest import metrics, model_policy, run_backtest
from ..utils import get_logger
from .env import PortfolioEnv, segment, to_weights
from .model import ActorCritic, calibrate_bn, param_table, xavier_init

log = get_logger()


class RolloutBuffer:
    """采样缓存区 D（容量 C，先进先出），存储 (s, a, R, V, V', μ, σ, σ_noise, 回合结束标记)。"""

    def __init__(self, capacity, obs_shape, act_dim):
        C = capacity
        self.C, self.ptr, self.size = C, 0, 0
        self.obs = np.zeros((C,) + tuple(obs_shape), np.float32)
        self.act, self.mu, self.sigma = (np.zeros((C, act_dim), np.float32) for _ in range(3))
        self.noise, self.rew, self.val, self.next_val = (np.zeros(C, np.float32) for _ in range(4))
        self.end = np.zeros(C, bool)

    def add(self, obs, act, mu, sigma, noise, rew, val, end):
        i = self.ptr
        self.obs[i], self.act[i], self.mu[i], self.sigma[i] = obs, act, mu, sigma
        self.noise[i], self.rew[i], self.val[i], self.end[i] = noise, rew, val, end
        self.next_val[i] = 0.0
        self.ptr, self.size = (i + 1) % self.C, min(self.size + 1, self.C)
        return i

    def gae(self, gamma, lam):
        """δ_i = R_i + γV'_i − V_i，M̂_i = Σ_l (γλ)^l δ_{i+l}；按时间顺序在回合边界截断。"""
        order = (np.arange(self.size) + self.ptr - self.size) % self.C
        adv, g = np.zeros(self.C, np.float32), 0.0
        for i in order[::-1]:
            delta = self.rew[i] + gamma * self.next_val[i] - self.val[i]
            g = delta + gamma * lam * (0.0 if self.end[i] else 1.0) * g
            adv[i] = g
        return adv[:self.size] if self.size < self.C else adv


class PPOTrainer:
    def __init__(self, data, cfg, device, out_dir):
        self.data, self.c, self.dev, self.out = data, cfg, device, out_dir
        self.env = PortfolioEnv(data, cfg, *segment(data, 'train', cfg['T']), random_start=True)
        self.U, self.F = self.env.U, self.env.n_features
        self.model = ActorCritic(self.F, self.U, cfg).to(device)
        xavier_init(self.model)
        calibrate_bn(self.model, self.env, device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.buf = RolloutBuffer(cfg['buffer_size'], (self.U, cfg['T'], self.F), self.U + 1)
        assert cfg['B'] <= cfg['buffer_size'], '批量 B 不能大于缓存区容量 C'

        # 市场波动率：基准指数日对数收益的滚动标准差 σ_t
        r = np.r_[0.0, np.diff(np.log(data['bench']))]
        self.sigma_mkt = pd.Series(r).rolling(cfg['vol_win']).std().bfill().values

        self.table3 = param_table(self.model)
        log.info('表3 核心架构参数数量：\n' + self.table3.to_string(index=False))

    # ---------------- 工具 ----------------
    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def _train_mode(self):
        self.model.train()
        self.model.enc.bn.eval()              # BatchNorm 统计量保持冻结

    @torch.no_grad()
    def _value(self, obs):
        return self.model(self._t(obs)[None])[2].item()

    # ---------------- 4. PPO-Clip 优化 ----------------
    def update(self, eps):
        c, b = self.c, self.buf
        adv = b.gae(c['gamma'], c['lam'])
        n = b.size
        ret = adv + b.val[:n]                                      # V^target = M̂ + V_old(s)
        if c['adv_norm']:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = []
        for _ in range(c['E']):
            ix = np.random.choice(n, c['B'], replace=False)
            obs, act = self._t(b.obs[ix]), self._t(b.act[ix])
            noise = self._t(b.noise[ix])[:, None]
            old_lp = Normal(self._t(b.mu[ix]), self._t(b.sigma[ix]) + noise).log_prob(act).sum(-1)
            mu, sigma, v = self.model(obs)
            dist = Normal(mu, sigma + noise)
            lp = dist.log_prob(act).sum(-1)
            ratio = torch.exp(lp - old_lp)
            a = self._t(adv[ix])
            l_pol = -torch.min(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a).mean()
            l_val = ((v - self._t(ret[ix])) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = l_pol + c['value_coef'] * l_val - c['entropy_coef'] * ent
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), c['max_grad_norm'])
            self.opt.step()
            with torch.no_grad():
                stats.append([l_pol.item(), l_val.item(), ent.item(), (old_lp - lp).mean().item(),
                              ((ratio - 1).abs() > eps).float().mean().item()])
        return dict(zip(['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_frac'],
                        np.mean(stats, axis=0)))

    # ---------------- 5. 自适应参数 ----------------
    def adapt_eps(self):
        """ε_t = ε0·(1 + 0.5·σ_t / σ_{t,old_W})"""
        c, t = self.c, self.env.t
        sig_w = self.sigma_mkt[max(0, t - c['W']):t].mean() if t > 0 else self.sigma_mkt[0]
        eps = c['eps0'] * (1 + 0.5 * self.sigma_mkt[t] / (sig_w + 1e-12))
        return float(np.clip(eps, *c['eps_bounds']))

    def adapt_lr(self, L, L_prev):
        """η_t = η0·exp(−α·ΔL_policy / L_policy)"""
        c = self.c
        rel = np.clip((L - L_prev) / (abs(L) + 1e-8), -5, 5)
        lr = float(np.clip(c['lr'] * math.exp(-c['alpha'] * rel), *c['lr_bounds']))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr

    # ---------------- 验证 ----------------
    def evaluate(self, name='val'):
        self.model.eval()
        nav, *_ = run_backtest(self.data, self.c, model_policy(self.model, self.dev), name)
        self._train_mode()
        return metrics(nav, self.c['rf'])

    # ---------------- 主训练循环 ----------------
    def train(self):
        c = self.c
        self._train_mode()
        obs, pending = self.env.reset(), None
        eps, lr = c['eps0'], c['lr']
        n_updates, last = 0, {}
        L_hist, ep_values, logs = [], [], []
        best, t0 = -np.inf, time.time()

        for k in range(1, c['K'] + 1):
            # 1-2. 前向传播
            with torch.no_grad():
                mu, sigma, v = self.model(self._t(obs)[None])
            v = v.item()
            if pending is not None:
                self.buf.next_val[pending] = v
            # 3. 课程噪声注入与动作采样
            noise = c['sigma0'] * math.exp(-c['beta'] * n_updates)
            mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()
            a = mu + (sigma + noise) * np.random.randn(self.U + 1).astype(np.float32)
            obs2, r, done, info = self.env.step(to_weights(a))
            idx = self.buf.add(obs, a, mu, sigma, noise, r, v, done)
            if done:
                self.buf.next_val[idx] = self._value(obs2)
                ep_values.append(info['V'])
                obs, pending = self.env.reset(), None
            else:
                obs, pending = obs2, idx

            # 4. PPO-Clip 优化
            if k % c['update_interval'] == 0 and self.buf.size >= c['B']:
                if pending is not None:
                    self.buf.next_val[pending] = self._value(obs)
                last = self.update(eps)
                n_updates += 1
                L_hist.append(last['policy_loss'])
                if n_updates % c['eval_interval'] == 0:
                    m = self.evaluate('val')
                    log.info(f'[验证] update {n_updates} 夏普={m["夏普比率"]:.3f} 年化={m["年化收益"]:.2%} '
                             f'回撤={m["最大回撤"]:.2%}')
                    last.update(val_sharpe=m['夏普比率'], val_return=m['年化收益'])
                    if m['夏普比率'] > best:
                        best = m['夏普比率']
                        self.save(self.out / 'best.pt', extra=dict(update=n_updates, val=m))

            # 5. 自适应参数调整
            if k % c['adapt_interval'] == 0:
                eps = self.adapt_eps()
                if len(L_hist) >= 2:
                    lr = self.adapt_lr(L_hist[-1], L_hist[-2])
                logs.append(dict(step=k, update=n_updates, eps=eps, lr=lr, noise=noise,
                                 reward=float(np.mean(self.buf.rew[:self.buf.size])),
                                 episode_value=np.mean(ep_values[-10:]) if ep_values else np.nan, **last))
                last = {}
                if k % (c['adapt_interval'] * 10) == 0:
                    L = logs[-1]
                    log.info(f'step {k}/{c["K"]} update={n_updates} reward={L["reward"]:+.4f} '
                             f'L_pol={L.get("policy_loss", np.nan):+.4f} L_val={L.get("value_loss", np.nan):.4f} '
                             f'kl={L.get("approx_kl", np.nan):.4f} eps={eps:.3f} lr={lr:.2e} noise={noise:.3f} '
                             f'近10回合终值={L["episode_value"]:.4f} 用时={time.time() - t0:.0f}s')

        self.save(self.out / 'last.pt', extra=dict(update=n_updates))
        if not (self.out / 'best.pt').exists():
            self.save(self.out / 'best.pt', extra=dict(update=n_updates))
        return pd.DataFrame(logs)

    def save(self, path, extra=None):
        torch.save({'model': self.model.state_dict(), 'cfg': copy.deepcopy(self.c), 'F': self.F, 'U': self.U,
                    'codes': self.data['codes'], 'features': self.data['features'], **(extra or {})}, path)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ck['F'], ck['U'], ck['cfg']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck
