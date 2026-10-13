"""PPO-Clip 训练（算法1 主训练循环）。

for k = 1..K:
    1-2. 状态特征提取，策略网络/价值网络前向传播
    3.   课程噪声注入与动作采样：a_t = μ_t + (σ_t + σ_noise)⊙ε_t，A_t = clip(a_t,0,1)/Σ
    4.   若 |D| ≥ B：计算 GAE，E 轮从 D 中采样批量 B，更新 L_total = L_policy + 0.5·L_value
    5.   若 k mod 100 = 0：自适应学习率 η_t；裁剪参数 ε 按样本所处交易日的市场波动率自适应

训练稳定性处理（均可在配置中关闭，见 configs/default.yaml 的 rl 段）：
  reward_norm      奖励按折扣回报的滑动标准差缩放，避免价值损失主导共享编码器的梯度
  每轮更新开始时用本轮更新前的网络重算缓存区的 V(s)、V(s')，GAE 不使用采样时的过期估值
  clip_center      缓存区样本来自多个历史策略。current：以本轮更新前的策略 π_k 为裁剪中心，
                   并用截断重要性权重 π_k/π_b 修正行为策略偏差（GePPO, Queeney et al. 2021）；
                   behavior：直接以采样时的行为策略为中心（算法1字面实现）
  target_kl        本轮更新中 KL(π_k‖π_θ) 超过 1.5×target_kl 时提前停止
  logit_l2         μ 的 logit 的 L2 约束：sigmoid 无法输出 0，压低某只股票权重会把 logit 推向 −∞，
                   饱和后梯度消失、策略冻结
  weight_decay     AdamW 权重衰减，抑制模型记忆训练期噪声
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
LOG_RATIO_CLIP = 20.0      # log 概率比的截断范围，e^20 ≈ 4.9e8


class RunningStd:
    """折扣回报 G_t = γG_{t-1} + r_t 的滑动方差，用于奖励缩放。"""

    def __init__(self, gamma):
        self.gamma, self.G, self.n, self.mean, self.M2 = gamma, 0.0, 0, 0.0, 0.0

    def update(self, r, end):
        self.G = self.gamma * self.G + r
        self.n += 1
        d = self.G - self.mean
        self.mean += d / self.n
        self.M2 += d * (self.G - self.mean)
        if end:
            self.G = 0.0

    @property
    def std(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 1.0


class RolloutBuffer:
    """采样缓存区 D（容量 C，先进先出）：(s, a, R, s', μ_b, σ_b, σ_noise, 交易日 t, 回合结束标记)。"""

    def __init__(self, capacity, obs_shape, act_dim):
        C = capacity
        self.C, self.ptr, self.size = C, 0, 0
        self.obs = np.zeros((C,) + tuple(obs_shape), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.act, self.mu, self.sigma = (np.zeros((C, act_dim), np.float32) for _ in range(3))
        self.noise, self.rew, self.val = (np.zeros(C, np.float32) for _ in range(3))
        self.t = np.zeros(C, np.int64)
        self.end = np.zeros(C, bool)

    def add(self, obs, act, mu, sigma, noise, rew, val, next_obs, t, end):
        i = self.ptr
        self.obs[i], self.act[i], self.mu[i], self.sigma[i] = obs, act, mu, sigma
        self.noise[i], self.rew[i], self.val[i] = noise, rew, val
        self.next_obs[i], self.t[i], self.end[i] = next_obs, t, end
        self.ptr, self.size = (i + 1) % self.C, min(self.size + 1, self.C)

    def order(self):
        """缓存区样本的时间顺序（环形缓冲）。"""
        return (np.arange(self.size) + self.ptr - self.size) % self.C


def gae(order, rew, val, next_val, end, gamma, lam):
    """δ_i = R_i + γV'_i − V_i，M̂_i = Σ_l (γλ)^l δ_{i+l}；按时间顺序计算，在回合边界截断。
    回合结束均为时间截断（市场不会终止），因此 δ 始终用 V(s') 自举。"""
    adv, g = np.zeros_like(rew), 0.0
    for i in order[::-1]:
        delta = rew[i] + gamma * next_val[i] - val[i]
        g = delta + gamma * lam * (0.0 if end[i] else 1.0) * g
        adv[i] = g
    return adv


class PPOTrainer:
    def __init__(self, data, cfg, device, out_dir):
        self.data, self.c, self.dev, self.out = data, cfg, device, out_dir
        self.env = PortfolioEnv(data, cfg, *segment(data, 'train', cfg['T']), random_start=True)
        self.U, self.F = self.env.U, self.env.n_features
        self.model = ActorCritic(self.F, self.U, cfg).to(device)
        xavier_init(self.model, cfg['sigma_init'])
        calibrate_bn(self.model, self.env, device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'],
                                     weight_decay=cfg.get('weight_decay', 0.0))
        self.buf = RolloutBuffer(cfg['buffer_size'], (self.U, cfg['T'], self.F), self.U + 1)
        self.rnorm = RunningStd(cfg['gamma'])
        self.nonfinite = 0
        assert cfg['B'] <= cfg['buffer_size'], '批量 B 不能大于缓存区容量 C'

        # 市场波动率 σ_t（基准指数日对数收益的滚动标准差）与逐日裁剪参数 ε_t = ε0·(1 + 0.5·σ_t/σ_{t,old_W})
        r = np.r_[0.0, np.diff(np.log(data['bench']))]
        sig = pd.Series(r).rolling(cfg['vol_win']).std().bfill()
        sig_w = sig.shift(1).rolling(cfg['W'], min_periods=1).mean().bfill()
        self.eps_t = np.clip(cfg['eps0'] * (1 + 0.5 * sig / sig_w), *cfg['eps_bounds']).values.astype(np.float32)

        self.table3 = param_table(self.model)
        log.info('表3 核心架构参数数量：\n' + self.table3.to_string(index=False))

    # ---------------- 工具 ----------------
    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def _train_mode(self):
        self.model.train()
        self.model.enc.bn.eval()              # BatchNorm 统计量保持冻结

    @torch.no_grad()
    def _forward_all(self, obs, chunk=250):
        outs = [self.model(self._t(obs[i:i + chunk])) for i in range(0, len(obs), chunk)]
        return [torch.cat(x) for x in zip(*outs)]

    # ---------------- 4. PPO-Clip 优化 ----------------
    def update(self):
        c, b, n = self.c, self.buf, self.buf.size
        act, noise = self._t(b.act[:n]), self._t(b.noise[:n])[:, None]
        logp_b = Normal(self._t(b.mu[:n]), self._t(b.sigma[:n]) + noise).log_prob(act).sum(-1)

        # 本轮更新前的策略 π_k 与价值估计 V_old
        mu_k, sig_k, v_k = self._forward_all(b.obs[:n])
        logp_k = Normal(mu_k, sig_k + noise).log_prob(act).sum(-1)
        val = v_k.cpu().numpy()
        next_val = self._forward_all(b.next_obs[:n])[2].cpu().numpy()
        scale = self.rnorm.std if c['reward_norm'] else 1.0
        adv = gae(b.order(), b.rew[:n] / scale, val, next_val, b.end[:n], c['gamma'], c['lam'])
        ret = self._t(adv + val)                                     # V^target = M̂ + V_old(s)
        ev = 1 - np.var(adv) / (np.var(adv + val) + 1e-8)           # 价值网络解释方差
        adv = self._t((adv - adv.mean()) / (adv.std() + 1e-8) if c['adv_norm'] else adv)
        eps = self._t(self.eps_t[b.t[:n]])                           # 逐样本自适应裁剪参数
        w_is = torch.exp(logp_k - logp_b).clamp(max=c['is_clip'])   # 截断重要性权重 π_k/π_b

        stats, epochs = [], 0
        if not torch.isfinite(logp_b).all() or not torch.isfinite(logp_k).all():
            raise FloatingPointError('缓存区或当前策略的 log 概率出现非有限值，请检查数据与模型参数')
        for _ in range(c['E']):
            ix = torch.as_tensor(np.random.choice(n, c['B'], replace=False), device=self.dev)
            mu, sigma, v = self.model(self._t(b.obs[ix.cpu().numpy()]))
            dist = Normal(mu, sigma + noise[ix])
            lp = dist.log_prob(act[ix]).sum(-1)
            a, e = adv[ix], eps[ix]
            if c['clip_center'] == 'current':
                # log ρ 截断后再取指数：旧样本在 π_k 下的 log 概率可低至 −500，参数稍有变化 ρ 即溢出为 inf，
                # 而其重要性权重又下溢为 0，0×inf 会产生 NaN 并污染全部参数
                log_rho = (lp - logp_k[ix]).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
                rho, w = log_rho.exp(), w_is[ix]
                surr = torch.min(rho * a, torch.clamp(rho, 1 - e, 1 + e) * a)
                l_pol = -(w * surr).sum() / w.sum().clamp_min(1e-8)
            else:
                log_rho = (lp - logp_b[ix]).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
                rho, w = log_rho.exp(), torch.ones_like(lp)
                l_pol = -torch.min(rho * a, torch.clamp(rho, 1 - e, 1 + e) * a).mean()
            l_val = ((v - ret[ix]) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = l_pol + c['value_coef'] * l_val - c['entropy_coef'] * ent
            if c.get('logit_l2'):                 # 抑制 μ 的 logit 发散，避免 sigmoid 饱和后梯度消失
                loss = loss + c['logit_l2'] * (self.model.mu_logit ** 2).mean()
            self.opt.zero_grad()
            if not torch.isfinite(loss):          # 兜底：跳过非有限损失，避免参数被 NaN 污染
                self.nonfinite += 1
                log.warning(f'第 {self.nonfinite} 次出现非有限损失，已跳过该步更新')
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), c['max_grad_norm'])
            self.opt.step()
            epochs += 1
            with torch.no_grad():
                # KL(π_k‖π_θ) 的非负估计 E[(r−1) − log r]，r = π_θ/π_k；样本来自行为策略，按 π_k/π_b 加权
                log_r = (lp - logp_k[ix]).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
                wk = w_is[ix]
                kl = ((wk * (log_r.exp() - 1 - log_r)).sum() / wk.sum().clamp_min(1e-8)).item()
                stats.append([l_pol.item(), l_val.item(), ent.item(), kl,
                              ((rho - 1).abs() > e).float().mean().item()])
            if c['target_kl'] and kl > 1.5 * c['target_kl']:
                break
        if not stats:
            return dict(epochs=0)
        out = dict(zip(['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_frac'], np.mean(stats, 0)))
        out.update(epochs=epochs, explained_var=float(ev), eps=float(eps.mean()),
                   is_weight=float(w_is.mean()), sigma=float(sig_k.mean()))
        return out

    # ---------------- 5. 自适应学习率 ----------------
    def adapt_lr(self, L_now, L_prev, L_scale):
        """η_t = η0·exp(−α·ΔL_policy / L_policy)；L 取各轮策略损失的指数滑动平均，分母取 |L| 的滑动平均。"""
        c = self.c
        rel = np.clip((L_now - L_prev) / (L_scale + 1e-8), -2, 2)
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
        obs = self.env.reset()
        lr, n_updates, last = c['lr'], 0, {}
        L_ema = L_abs = L_prev = None
        ep_values, logs = [], []
        best, bad_evals, t0 = -np.inf, 0, time.time()

        for k in range(1, c['K'] + 1):
            # 1-2. 前向传播
            with torch.no_grad():
                mu, sigma, v = self.model(self._t(obs)[None])
            # 3. 课程噪声注入与动作采样（噪声强度按已完成的更新轮次衰减）
            noise = c['sigma0'] * math.exp(-c['beta'] * n_updates)
            mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()
            a = mu + (sigma + noise) * np.random.randn(self.U + 1).astype(np.float32)
            t = self.env.t
            obs2, r, done, info = self.env.step(to_weights(a))
            self.rnorm.update(r, done)
            self.buf.add(obs, a, mu, sigma, noise, r, v.item(), obs2, t, done)
            if done:
                ep_values.append(info['V'])
                obs = self.env.reset()
            else:
                obs = obs2

            # 4. PPO-Clip 优化
            if k % c['update_interval'] == 0 and self.buf.size >= c['B']:
                last = self.update()
                n_updates += 1
                L = last['policy_loss']
                L_ema = L if L_ema is None else c['lr_ema'] * L_ema + (1 - c['lr_ema']) * L
                L_abs = abs(L) if L_abs is None else c['lr_ema'] * L_abs + (1 - c['lr_ema']) * abs(L)
                if n_updates % c['eval_interval'] == 0:
                    m = self.evaluate('val')
                    last.update(val_sharpe=m['夏普比率'], val_return=m['年化收益'])
                    if m['夏普比率'] > best:
                        best, bad_evals = m['夏普比率'], 0
                        self.save(self.out / 'best.pt', extra=dict(update=n_updates, step=k, val=m))
                    else:
                        bad_evals += 1
                    log.info(f'[验证] update {n_updates} 夏普={m["夏普比率"]:.3f} 年化={m["年化收益"]:.2%} '
                             f'回撤={m["最大回撤"]:.2%}（最优 {best:.3f}）')
                    if c['patience'] and bad_evals >= c['patience']:
                        log.info(f'验证集夏普连续 {bad_evals} 次未提升，提前停止')
                        break

            # 5. 自适应学习率（每 adapt_interval 步）
            if k % c['adapt_interval'] == 0:
                if L_ema is not None and L_prev is not None:
                    lr = self.adapt_lr(L_ema, L_prev, L_abs)
                L_prev = L_ema
                logs.append(dict(step=k, update=n_updates, lr=lr, noise=noise, reward=float(np.mean(
                    self.buf.rew[:self.buf.size])), episode_value=np.mean(ep_values[-10:]) if ep_values else np.nan,
                    **last))
                last = {}
                if k % (c['adapt_interval'] * 10) == 0:
                    L = logs[-1]
                    g = lambda key, f='.4f': format(L.get(key, np.nan), f)
                    log.info(f'step {k}/{c["K"]} update={n_updates} reward={g("reward", "+.4f")} '
                             f'L_pol={g("policy_loss", "+.4f")} L_val={g("value_loss")} EV={g("explained_var", ".2f")} '
                             f'kl={g("approx_kl")} clip={g("clip_frac", ".2f")} eps={g("eps", ".3f")} lr={lr:.2e} '
                             f'noise={noise:.3f} σ={g("sigma", ".3f")} 近10回合终值={g("episode_value")} '
                             f'用时={time.time() - t0:.0f}s')

        self.save(self.out / 'last.pt', extra=dict(update=n_updates, step=k))
        if not (self.out / 'best.pt').exists():
            self.save(self.out / 'best.pt', extra=dict(update=n_updates, step=k))
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
