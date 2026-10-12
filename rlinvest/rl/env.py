"""投资组合交易环境。

状态 s_t：过去 T 日的 S_t 窗口，形状 (U,T,F)
动作 A_t ∈ [0,1]^{m+1}：前 U 维为股票权重，最后一维为现金，Σ=1
在 t 日收盘按 A_t 调仓，获得 t→t+1 的收益；交易成本 τ·Σ|ΔA|
奖励 r_t = w1·短期收益 + w2·差分夏普比率 − w3·资本（回撤）惩罚
"""
import math
import random

import numpy as np


def segment(data, name, T):
    """返回数据段在时间轴上的 [lo, hi]：t∈[lo, hi) 上决策，使用 close[lo..hi]。"""
    D, v, s = len(data['close']), data['val_idx'], data['test_idx']
    return {'train': (T - 1, v - 1), 'val': (v - 1, s - 1), 'test': (s - 1, D - 1)}[name]


class PortfolioEnv:
    def __init__(self, data, cfg, lo, hi, random_start=False):
        self.X, self.P = data['X'], data['close']
        self.buy_ok, self.sell_ok = data['buy_ok'], data['sell_ok']
        self.c, self.lo, self.hi, self.random_start = cfg, lo, hi, random_start
        self.U, self.T = self.P.shape[1], cfg['T']
        self.n_features = self.X.shape[2]
        assert lo >= self.T - 1, '数据段起点不足一个历史窗口'

    def reset(self):
        if self.random_start:
            self.t = random.randint(self.lo, max(self.lo, self.hi - self.c['ep_len']))
            self.end = min(self.hi, self.t + self.c['ep_len'])
        else:
            self.t, self.end = self.lo, self.hi
        self.w = np.zeros(self.U + 1)
        self.w[-1] = 1.0                       # a_{-1}=0：初始全部为现金
        self.V, self.Vmax = 1.0, 1.0           # V_0^portfolio=1, V_max=1
        self.M, self.Bm = 0.0, 1e-4            # 差分夏普比率：收益一阶矩 M_0、二阶矩 B_0
        return self.obs()

    def obs(self):
        return self.X[self.t - self.T + 1:self.t + 1].transpose(1, 0, 2)    # (U,T,F)

    def apply_limits(self, w):
        """停牌不可交易、涨停不可买入、跌停不可卖出；受限部分维持原持仓，超出部分从增仓中按比例扣减。"""
        if not self.c['trade_limits']:
            return w
        old, new = self.w[:-1], w[:-1].copy()
        blocked = ((new > old) & ~self.buy_ok[self.t]) | ((new < old) & ~self.sell_ok[self.t])
        new[blocked] = old[blocked]
        excess = new.sum() - 1.0
        inc = np.clip(new - old, 0, None)
        if excess > 1e-10 and inc.sum() > 0:     # 受限持仓挤占的部分，从可交易的增仓中按比例扣减
            new -= inc * min(1.0, excess / inc.sum())
        new = np.clip(new, 0, None)
        return np.append(new, max(0.0, 1.0 - new.sum()))

    def step(self, w):
        c = self.c
        w = self.apply_limits(w)
        r = self.P[self.t + 1] / self.P[self.t] - 1
        growth = np.append(1 + r, 1 + c['rf'] / 252)
        turnover = np.abs(w[:-1] - self.w[:-1]).sum()
        cost = c['tau'] * turnover
        gross = float((w * growth).sum())
        V0 = self.V
        Vn = V0 * gross * (1 - cost)
        R = math.log(Vn / V0)

        # 差分夏普比率 DSR (Moody & Saffell, 2001)
        dM, dB = R - self.M, R * R - self.Bm
        var = self.Bm - self.M ** 2
        dsr = (self.Bm * dM - 0.5 * self.M * dB) / var ** 1.5 if var > 1e-10 else 0.0
        self.M += c['dsr_eta'] * dM
        self.Bm += c['dsr_eta'] * dB

        self.V, self.Vmax = Vn, max(self.Vmax, Vn)
        dd = 1 - self.V / self.Vmax
        reward = (c['w1'] * c['ret_scale'] * (R - c['rf'] / 252)
                  + c['w2'] * float(np.clip(dsr, -c['dsr_clip'], c['dsr_clip']))
                  - c['w3'] * c['dd_scale'] * dd)
        self.w = w * growth / gross            # 价格变动后的实际持仓
        self.t += 1
        info = dict(V=Vn, ret=Vn / V0 - 1, cost=cost, turnover=turnover, weights=w, dd=dd)
        return self.obs(), reward, self.t >= self.end, info


def to_weights(a, eps=1e-8):
    """Â = clip(a, 0, 1)，A = Â / (ΣÂ + ε)；全为 0 时全仓现金。"""
    w = np.clip(a, 0, 1)
    s = w.sum()
    if s < eps:
        w = np.zeros_like(w)
        w[-1] = 1.0
        return w
    return w / (s + eps)
