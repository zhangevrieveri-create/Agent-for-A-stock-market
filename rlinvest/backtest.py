"""回测与评价指标：PPO 策略、等权再平衡、等权买入持有、基准指数。"""
import numpy as np
import pandas as pd
import torch

from .rl.env import PortfolioEnv, segment, to_weights


def run_backtest(data, cfg, policy, name='test'):
    """在指定数据段逐日调仓。policy(obs, env) -> 权重。返回净值、每日目标权重、每日换手与成本。"""
    lo, hi = segment(data, name, cfg['T'])
    env = PortfolioEnv(data, cfg, lo, hi)
    obs, done = env.reset(), False
    nav, weights, trade = [1.0], [], []
    while not done:
        obs, _, done, info = env.step(policy(obs, env))
        nav.append(info['V'])
        weights.append(info['weights'])
        trade.append((info['turnover'], info['cost']))
    dates = data['dates'][lo:hi + 1]
    nav = pd.Series(nav, index=dates, name='nav')
    weights = pd.DataFrame(weights, index=dates[:-1], columns=list(data['codes']) + ['cash'])
    trade = pd.DataFrame(trade, index=dates[:-1], columns=['turnover', 'cost'])
    return nav, weights, trade


def model_policy(model, device):
    @torch.no_grad()
    def f(obs, env):
        mu, _, _ = model(torch.as_tensor(obs, device=device)[None])
        return to_weights(mu[0].cpu().numpy())     # 评估时使用确定性动作 μ
    return f


def equal_weight_policy(U):
    w = np.append(np.ones(U) / U, 0.0)
    return lambda obs, env: w


def buy_and_hold_policy(U):
    w = np.append(np.ones(U) / U, 0.0)
    return lambda obs, env: w if env.t == env.lo else env.w


def benchmark_nav(data, cfg, name='test'):
    lo, hi = segment(data, name, cfg['T'])
    b = data['bench'][lo:hi + 1]
    return pd.Series(b / b[0], index=data['dates'][lo:hi + 1], name='benchmark')


def metrics(nav: pd.Series, rf=0.01, bench: pd.Series = None, trade: pd.DataFrame = None):
    r = nav.pct_change().dropna()
    n = len(r)
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (1 - nav / nav.cummax()).max()
    down = r[r < 0].std() * np.sqrt(252)
    out = {'累计收益': nav.iloc[-1] - 1, '年化收益': ann, '年化波动': vol,
           '夏普比率': (ann - rf) / (vol + 1e-12), '索提诺比率': (ann - rf) / (down + 1e-12),
           '最大回撤': mdd, '卡玛比率': ann / (mdd + 1e-12), '日胜率': (r > 0).mean()}
    if bench is not None:
        ex = r - bench.pct_change().dropna().reindex(r.index).fillna(0)
        out['超额年化收益'] = ex.mean() * 252
        out['信息比率'] = ex.mean() / (ex.std() + 1e-12) * np.sqrt(252)
    if trade is not None:
        out['日均换手率'] = trade['turnover'].mean()
        out['累计交易成本'] = trade['cost'].sum()
    return out
