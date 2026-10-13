"""构建强化学习数据集：量价技术指标 X_t ⊕ 舆情指标 N_t -> S_t ∈ R^{D×U×F}。

npz 内容：
  X        (D,U,F) 训练集统计量标准化后的特征
  close    (D,U)   前复权收盘价（停牌日沿用前收盘）
  buy_ok   (D,U)   当日收盘可买入（未停牌且未涨停）
  sell_ok  (D,U)   当日收盘可卖出（未停牌且未跌停）
  bench    (D,)    基准指数收盘价
  dates, codes, features, val_idx, test_idx, mu, sd
"""
import numpy as np
import pandas as pd

from ..data import rq_api
from ..utils import get_logger
from .indicators import compute_indicators, to_model_features

log = get_logger()
WARMUP = 130          # MA120 等长窗口指标的预热期
LIMIT_TOL = 1e-3      # 收盘价距涨跌停价的相对容差


def select_universe(cfg, price, news_daily, trading_dates):
    u, d = cfg['dataset']['universe'], cfg['dataset']
    if u['method'] == 'codes':
        if len(u['codes']) == 0:
            raise ValueError('dataset.universe.method=codes 时需填写 codes 列表')
        return list(u['codes'])
    td = trading_dates[(trading_dates >= d['start_date']) & (trading_dates <= d['end_date'])]
    p = price[(price['date'] >= d['start_date']) & (price['date'] <= d['end_date'])]
    cover = p[p['volume'] > 0].groupby('order_book_id')['date'].nunique() / len(td)
    ok = cover[cover >= u['min_coverage']].index
    nd = news_daily[news_daily['order_book_id'].isin(ok)]
    rank_until = d.get(u.get('rank_until'), u.get('rank_until'))   # 可填日期或 val_date / test_date
    if rank_until:
        nd = nd[nd['date'] < pd.Timestamp(rank_until)]
    cnt = nd.groupby('order_book_id')['news_cnt'].sum()
    if len(cnt) < u['n']:
        raise ValueError(f'满足覆盖率要求且有新闻的股票只有 {len(cnt)} 只，少于 U={u["n"]}')
    return list(cnt.sort_values(ascending=False).head(u['n']).index)


def decay_fill(s: pd.Series, decay: float):
    """无新闻交易日情绪按 decay 衰减：s_t = decay · s_{t-1}"""
    out, last = np.zeros(len(s)), 0.0
    for i, x in enumerate(s.values):
        last = x if np.isfinite(x) else last * decay
        out[i] = last
    return pd.Series(out, index=s.index)


def _stock_panel(code, price, news_daily, rq_senti, dates, cfg):
    p = price[price['order_book_id'] == code].set_index('date').sort_index().reindex(dates)
    listed = p['close'].notna()
    traded = listed & (p['volume'].fillna(0) > 0)
    p[['open', 'high', 'low', 'close']] = p[['open', 'high', 'low', 'close']].ffill()
    for k in ['open', 'high', 'low']:                     # 停牌日 OHLC 取前收盘
        p[k] = p[k].where(traded, p['close'])
    p['volume'] = p['volume'].fillna(0)
    p['turnover'] = p['turnover'].where(traded, 0.0) if 'turnover' in p else 0.0

    f = to_model_features(compute_indicators(p))
    nd = news_daily[news_daily['order_book_id'] == code].set_index('date').reindex(dates)
    f['sina_senti'] = decay_fill(nd['sina_senti'], cfg['sentiment']['decay'])
    f['news_cnt'] = np.log1p(nd['news_cnt'].fillna(0))
    if rq_senti is not None:
        rs = rq_senti[rq_senti['order_book_id'] == code].set_index('date')['rq_senti']
        lag = cfg['dataset'].get('rq_senti_lag', 0)
        f['rq_senti'] = rs.reindex(dates).ffill().shift(lag).fillna(0).clip(-1, 1)
    else:
        f['rq_senti'] = 0.0

    up = p['limit_up'] if 'limit_up' in p else np.inf
    dn = p['limit_down'] if 'limit_down' in p else 0.0
    buy_ok = traded & ~(p['close'] >= up * (1 - LIMIT_TOL))
    sell_ok = traded & ~(p['close'] <= dn * (1 + LIMIT_TOL))
    return f, p['close'], buy_ok, sell_ok


def build_dataset(cfg):
    P, d = cfg['paths'], cfg['dataset']
    td = rq_api.load_trading_dates(P['raw'])
    news_daily = pd.read_parquet(P['interim'] / 'news_daily.parquet')
    price = rq_api.load_price(P['raw'])
    codes = select_universe(cfg, price, news_daily, td)
    price = price[price['order_book_id'].isin(codes)]
    rq_senti = rq_api.load_rq_sentiment(P['raw'], codes)
    if rq_senti is None:
        log.warning('未找到米筐情绪指数，rq_senti 特征置 0')
    log.info(f'投资域 U={len(codes)}：{codes}')

    dates = td[(td >= pd.Timestamp(d['start_date'])) & (td <= pd.Timestamp(d['end_date']))]
    names = list(d['technical']) + list(d['sentiment'])
    feats, closes, buys, sells = [], [], [], []
    for code in codes:
        f, c, b, s = _stock_panel(code, price, news_daily, rq_senti, dates, cfg)
        missing = set(names) - set(f.columns)
        if missing:
            raise KeyError(f'配置中的特征不存在：{missing}')
        feats.append(f[names].values)
        closes.append(c.values)
        buys.append(b.values)
        sells.append(s.values)

    X = np.stack(feats, axis=1)                        # (D,U,F)
    C = np.stack(closes, axis=1)
    BUY, SELL = np.stack(buys, axis=1), np.stack(sells, axis=1)
    first = max(WARMUP, int(np.argmax(np.isfinite(C).all(1))))
    X, C, BUY, SELL, dates = X[first:], C[first:], BUY[first:], SELL[first:], dates[first:]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    bench = rq_api.load_benchmark(P['raw']).reindex(dates).ffill().bfill().values
    val_idx = int(dates.searchsorted(pd.Timestamp(d['val_date'])))
    test_idx = int(dates.searchsorted(pd.Timestamp(d['test_date'])))
    if not (cfg['rl']['T'] < val_idx < test_idx < len(dates)):
        raise ValueError(f'数据划分不合法：val_idx={val_idx}, test_idx={test_idx}, D={len(dates)}')

    mu, sd = X[:val_idx].mean((0, 1)), X[:val_idx].std((0, 1))   # 仅用训练集统计量，避免未来信息
    const = sd < 1e-8
    if const.any():
        log.warning(f'以下特征在训练集上为常数，标准化后为 0：{[n for n, k in zip(names, const) if k]}')
    X = np.clip((X - mu) / np.where(const, 1.0, sd), -5, 5).astype(np.float32)

    out = P['processed'] / f'{d["name"]}.npz'
    np.savez_compressed(out, X=X, close=C, buy_ok=BUY, sell_ok=SELL, bench=bench,
                        dates=np.array(dates.strftime('%Y-%m-%d'), dtype=str),
                        codes=np.array(codes, dtype=str), features=np.array(names, dtype=str),
                        val_idx=val_idx, test_idx=test_idx, mu=mu, sd=sd)
    log.info(f'数据集 {out.name}: X{X.shape}，训练 {val_idx} 天 / 验证 {test_idx - val_idx} 天 / '
             f'测试 {len(dates) - test_idx} 天，F={len(names)}')
    log.info(f'停牌或涨停不可买入比例 {1 - BUY.mean():.2%}，停牌或跌停不可卖出比例 {1 - SELL.mean():.2%}')
    return out


def load_dataset(path):
    z = np.load(path, allow_pickle=False)
    return dict(X=z['X'], close=z['close'], buy_ok=z['buy_ok'], sell_ok=z['sell_ok'], bench=z['bench'],
                dates=pd.to_datetime(z['dates']), codes=[str(c) for c in z['codes']],
                features=[str(f) for f in z['features']],
                val_idx=int(z['val_idx']), test_idx=int(z['test_idx']))
