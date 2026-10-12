"""米筐 RQData 数据接口：股票池、交易日历、日线量价（含涨跌停价）、换手率、基准指数、个股情绪指数。

结果以长表 (order_book_id, date, ...) 保存为 parquet，按块落盘，可断点续传。
"""
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..utils import get_logger

log = get_logger()
PRICE_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'limit_up', 'limit_down']


def init_rq(cfg):
    import rqdatac as rq
    user = cfg['rqdata'].get('username') or os.getenv('RQDATA_USERNAME')
    pwd = cfg['rqdata'].get('password') or os.getenv('RQDATA_PASSWORD')
    if user and pwd:
        rq.init(user, pwd)
    else:
        rq.init()
    return rq


def _to_long(df, value_cols=None):
    """把 rqdatac 返回的 (order_book_id, date) 多级索引表转为长表。"""
    df = df.reset_index()
    df = df.rename(columns={'tradedate': 'date', 'datetime': 'date', 'level_1': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    if value_cols:
        df = df[['order_book_id', 'date'] + value_cols]
    return df


def _chunks(codes, n):
    for i in range(0, len(codes), n):
        yield i // n, codes[i:i + n]


def _date_range(cfg):
    return cfg['rqdata']['start_date'], cfg['rqdata']['end_date']


# ============================ 下载 ============================
def fetch_instruments(rq, cfg, raw_dir: Path):
    """股票池：A 股普通股，按交易所、上市日期、是否 ST 筛选。"""
    u = cfg['rqdata']['universe']
    start, _ = _date_range(cfg)
    ins = rq.all_instruments(type='CS', market='cn')
    ins = ins[['order_book_id', 'symbol', 'listed_date', 'de_listed_date', 'exchange']].copy()
    if u.get('exclude_bj'):
        ins = ins[ins['order_book_id'].str.endswith(('.XSHG', '.XSHE'))]
    if u.get('exclude_st'):
        ins = ins[~ins['symbol'].str.contains('ST')]
    ins = ins[ins['listed_date'] <= u['listed_before']]
    alive = (ins['de_listed_date'] == '0000-00-00') | (ins['de_listed_date'] > start)
    ins = ins[alive].sort_values('order_book_id').reset_index(drop=True)
    if u.get('max_stocks'):
        ins = ins.head(u['max_stocks'])
    ins.to_csv(raw_dir / 'instruments.csv', index=False, encoding='utf-8-sig')
    log.info(f'股票池 {len(ins)} 只 -> instruments.csv')
    return ins


def fetch_trading_dates(rq, cfg, raw_dir: Path):
    start, end = _date_range(cfg)
    # 多取一个月，用于把样本末尾收盘后的新闻对齐到下一交易日
    end = (pd.Timestamp(end) + pd.Timedelta(days=31)).strftime('%Y-%m-%d')
    dates = pd.to_datetime(rq.get_trading_dates(start, end))
    pd.Series(dates, name='date').to_csv(raw_dir / 'trading_dates.csv', index=False)
    log.info(f'交易日 {len(dates)} 天 -> trading_dates.csv')
    return dates


def fetch_price(rq, cfg, codes, raw_dir: Path):
    """前复权日线量价 + 涨跌停价 + 换手率，按块保存到 raw/price/part_xxxx.parquet。"""
    out = raw_dir / 'price'
    out.mkdir(exist_ok=True)
    s, e = _date_range(cfg)
    for i, part in tqdm(list(_chunks(list(codes), cfg['rqdata']['chunk_size'])), desc='量价数据'):
        f = out / f'part_{i:04d}.parquet'
        if f.exists():
            continue
        px = rq.get_price(part, start_date=s, end_date=e, frequency='1d', fields=PRICE_FIELDS,
                          adjust_type='pre', expect_df=True)
        if px is None or px.empty:
            continue
        px = _to_long(px, PRICE_FIELDS)
        try:
            to = rq.get_turnover_rate(part, start_date=s, end_date=e, fields='today', expect_df=True)
            to = _to_long(to).rename(columns={'today': 'turnover'})[['order_book_id', 'date', 'turnover']]
            px = px.merge(to, on=['order_book_id', 'date'], how='left')
        except Exception as ex:  # 部分账户无换手率权限
            log.warning(f'换手率获取失败（该列置空）：{ex}')
            px['turnover'] = float('nan')
        px.to_parquet(f, index=False)


def fetch_benchmark(rq, cfg, raw_dir: Path):
    s, e = _date_range(cfg)
    code = cfg['rqdata']['benchmark']
    px = rq.get_price(code, start_date=s, end_date=e, frequency='1d', fields=['close'], expect_df=True)
    px = _to_long(px, ['close'])
    px.to_parquet(raw_dir / 'benchmark.parquet', index=False)
    log.info(f'基准指数 {code} {len(px)} 天 -> benchmark.parquet')
    return px


def fetch_rq_sentiment(rq, cfg, codes, raw_dir: Path):
    """米筐个股情绪指数：新闻标题、财务数据、行业政策等舆情信息的聚合因子，取值 [-1,1]。"""
    factor = cfg['rqdata'].get('sentiment_factor')
    if not factor:
        log.warning('未配置 rqdata.sentiment_factor，跳过米筐情绪指数（数据集中 rq_senti 置 0）')
        return None
    s, e = _date_range(cfg)
    out = raw_dir / 'rq_sentiment'
    out.mkdir(exist_ok=True)
    for i, part in tqdm(list(_chunks(list(codes), cfg['rqdata']['chunk_size'])), desc='米筐情绪指数'):
        f = out / f'part_{i:04d}.parquet'
        if f.exists():
            continue
        df = rq.get_factor(part, factor, start_date=s, end_date=e)
        if df is None or len(df) == 0:
            continue
        df = df.to_frame(factor) if isinstance(df, pd.Series) else df
        df = _to_long(df).rename(columns={factor: 'rq_senti'})
        df[['order_book_id', 'date', 'rq_senti']].to_parquet(f, index=False)


# ============================ 读取 ============================
def load_instruments(raw_dir: Path):
    return pd.read_csv(raw_dir / 'instruments.csv', dtype=str)


def load_trading_dates(raw_dir: Path):
    return pd.DatetimeIndex(pd.read_csv(raw_dir / 'trading_dates.csv', parse_dates=['date'])['date'])


def _load_parts(folder: Path, codes=None):
    files = sorted(folder.glob('part_*.parquet'))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df['order_book_id'].isin(codes)] if codes is not None else df


def load_price(raw_dir: Path, codes=None):
    df = _load_parts(raw_dir / 'price', codes)
    if df is None:
        raise FileNotFoundError(f'{raw_dir / "price"} 下没有行情数据，请先运行 01_fetch_market.py')
    return df


def load_benchmark(raw_dir: Path):
    return pd.read_parquet(raw_dir / 'benchmark.parquet').set_index('date')['close']


def load_rq_sentiment(raw_dir: Path, codes=None):
    return _load_parts(raw_dir / 'rq_sentiment', codes)
