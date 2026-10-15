"""把本地米筐数据导出为仓库内的示例数据，并强制截断到截止日（默认 2024-12-31）。

数据授权只允许公开发布截至 2024-12-31 的历史数据，超出截止日的任何记录都不得进入仓库。
本脚本裁剪后会再逐个文件检查最大日期，发现超出截止日即报错。

python examples/rq4/export_sample.py --src data/rq4 --cutoff 2024-12-31
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DST = Path(__file__).resolve().parent / 'data'


def trim(df, col, cutoff):
    t = pd.to_datetime(df[col])
    return df[t <= cutoff + pd.Timedelta(hours=23, minutes=59, seconds=59)]


def export(src: Path, cutoff: pd.Timestamp):
    raw, out = src / 'raw', DST / 'raw'
    for sub in ['price', 'news_rq', 'rq_sentiment']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(raw / 'instruments.csv', out / 'instruments.csv')
    td = pd.read_csv(raw / 'trading_dates.csv')
    trim(td, 'date', cutoff).to_csv(out / 'trading_dates.csv', index=False)
    for f in ['benchmark.parquet', 'price/part_0000.parquet', 'rq_sentiment/part_0000.parquet']:
        trim(pd.read_parquet(raw / f), 'date', cutoff).to_parquet(out / f, index=False)
    for f in sorted((raw / 'news_rq').glob('*.parquet')):
        df = pd.read_parquet(f)
        df = trim(trim(df, 'datetime', cutoff), 'original_time', cutoff)
        df.to_parquet(out / 'news_rq' / f.name, index=False)
    (DST / 'processed').mkdir(parents=True, exist_ok=True)
    for f in (src / 'processed').glob('*.npz'):
        shutil.copy(f, DST / 'processed' / f.name)


def check(cutoff: pd.Timestamp):
    """逐个文件检查最大日期不超过截止日。"""
    limit = cutoff + pd.Timedelta(days=1)
    rows = []
    for f in sorted(DST.rglob('*')):
        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
            cols = [c for c in ['date', 'datetime', 'original_time'] if c in df]
            mx = max(pd.to_datetime(df[c]).max() for c in cols)
        elif f.name == 'trading_dates.csv':
            mx = pd.to_datetime(pd.read_csv(f)['date']).max()
        elif f.suffix == '.npz':
            mx = pd.to_datetime(np.load(f)['dates']).max()
        else:
            continue
        rows.append((str(f.relative_to(DST)), mx))
        if mx >= limit:
            raise ValueError(f'{f} 含有截止日 {cutoff.date()} 之后的数据（最大日期 {mx}），不得公开')
    return pd.DataFrame(rows, columns=['文件', '最大日期'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='data/rq4', help='本地数据目录（含 raw/ 与 processed/）')
    ap.add_argument('--cutoff', default='2024-12-31', help='允许公开的最后日期')
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    export(Path(args.src), cutoff)
    print(check(cutoff).to_string(index=False))
    print(f'检查通过：所有文件的日期均不晚于 {cutoff.date()}')
