"""步骤3：合并爬取与历史新闻 -> 清洗去重 -> 交易日对齐 -> 标题情绪打分 -> 个股日度情绪；输出图1示例与表1。"""
import argparse

import _bootstrap  # noqa: F401
from rlinvest.data import rq_api
from rlinvest.data.news_clean import build_clean_news
from rlinvest.sentiment.scorer import daily_sentiment, get_scorer
from rlinvest.utils import get_logger, load_config

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = cfg['paths']
    log = get_logger(log_file=P['output'] / 'logs' / '03_build_sentiment.log')

    ins = rq_api.load_instruments(P['raw'])
    td = rq_api.load_trading_dates(P['raw'])
    news = build_clean_news(cfg, ins, td)
    news['score'] = get_scorer(cfg).score(news['title'].tolist())
    news.to_parquet(P['interim'] / 'news_scored.parquet', index=False)
    daily = daily_sentiment(news)
    daily.to_parquet(P['interim'] / 'news_daily.parquet', index=False)
    log.info(f'个股日度情绪 {len(daily):,} 条；标题正面占比 {(news["score"] > 0).mean():.2%}，'
             f'负面占比 {(news["score"] < 0).mean():.2%}')
