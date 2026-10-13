"""步骤6：验证集与测试集回测，对比等权再平衡、等权买入持有与基准指数，输出指标表、净值、持仓与图表。"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rlinvest.backtest import (benchmark_nav, buy_and_hold_policy, equal_weight_policy, metrics,
                               model_policy, run_backtest)
from rlinvest.data.rq_api import load_instruments
from rlinvest.features.dataset import load_dataset
from rlinvest.plotting import plot_nav, plot_training, plot_weights
from rlinvest.rl.ppo import load_model
from rlinvest.utils import get_device, get_logger, load_config

SEG_NAME = {'val': '验证集', 'test': '测试集'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpt', default='best.pt', help='best.pt 或 last.pt')
    ap.add_argument('--tag', default=None, help='实验标签，结果保存到 outputs/<数据集名>_<tag>/')
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, name = cfg['paths'], cfg['dataset']['name']
    run = f'{name}_{args.tag}' if args.tag else name
    out = P['output'] / run
    log = get_logger(log_file=P['output'] / 'logs' / f'06_backtest_{run}.log')
    device = get_device()

    data = load_dataset(P['processed'] / f'{name}.npz')
    model, ck = load_model(out / args.ckpt, device)
    rl, U = ck['cfg'], len(data['codes'])
    log.info(f'加载 {args.ckpt}（第 {ck.get("update")} 轮更新）')
    ins = load_instruments(P['raw'])
    code2name = dict(zip(ins['order_book_id'], ins['symbol']))

    policies = {'PPO智能投资': model_policy(model, device),
                '等权再平衡': equal_weight_policy(U),
                '等权买入持有': buy_and_hold_policy(U)}
    for seg in ['val', 'test']:
        bench = benchmark_nav(data, rl, seg)
        navs, res = {}, {}
        for k, pol in policies.items():
            nav, weights, trade = run_backtest(data, rl, pol, seg)
            navs[k], res[k] = nav, metrics(nav, rl['rf'], bench, trade)
            if k == 'PPO智能投资':
                weights.to_csv(out / f'{seg}_weights.csv', encoding='utf-8-sig')
                ppo_weights = weights
        navs['基准指数'], res['基准指数'] = bench, metrics(bench, rl['rf'])
        res = pd.DataFrame(res)
        res.to_csv(out / f'{seg}_metrics.csv', encoding='utf-8-sig')
        pd.DataFrame(navs).to_csv(out / f'{seg}_nav.csv', encoding='utf-8-sig')
        log.info(f'【{SEG_NAME[seg]}】{bench.index[0]:%Y-%m-%d} ~ {bench.index[-1]:%Y-%m-%d}\n'
                 + res.round(4).to_string())
        plot_nav(navs, out / f'{seg}_nav.png', f'{SEG_NAME[seg]}净值曲线')
        plot_weights(ppo_weights.rename(columns=lambda c: code2name.get(c, c)),
                     out / f'{seg}_weights.png', f'{SEG_NAME[seg]}持仓权重')

    if (out / 'train_log.csv').exists():
        plot_training(pd.read_csv(out / 'train_log.csv'), out / 'train_curves.png')
    log.info(f'回测结果保存在 {out}')
