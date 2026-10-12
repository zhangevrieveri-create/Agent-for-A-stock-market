"""步骤5：PPO-Clip 训练（算法1），输出模型、训练日志、表3参数统计与表4超参数表。"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rlinvest.features.dataset import load_dataset
from rlinvest.rl.ppo import PPOTrainer
from rlinvest.utils import get_device, get_logger, load_config, set_seed

TABLE4 = [('训练参数', '初始学习率', 'η', 'lr'), ('训练参数', '折扣因子', 'γ', 'gamma'),
          ('训练参数', 'GAE参数', 'λ', 'lam'), ('训练参数', '初始裁剪参数', 'ϵ', 'eps0'),
          ('课程学习', '初始噪声', 'σ0', 'sigma0'), ('课程学习', '衰减系数', 'β', 'beta'),
          ('奖励权重', '短期收益权重', 'w1', 'w1'), ('奖励权重', '夏普比率权重', 'w2', 'w2'),
          ('奖励权重', '资本惩罚权重', 'w3', 'w3'), ('环境参数', '交易成本率', 'τ', 'tau'),
          ('环境参数', '无风险利率', 'r_f', 'rf')]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, name, rl = cfg['paths'], cfg['dataset']['name'], cfg['rl']
    out = P['output'] / name
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=P['output'] / 'logs' / f'05_train_{name}.log')
    set_seed(rl['seed'])
    device = get_device()
    log.info(f'设备 {device}')

    data = load_dataset(P['processed'] / f'{name}.npz')
    trainer = PPOTrainer(data, rl, device, out)
    trainer.table3.to_csv(out / 'table3_params.csv', index=False, encoding='utf-8-sig')
    table4 = pd.DataFrame([(a, b, s, rl[k]) for a, b, s, k in TABLE4], columns=['类别', '参数名称', '符号', '值'])
    table4.to_csv(out / 'table4_hyperparams.csv', index=False, encoding='utf-8-sig')

    logs = trainer.train()
    logs.to_csv(out / 'train_log.csv', index=False, encoding='utf-8-sig')
    log.info(f'训练完成，模型保存在 {out}（best.pt：验证集夏普最优；last.pt：最终模型）')
