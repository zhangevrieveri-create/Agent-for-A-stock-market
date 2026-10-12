"""策略网络与价值网络（算法1 步骤1-2，结构与表3参数量一致）。

特征提取（策略网络与价值网络共享）：
  Ŝ_t      = BatchNorm(S_t)
  H^LSTM   = Bi-LSTM(Ŝ_t)                       ∈ R^{U×T×256}
  H^trans  = Transformer-Encoder(H^LSTM)        ∈ R^{U×T×256}
  C^feature= MHSA(H^trans)        （沿时间/特征视角） ∈ R^{U×T×256}
  C^asset  = MHSA(H^trans⊤)⊤      （沿标的视角）     ∈ R^{U×T×256}
  C_t      = [C^feature; C^asset]               ∈ R^{U×T×512}
  h^global = 时间与标的维度均值池化               ∈ R^{512}
决策输出：
  H_actor  = GELU(W_a1 h + b_a1)，μ = sigmoid(W_a2 H_actor + b_a2)，σ = softplus(W_a3 H_actor + b_a3)
  H_critic = GELU(W_c1 h + b_c1)，V = W_c2 H_critic + b_c2
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def sinusoidal_encoding(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Encoder(nn.Module):
    def __init__(self, F, U, c):
        super().__init__()
        d = 2 * c['lstm_hidden']
        # BatchNorm 不带仿射参数，仅做状态标准化（表3 不含该部分参数）
        self.bn = nn.BatchNorm1d(F, affine=False)
        self.lstm = nn.LSTM(F, c['lstm_hidden'], c['lstm_layers'], batch_first=True, bidirectional=True)
        layer = nn.TransformerEncoderLayer(d, c['n_heads'], c['ff_dim'], dropout=c['dropout'],
                                           activation='gelu', batch_first=True)
        self.trans = nn.TransformerEncoder(layer, c['n_layers'], enable_nested_tensor=False)
        self.feat_att = nn.MultiheadAttention(d, c['n_heads'], dropout=c['dropout'], batch_first=True)
        self.asset_att = nn.MultiheadAttention(d, c['n_heads'], dropout=c['dropout'], batch_first=True)
        # 固定的标的位置编码（无可训练参数）：池化前为各标的打上身份标记，
        # 使 h^global 到 m+1 维权重的映射能够区分不同标的
        pe = sinusoidal_encoding(U, d) if c['asset_encoding'] else torch.zeros(U, d)
        self.register_buffer('asset_pe', pe, persistent=False)
        self.out_dim = 2 * d

    def forward(self, x):                                      # x: (N,U,T,F)
        N, U, T, F = x.shape
        x = self.bn(x.reshape(-1, F)).reshape(N * U, T, F)
        h, _ = self.lstm(x)                                    # (N·U,T,d)
        d = h.shape[-1]
        h = (h.reshape(N, U, T, d) + self.asset_pe[None, :, None, :]).reshape(N * U, T, d)
        h = self.trans(h)
        cf, _ = self.feat_att(h, h, h)                         # C^feature
        ha = h.reshape(N, U, T, d).transpose(1, 2).reshape(N * T, U, d)
        ca, _ = self.asset_att(ha, ha, ha)                     # C^asset
        ca = ca.reshape(N, T, U, d).transpose(1, 2)
        C = torch.cat([cf.reshape(N, U, T, d), ca], -1)        # (N,U,T,2d)
        return C.mean(dim=(1, 2))                              # h^global: (N,2d)


class ActorCritic(nn.Module):
    def __init__(self, F, U, c):
        super().__init__()
        self.enc = Encoder(F, U, c)
        g, h, A = self.enc.out_dim, c['mlp_hidden'], U + 1
        self.actor_hidden = nn.Linear(g, h)
        self.mu_head = nn.Linear(h, A)
        self.sigma_head = nn.Linear(h, A)
        self.critic_hidden = nn.Linear(g, h)
        self.value_head = nn.Linear(h, 1)

    def forward(self, x):
        g = self.enc(x)
        ha = Fn.gelu(self.actor_hidden(g))
        mu = torch.sigmoid(self.mu_head(ha))                   # μ ∈ [0,1]^{m+1}
        sigma = Fn.softplus(self.sigma_head(ha)) + 1e-4        # σ ∈ R_+^{m+1}
        hc = Fn.gelu(self.critic_hidden(g))
        v = self.value_head(hc).squeeze(-1)
        return mu, sigma, v


def xavier_init(model):
    """所有权重矩阵 Xavier 均匀初始化，偏置置 0。"""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)


@torch.no_grad()
def calibrate_bn(model, env, device, n=512):
    """在训练段随机状态上估计 BatchNorm 统计量后冻结，保证 PPO 新旧策略概率计算一致。"""
    obs = []
    for _ in range(n):
        env.reset()
        obs.append(env.obs())
    bn = model.enc.bn
    bn.reset_running_stats()
    bn.momentum = None                     # 累计平均
    model.train()
    for chunk in np.array_split(np.array(obs), 8):
        model(torch.as_tensor(chunk, device=device))
    bn.eval()


def param_table(model: ActorCritic):
    """表3：核心架构参数数量。"""
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    e = model.enc
    lstm_b = sum(p.numel() for n, p in e.lstm.named_parameters() if n.endswith('_reverse'))
    lstm_f = cnt(e.lstm) - lstm_b
    rows = [('特征提取', 'Bi-LSTM', '前向LSTM', lstm_f), ('特征提取', 'Bi-LSTM', '后向LSTM', lstm_b)]
    for i, layer in enumerate(e.trans.layers, 1):
        rows += [('特征提取', f'Transformer编码器层{i}', '多头注意力', cnt(layer.self_attn)),
                 ('特征提取', f'Transformer编码器层{i}', '前馈网络', cnt(layer.linear1) + cnt(layer.linear2)),
                 ('特征提取', f'Transformer编码器层{i}', '层归一化×2', cnt(layer.norm1) + cnt(layer.norm2))]
    rows += [('双重注意力', '双重注意力', '特征注意力', cnt(e.feat_att)),
             ('双重注意力', '双重注意力', '标的注意力', cnt(e.asset_att)),
             ('决策输出层', '策略网络', '隐藏层', cnt(model.actor_hidden)),
             ('决策输出层', '策略网络', '均值输出', cnt(model.mu_head)),
             ('决策输出层', '策略网络', '标准差输出', cnt(model.sigma_head)),
             ('决策输出层', '价值网络', '隐藏层', cnt(model.critic_hidden)),
             ('决策输出层', '价值网络', '价值输出', cnt(model.value_head))]
    df = pd.DataFrame(rows, columns=['主架构', '模块', '子架构', '参数数量'])
    total = cnt(model)
    assert df['参数数量'].sum() == total
    df['百分比'] = (df['参数数量'] / total).map('{:.2%}'.format)
    df.loc[len(df)] = ['模型总计', '', '', total, '100.00%']
    return df
