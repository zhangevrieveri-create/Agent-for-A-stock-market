"""策略网络与价值网络（算法1 步骤1-2，结构与表3参数量一致）。

特征提取（策略网络与价值网络共享）：
  Ŝ_t      = BatchNorm(S_t)
  H^LSTM   = Bi-LSTM(Ŝ_t)                       ∈ R^{U×T×256}
  H^trans  = Transformer-Encoder(H^LSTM)        ∈ R^{U×T×256}
  C^feature= MHSA(H^trans)        （沿时间/特征视角） ∈ R^{U×T×256}
  C^asset  = MHSA(H^trans⊤)⊤      （沿标的视角）     ∈ R^{U×T×256}
  C_t      = [C^feature; C^asset]               ∈ R^{U×T×512}
  h_u      = (1/T) Σ_t C_t （逐标的时间池化）       ∈ R^{U×512}
  h^global = (1/U) Σ_u h_u                       ∈ R^{512}
决策输出：
  H_critic = GELU(W_c1 h^global + b_c1)，V = W_c2 H_critic + b_c2
  策略头 head=asset（默认）：H_actor,u = GELU(W_a1 h_u + b_a1)，
      μ_u = sigmoid(W_a2[u]·H_actor,u + b_a2[u])，现金 μ_{m+1} 使用 GELU(W_a1 h^global + b_a1)；
      W_a2 ∈ R^{(m+1)×128} 的第 u 行只作用于第 u 只标的，与表3参数量完全相同，且显式保留标的身份
  策略头 head=global：H_actor = GELU(W_a1 h^global + b_a1)，μ = sigmoid(W_a2 H_actor + b_a2)
  σ 与 μ 的计算方式相同（softplus），并以 σ_max·tanh(σ/σ_max) 平滑限制上限
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
        # 消融实验：dual_attention = both / feature / asset / none，关闭的分支以 H^trans 直通代替
        mode = c.get('dual_attention', 'both')
        mha = lambda: nn.MultiheadAttention(d, c['n_heads'], dropout=c['dropout'], batch_first=True)
        self.feat_att = mha() if mode in ('both', 'feature') else None
        self.asset_att = mha() if mode in ('both', 'asset') else None
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
        cf = self.feat_att(h, h, h)[0] if self.feat_att is not None else h          # C^feature
        ha = h.reshape(N, U, T, d).transpose(1, 2).reshape(N * T, U, d)
        ca = self.asset_att(ha, ha, ha)[0] if self.asset_att is not None else ha    # C^asset
        ca = ca.reshape(N, T, U, d).transpose(1, 2)
        C = torch.cat([cf.reshape(N, U, T, d), ca], -1)        # (N,U,T,2d)
        return C.mean(2)                                       # h_u: (N,U,2d)


class ActorCritic(nn.Module):
    def __init__(self, F, U, c):
        super().__init__()
        self.enc = Encoder(F, U, c)
        g, h, A = self.enc.out_dim, c['mlp_hidden'], U + 1
        self.head = c.get('head', 'asset')
        self.sigma_max = c.get('sigma_max')
        self.actor_hidden = nn.Linear(g, h)
        self.mu_head = nn.Linear(h, A)
        self.sigma_head = nn.Linear(h, A)
        self.critic_hidden = nn.Linear(g, h)
        self.value_head = nn.Linear(h, 1)

    @staticmethod
    def _asset_linear(layer, za, zc):
        """逐标的输出：第 u 行权重只作用于第 u 只标的的表示，最后一行作用于现金（全局）表示。"""
        W, b = layer.weight, layer.bias
        U = za.shape[1]
        stock = (za * W[:U]).sum(-1) + b[:U]                   # (N,U)
        cash = zc @ W[U] + b[U]                                # (N,)
        return torch.cat([stock, cash[:, None]], -1)

    @staticmethod
    def _shared_linear(layer, za, zc):
        """共享评估器（EIIE 式）：所有股票使用同一行权重，现金使用最后一行。"""
        W, b = layer.weight, layer.bias
        stock = za @ W[0] + b[0]                               # (N,U)
        cash = zc @ W[-1] + b[-1]
        return torch.cat([stock, cash[:, None]], -1)

    def forward(self, x):
        hu = self.enc(x)                                       # (N,U,512)
        g = hu.mean(1)                                         # h^global: (N,512)
        zc = Fn.gelu(self.actor_hidden(g))
        if self.head == 'asset':
            za = Fn.gelu(self.actor_hidden(hu))
            mu_logit = self._asset_linear(self.mu_head, za, zc)
            sd_logit = self._asset_linear(self.sigma_head, za, zc)
        elif self.head == 'shared':
            za = Fn.gelu(self.actor_hidden(hu))
            mu_logit = self._shared_linear(self.mu_head, za, zc)
            sd_logit = self._shared_linear(self.sigma_head, za, zc)
        else:
            mu_logit, sd_logit = self.mu_head(zc), self.sigma_head(zc)
        self.mu_logit = mu_logit                               # 供 logit 正则使用（防止 sigmoid 饱和）
        mu = torch.sigmoid(mu_logit)                           # μ ∈ [0,1]^{m+1}
        sigma = Fn.softplus(sd_logit)                          # σ ∈ R_+^{m+1}
        if self.sigma_max:   # 平滑上限：σ 较小时与 softplus 相同，接近 σ_max 时饱和，梯度不中断
            sigma = self.sigma_max * torch.tanh(sigma / self.sigma_max)
        sigma = sigma + 1e-3
        v = self.value_head(Fn.gelu(self.critic_hidden(g))).squeeze(-1)
        return mu, sigma, v


def xavier_init(model, sigma_init=0.1, out_gain=0.01):
    """所有权重矩阵 Xavier 均匀初始化，偏置置 0。
    策略输出层使用小增益（初始 μ≈0.5，即近似等权组合），σ 的偏置使初始标准差约为 sigma_init。"""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)
    for layer in (model.mu_head, model.sigma_head):
        nn.init.xavier_uniform_(layer.weight, gain=out_gain)
    nn.init.constant_(model.sigma_head.bias, math.log(math.expm1(sigma_init)))


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
    cnt = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
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
