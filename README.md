# RL-Invest：融合新闻舆情与量价数据的强化学习智能投资系统

新浪财经个股新闻情绪、米筐 RQData 个股情绪指数与量价技术指标拼接为状态 S_t，经
**Bi-LSTM → Transformer 编码器 → 双重视角注意力** 提取特征，由 **PPO-Clip**
（课程噪声、自适应裁剪阈值、自适应学习率）学习 A 股投资组合的调仓权重。

## 项目结构

```
rl_invest/
├── configs/default.yaml        全部参数（标注了论文表4/算法1/表3的出处）
├── rlinvest/
│   ├── data/
│   │   ├── rq_api.py           米筐：股票池、交易日历、量价+涨跌停价+换手率、基准指数、情绪指数
│   │   ├── sina_news.py        新浪财经爬虫：按个股名称搜索 + 个股新闻列表页，断点续爬
│   │   └── news_clean.py       【历史新闻导入位置】、清洗去重、交易日对齐、图1示例、表1统计
│   ├── sentiment/scorer.py     标题情绪打分（金融词典+否定词 / 预训练模型），日度聚合
│   ├── features/
│   │   ├── indicators.py       表2 技术指标
│   │   └── dataset.py          投资域选择、量价与舆情融合、训练集标准化、交易约束掩码
│   ├── rl/
│   │   ├── env.py              交易环境：交易成本、差分夏普比率、回撤惩罚、停牌/涨跌停约束
│   │   ├── model.py            策略网络与价值网络（共享特征提取），表3参数统计
│   │   └── ppo.py              算法1：采样缓存区、GAE、PPO-Clip、课程噪声、自适应参数
│   ├── backtest.py             回测与评价指标
│   ├── plotting.py             净值、持仓、训练曲线
│   └── utils.py
├── scripts/                    按顺序运行 01 → 06
├── pyproject.toml
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt      # 或 pip install -e .[rqdata]
```

米筐账号建议通过环境变量传入，不要写进配置文件提交：

```bash
export RQDATA_USERNAME=xxx RQDATA_PASSWORD=xxx      # Windows PowerShell: $env:RQDATA_USERNAME="xxx"
```

## 运行流程

| 步骤 | 命令 | 主要输出 |
|---|---|---|
| 1 米筐数据 | `python scripts/01_fetch_market.py` | `data/raw/` 股票池、交易日、`price/`、`benchmark.parquet`、`rq_sentiment/` |
| 2 新浪新闻 | `python scripts/02_crawl_news.py` | `data/raw/news/{代码}.csv` |
| 3 舆情情绪 | `python scripts/03_build_sentiment.py` | `data/interim/news_daily.parquet`、`news_sample.csv`（图1）、`table1_news_stats.csv`（表1） |
| 4 数据集 | `python scripts/04_build_dataset.py` | `data/processed/u10.npz` |
| 5 训练 | `python scripts/05_train.py` | `outputs/u10/best.pt`、`last.pt`、`train_log.csv`、`table3_params.csv`、`table4_hyperparams.csv` |
| 6 回测 | `python scripts/06_backtest.py` | `outputs/u10/{val,test}_metrics.csv`、净值、持仓、图表 |

每个脚本都支持 `--config`。做对比实验时，新建一个只写差异字段的配置即可：

```yaml
# configs/local_no_sentiment.yaml —— 去掉舆情特征的消融实验
base: default.yaml
dataset: {name: u10_no_senti, sentiment: []}
```

## 数据

**米筐 RQData**：样本期为 2014-01-06 至 2025-10-29。全市场普通股（默认排除北交所）的前复权日线
包括开高低收、成交量、成交额、涨跌停价和换手率，另取沪深300作为基准。个股情绪指数通过
`get_factor` 获取，因子名填在 `rqdata.sentiment_factor`，以你账户的数据文档为准；不填时该特征置 0。

**新浪财经新闻**：按个股名称调用新浪搜索接口，同时抓取个股新闻列表页。两路结果按链接去重，并且
只保留标题中出现股票名称的新闻。首页请求失败的股票不会写文件，下次运行会重新爬取。

**【历史新闻导入位置】**：把历史新闻文件（csv 或 parquet）放进一个文件夹，在配置里设置
`news.history_dir: 路径`。步骤3 会把它们和爬取结果合并，再一起清洗。文件需包含 `datetime`、`title`
两列，再加 `order_book_id` 或 `name` 其中一列。读取逻辑在 `rlinvest/data/news_clean.py` 的
`load_history_news()`。

**防止未来信息**
- 15:00 之后以及非交易日发布的新闻，归入下一个交易日。
- 特征标准化只使用训练集的统计量。
- 在 t 日收盘按当日价格调仓，获得 t 到 t+1 的收益。

## 特征（F = 29）

| 类别 | 特征 | 进入模型的形式 |
|---|---|---|
| 基础量价 | 开盘、收盘、最高、最低、成交量、换手率 | 开/高/低相对收盘的偏离；收盘取对数收益；成交量取相对 20 日均量的对数 |
| 趋势指标 | MA5/10/30/120、EMA30、布林带上下轨 | 相对收盘价的偏离 |
| 波动指标 | ATR、ADX | ATR/收盘价，ADX/100 |
| 超买超卖 | KDJ-K/D/J、RSI6/12/24、CCI | 缩放到 0 附近 |
| 动量 | MACD（DIF/DEA/柱）、MTM | MACD/收盘价 |
| 舆情 | 新浪标题情绪、新闻条数、米筐情绪指数 | 无新闻日情绪按 0.8 衰减；条数取 log1p |

## 与论文的对齐

**表3 参数量完全复现**。`05_train.py` 会输出 `table3_params.csv`，总参数量为 **1,483,671**，各子模块数量与论文一致：

| 模块 | 参数量 | 由此确定的结构 |
|---|---|---|
| 前向/后向 LSTM | 81,408 ×2 | 输入 F=29，隐藏层 128 |
| Transformer 层（×2） | 330,112 | d=256，8 头，前馈维度 128 |
| 特征注意力、标的注意力 | 263,168 ×2 | d=256，8 头 |
| 策略网络 | 65,664 + 1,419 + 1,419 | 512→128→11（U=10 只股票 + 现金） |
| 价值网络 | 65,664 + 129 | 512→128→1 |

表3 只统计了一份特征提取模块，说明策略网络和价值网络共享编码器，本项目也按共享实现。BatchNorm
不带可训练参数，所以不计入表3。

**表4 超参数**全部按论文取值：η=3e-4，γ=0.99，λ=0.95，ε=0.2，σ₀=0.4，β=0.004，w₁/w₂/w₃=0.4/0.5/0.1，
τ=0.0018，r_f=0.01。

**算法1** 的对应实现：
- 采样缓存区容量 C=1000。
- 当 |D| ≥ B 时，进行 E 轮更新，每轮从 D 中采样 B 条。
- 旧策略概率用缓存中保存的 μ_i、σ_i 计算。
- 价值目标 V^target = M̂ + V_old(s)。
- 总损失 L_total = L_policy + 0.5·L_value。
- 每 100 步调整一次 ε_t = 0.2·(1 + 0.5·σ_t/σ_{t,old_W}) 与 η_t = η₀·exp(−α·ΔL/L)。

**论文没有给出、由本项目设定的参数**（在 `default.yaml` 中标注为 `[未给出]`）：

| 参数 | 取值 | 说明 |
|---|---|---|
| T | 20 | 历史窗口长度 |
| B / E | 256 / 10 | 轨迹批量、每批更新轮数 |
| update_interval | 100 | 每 100 步检查一次是否更新 |
| K | 200,000 | 训练总步数，GPU 上约 45 分钟 |
| α | 0.1 | 自适应学习率衰减系数 |
| 波动率窗口 / W | 20 / 60 | 当前波动率 σ_t 与历史平均波动率的窗口 |
| 数据划分 | 训练 2014–2021 / 验证 2022 / 测试 2023–2025.10 | 按验证集夏普比率保存最优模型 |

**实现上的几处说明**
- **课程噪声**：σ_noise = σ₀·exp(−β·n)，n 为已完成的更新轮次。如果按每个交易步衰减，β=0.004 时
  约 1000 步后噪声就接近 0，课程学习起不到作用，所以按更新轮次衰减。
- **标的位置编码**：h^global 对时间和标的两个维度取平均后得到 512 维。如果直接平均，模型分不清
  不同股票。所以在 Bi-LSTM 之后加了固定的正弦标的位置编码，它没有可训练参数，不改变表3的参数量。
- **ε_t 的取值**：按论文公式，波动率比值为 1 时 ε_t=0.3，并不等于初始值 0.2。程序把 ε_t 限制在
  [0.1, 0.4]，把 η_t 限制在 [3e-5, 1e-3]。
- **A 股交易约束**：停牌不可交易，涨停不可买入，跌停不可卖出。受限部分保持原持仓。配置项
  `rl.trade_limits` 可以关闭这些约束。
