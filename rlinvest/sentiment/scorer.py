"""新闻标题情绪打分，输出 [-1,1]，并聚合为个股日度情绪。

LexiconScorer     : 金融情感词典 + 否定词翻转，无需 GPU，可通过 txt 扩充词典
TransformerScorer : 任意 HuggingFace 中文情感分类模型（需安装 transformers）
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

POS_WORDS = """增长 大增 预增 扭亏 盈利 超预期 新高 创新高 突破 中标 订单 签约 签订 合作 战略合作 获批 批准 通过 上市
回购 增持 分红 派息 高送转 涨停 大涨 上涨 走强 领涨 利好 受益 景气 提升 加速 扩产 投产 量产 首发 发布 推出 研发成功
入选 获奖 龙头 看好 买入 推荐 上调 增资 注资 并购 收购 重组成功 解禁结束 摘帽 业绩亮眼 翻倍 同比增长 净利增长""".split()
NEG_WORDS = """下滑 下降 下跌 大跌 跌停 走弱 亏损 预亏 预减 首亏 续亏 减持 清仓 质押 冻结 立案 调查 处罚 罚款 警示 问询
留置 被查 违规 违法 造假 诉讼 仲裁 退市 风险警示 戴帽 爆雷 违约 逾期 债务 终止 取消 失败 暂停 停产 召回 事故 下调
卖出 利空 承压 萎缩 流出 跳水 闪崩 腰斩 商誉减值 减值 计提 解禁 辞职 失联 被执行""".split()
NEGATIONS = ['未', '不', '无', '没有', '否认', '并非', '免于', '解除']


class LexiconScorer:
    def __init__(self, extra_pos=None, extra_neg=None):
        pos, neg = set(POS_WORDS), set(NEG_WORDS)
        for path, s in [(extra_pos, pos), (extra_neg, neg)]:
            if path:
                s.update(w.strip() for w in Path(path).read_text(encoding='utf-8').splitlines() if w.strip())
        # 长词优先匹配，避免“增长”与“同比增长”重复计数
        self.words = sorted([(w, 1) for w in pos] + [(w, -1) for w in neg], key=lambda x: -len(x[0]))

    def score_one(self, text: str) -> float:
        s, used = 0.0, [False] * len(text)
        for w, sign in self.words:
            start = text.find(w)
            while start != -1:
                if not any(used[start:start + len(w)]):
                    for i in range(start, start + len(w)):
                        used[i] = True
                    pre = text[max(0, start - 3):start]
                    s += -sign if any(n in pre for n in NEGATIONS) else sign
                start = text.find(w, start + 1)
        return float(np.tanh(s / 2))   # 压缩到 (-1,1)

    def score(self, texts):
        return np.array([self.score_one(t) for t in tqdm(texts, desc='词典情绪打分', mininterval=5)])


class TransformerScorer:
    def __init__(self, model_name, batch_size=64):
        from transformers import pipeline
        import torch
        self.pipe = pipeline('text-classification', model=model_name, top_k=None, truncation=True,
                             device=0 if torch.cuda.is_available() else -1)
        self.bs = batch_size

    @staticmethod
    def _to_score(preds):
        """把各标签概率映射为 P(正面) - P(负面)。"""
        p = {d['label'].lower(): d['score'] for d in preds}
        pos = sum(v for k, v in p.items() if any(x in k for x in ['pos', '正', 'bull']))
        neg = sum(v for k, v in p.items() if any(x in k for x in ['neg', '负', 'bear']))
        return pos - neg

    def score(self, texts):
        out = []
        for i in tqdm(range(0, len(texts), self.bs), desc='模型情绪打分'):
            out += [self._to_score(p) for p in self.pipe(list(texts[i:i + self.bs]))]
        return np.array(out)


def get_scorer(cfg):
    c = cfg['sentiment']
    if c['method'] == 'transformer':
        return TransformerScorer(c['model_name'], c['batch_size'])
    return LexiconScorer(c.get('extra_pos_dict'), c.get('extra_neg_dict'))


def daily_sentiment(news: pd.DataFrame):
    """个股日度聚合：平均情绪、新闻条数。"""
    g = news.groupby(['order_book_id', 'date'])
    return g['score'].agg(sina_senti='mean', news_cnt='size').reset_index()
