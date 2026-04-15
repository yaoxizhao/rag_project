"""
rag/crag/refiner.py — CRAG 知识精炼模块

实现 CRAG 论文 Section 4.4 的 Decompose-then-Recompose 算法：
    1. 将检索到的文档分解为知识条（knowledge strips，按句子切分）
    2. 对每条用 bge-m3 计算 query-strip 余弦相似度
    3. 过滤掉低于阈值的无关条
    4. 按原顺序重组为精炼后的上下文

公开接口：
    CRAGRefiner(embed_model, strip_threshold)
        .refine(query, hits) -> list[str]
"""
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


class CRAGRefiner:
    """
    知识精炼器：decompose → filter → recompose。

    使用方法：
        refiner = CRAGRefiner(embed_model=retriever._model)
        refined = refiner.refine(query, hits)
    """

    def __init__(self, embed_model, strip_threshold: float = 0.40):
        """
        Args:
            embed_model:    SentenceTransformer 实例（复用 Retriever 的 bge-m3）
            strip_threshold: 知识条相关性过滤阈值（余弦相似度 < 此值则丢弃）
        """
        self._model = embed_model
        self.strip_threshold = strip_threshold
        logger.info(
            f"[CRAGRefiner] 初始化完成 "
            f"(strip_threshold={strip_threshold})"
        )

    def refine(self, query: str, hits: list[dict]) -> list[str]:
        """
        对检索结果做知识精炼。

        Args:
            query: 用户查询
            hits:  Retriever.retrieve() 的返回结果

        Returns:
            精炼后的上下文列表（每个元素是一个文档的精炼后文本）
        """
        if not hits:
            return []

        # 一次性编码 query（后续复用）
        query_emb = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )

        refined_contexts = []
        for hit in hits:
            strips = self._decompose(hit["text"])
            if not strips:
                continue

            relevant_strips = self._filter_strips(query_emb, strips)
            if relevant_strips:
                recomposed = " ".join(relevant_strips)
                refined_contexts.append(recomposed)

        logger.debug(
            f"[CRAGRefiner] {len(hits)} docs → {len(refined_contexts)} refined contexts"
        )
        return refined_contexts

    # ──────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────

    def _filter_strips(
        self, query_emb: np.ndarray, strips: list[str],
    ) -> list[str]:
        """对知识条逐条打分，过滤低分条。"""
        strip_embs = self._model.encode(
            strips, normalize_embeddings=True, show_progress_bar=False,
        )
        # 余弦相似度（已归一化，点积即余弦）
        scores = (query_emb @ strip_embs.T).flatten()

        relevant = []
        for strip, score in zip(strips, scores):
            if score >= self.strip_threshold:
                relevant.append(strip)

        return relevant

    @staticmethod
    def _decompose(text: str) -> list[str]:
        """
        将文本分解为知识条。

        规则：
            - 按句子边界（. ! ?）切分
            - 过短的条（<10 字符）合并或丢弃
        """
        # 按句子标点切分，保留标点
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # 过滤空串和过短片段
        return [s.strip() for s in sentences if len(s.strip()) >= 10]
