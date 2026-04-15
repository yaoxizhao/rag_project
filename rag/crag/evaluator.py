"""
rag/crag/evaluator.py — CRAG 检索质量评估器

基于 ChromaDB 返回的余弦相似度分数，判断检索结果的整体质量，
触发三档动作：correct / incorrect / ambiguous。

公开接口：
    CRAGEvaluator(upper_threshold, lower_threshold)
        .evaluate(hits) -> str       # 返回 "correct" / "incorrect" / "ambiguous"

评估逻辑（忠实于 CRAG 论文 Section 4.3）：
    - 任一文档 score ≥ upper_threshold → correct（至少一篇相关）
    - 所有文档 score < lower_threshold → incorrect（全部无关）
    - 其余 → ambiguous（难以判定）
"""
import logging

logger = logging.getLogger(__name__)


class CRAGEvaluator:
    """
    轻量级检索评估器（零训练路径：复用 bge-m3 余弦相似度分数）。

    使用方法：
        ev = CRAGEvaluator()
        action = ev.evaluate([{"score": 0.72, ...}, {"score": 0.31, ...}])
        # action = "correct"
    """

    def __init__(
        self,
        upper_threshold: float = 0.50,
        lower_threshold: float = 0.30,
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        logger.info(
            f"[CRAGEvaluator] 初始化完成 "
            f"(upper={upper_threshold}, lower={lower_threshold})"
        )

    def evaluate(self, hits: list[dict]) -> str:
        """
        评估检索结果质量，返回触发的动作。

        Args:
            hits: Retriever.retrieve() 返回的结果列表，
                  每条包含 "score" 字段（余弦相似度 ∈ [0, 1]）

        Returns:
            "correct"    — 至少一篇文档相关，可做知识精炼
            "incorrect"  — 所有文档无关，应拒绝回答
            "ambiguous"  — 不确定，保守处理
        """
        if not hits:
            logger.debug("[CRAGEvaluator] 无检索结果 → incorrect")
            return "incorrect"

        scores = [h["score"] for h in hits]
        max_score = max(scores)

        if max_score >= self.upper_threshold:
            action = "correct"
        elif max_score < self.lower_threshold:
            action = "incorrect"
        else:
            action = "ambiguous"

        logger.debug(
            f"[CRAGEvaluator] scores: {[f'{s:.3f}' for s in scores]} "
            f"→ max={max_score:.3f} → {action}"
        )
        return action
