"""
rag/crag/pipeline.py — CRAG-Lite Pipeline

继承基类 Pipeline，重写 process() 实现 CRAG 核心流程：
    检索 → 评估检索质量 → 触发动作 → 知识精炼/拒绝/保守 → 生成

三种动作对应三种 Prompt 策略：
    correct    → 精炼后的上下文 + 标准 RAG prompt
    incorrect  → 空 context + 拒绝回答 prompt
    ambiguous  → 精炼后的上下文 + 含不确定性提示的 prompt

公开接口：
    CRAGPipeline(collection_name)
        .process(query) -> {"answer": str, "contexts": list[str]}
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg  # noqa: E402

from rag.pipeline import Pipeline  # noqa: E402

logger = logging.getLogger(__name__)

# ── Prompt 模板 ──────────────────────────────────────────────

_CORRECT_PROMPT = """\
Use the following context passages to answer the question.
Base your answer primarily on the provided context.

Context:
{context_block}

Question: {query}
Answer:"""

_INCORRECT_PROMPT = """\
Based on the available information, you cannot find relevant evidence to answer the following question.
Respond with exactly: "I cannot answer this question based on the available information."

Question: {query}
Answer:"""

_AMBIGUOUS_PROMPT = """\
Use the following context passages to answer the question.
The context may be incomplete or partially relevant. Base your answer on the context, \
but if you are not confident, say you cannot answer.

Context:
{context_block}

Question: {query}
Answer:"""


class CRAGPipeline(Pipeline):
    """CRAG-Lite: retrieval-evaluation-driven adaptive RAG."""

    def __init__(self, collection_name=None, **kwargs):
        from rag.crag.evaluator import CRAGEvaluator
        from rag.crag.refiner import CRAGRefiner
        from rag.generator import Generator
        from rag.retriever import Retriever

        if not collection_name:
            raise ValueError("crag_lite 需要 collection_name 参数")

        retriever = Retriever(collection_name=collection_name)
        if retriever.count() == 0:
            raise RuntimeError(
                f"ChromaDB 集合 '{collection_name}' 为空！\n"
                f"请先运行: python build_index.py --mode eval"
            )

        # 基类初始化（augmenter 不使用，prompt 由本类自行管理）
        super().__init__(
            retriever=retriever,
            augmenter=None,
            generator=Generator(),
        )

        # CRAG 特有模块
        self._evaluator = CRAGEvaluator()
        self._refiner = CRAGRefiner(embed_model=retriever._model)

        logger.info("[CRAGPipeline] 初始化完成")

    def process(self, query: str) -> dict:
        """
        CRAG-Lite inference: retrieve -> evaluate -> act -> generate.

        Returns:
            {"answer": str, "contexts": list[str]}
        """
        # 1. 检索（保留 score）
        hits = self.retriever.retrieve(query, top_k=cfg.TOP_K)

        # 2. 评估检索质量
        action = self._evaluator.evaluate(hits)

        # 3. 根据动作处理
        if action == "correct":
            contexts = self._refiner.refine(query, hits)
            prompt = self._build_prompt("correct", query, contexts)

        elif action == "incorrect":
            contexts = []  # 不传递上下文
            prompt = self._build_prompt("incorrect", query, contexts)

        else:  # ambiguous
            contexts = self._refiner.refine(query, hits)
            prompt = self._build_prompt("ambiguous", query, contexts)

        # 4. 生成
        answer = self.generator.generate(prompt)

        return {"answer": answer, "contexts": contexts}

    # ──────────────────────────────────────────────────────────
    # Prompt 构建
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        action: str, query: str, contexts: list[str],
    ) -> str:
        """Build prompt based on action type."""
        query = query.strip()

        if action == "incorrect":
            return _INCORRECT_PROMPT.format(query=query)

        context_block = "\n\n".join(
            f"[{i + 1}] {ctx.strip()}" for i, ctx in enumerate(contexts)
        )

        if action == "correct":
            template = _CORRECT_PROMPT
        else:  # ambiguous
            template = _AMBIGUOUS_PROMPT

        return template.format(context_block=context_block, query=query)
