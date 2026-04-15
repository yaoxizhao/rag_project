"""
rag/augmenter.py — Prompt 构建与上下文注入

封装两种实验模式的 Prompt 模板：
    no_rag   : contexts=[] → 裸 query，LLM 凭自身知识作答
    naive_rag: contexts=[...] → 注入检索段落，要求 LLM 基于上下文作答

公开接口（符合架构文档规范）：
    Augmenter()
        .build_prompt(query, contexts) -> str

── 架构插拔点 ────────────────────────────────────────────────
后续可在此处替换为 CoTAugmenter、SelfRAGAugmenter、
HyDEAugmenter 等，只需保持 build_prompt() 签名不变。
─────────────────────────────────────────────────────────────
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402


# ── Prompt 模板常量 ────────────────────────────────────────
# 集中定义，方便后续对比消融不同模板的效果

_NO_RAG_TEMPLATE = """\
Answer the following question concisely and accurately.
If you are not sure, give your best answer based on your knowledge.

Question: {query}
Answer:"""

_NAIVE_RAG_TEMPLATE = """\
Use the following context passages to answer the question.
Base your answer primarily on the provided context. If the context \
partially addresses the question, provide what you can and briefly \
note any gaps. Only say you cannot answer if the context contains \
no relevant information at all.

Context:
{context_block}

Question: {query}
Answer:"""


class Augmenter:
    """
    Prompt 构建器。

    根据 contexts 是否为空自动切换模式：
        contexts=[]   → no_rag  模式
        contexts=[..] → naive_rag 模式

    使用方法：
        aug = Augmenter()
        prompt = aug.build_prompt(query, contexts)
    """

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        """
        构建完整 Prompt 字符串。

        Args:
            query:    用户查询文本
            contexts: 检索到的段落列表（空列表 → no_rag 模式）

        Returns:
            str: 可直接传入 Generator.generate() 的 Prompt
        """
        if not contexts:
            return self._build_no_rag(query)
        return self._build_naive_rag(query, contexts)

    # ──────────────────────────────────────────────────────
    # 内部模板方法
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _build_no_rag(query: str) -> str:
        return _NO_RAG_TEMPLATE.format(query=query.strip())

    @staticmethod
    def _build_naive_rag(query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx.strip()}" for i, ctx in enumerate(contexts)
        )
        return _NAIVE_RAG_TEMPLATE.format(
            context_block=context_block,
            query=query.strip(),
        )


# ──────────────────────────────────────────────────────────
# 本地测试入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60
    aug = Augmenter()

    # ── Test 1: no_rag 模式 ────────────────────────────────
    print(f"\n{SEP}")
    print("Test 1 / no_rag 模式（contexts=[]）")
    print(SEP)
    q = "who wrote hallelujah i just love her so"
    prompt_no_rag = aug.build_prompt(q, contexts=[])
    print(prompt_no_rag)
    assert "Question:" in prompt_no_rag
    assert q in prompt_no_rag
    assert "Context:" not in prompt_no_rag
    print("\n  [PASS] no_rag prompt 格式验证通过")

    # ── Test 2: naive_rag 模式 ─────────────────────────────
    print(f"\n{SEP}")
    print("Test 2 / naive_rag 模式（contexts=[...] ×3）")
    print(SEP)
    fake_contexts = [
        '"Hallelujah I Love Her So" is a single from Ray Charles released in 1956.',
        "The song was written by Ray Charles himself and became a blues classic.",
        "Ray Charles, born Ray Charles Robinson, was an American singer and musician.",
    ]
    prompt_rag = aug.build_prompt(q, contexts=fake_contexts)
    print(prompt_rag)
    assert "[1]" in prompt_rag and "[2]" in prompt_rag and "[3]" in prompt_rag
    assert "Context:" in prompt_rag
    assert q in prompt_rag
    print("\n  [PASS] naive_rag prompt 格式验证通过")

    # ── Test 3: 边界条件 ───────────────────────────────────
    print(f"\n{SEP}")
    print("Test 3 / 边界条件")
    print(SEP)
    # 单条 context
    p1 = aug.build_prompt("test question", ["single context passage"])
    assert "[1]" in p1 and "[2]" not in p1
    print("  单条 context: [PASS]")
    # query 前后有空白
    p2 = aug.build_prompt("  spaced query  ", [])
    assert "spaced query" in p2
    print("  query trim:   [PASS]")

    print(f"\n{SEP}")
    print("rag/augmenter.py 全部测试通过！")
    print(SEP)
