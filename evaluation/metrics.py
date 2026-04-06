"""
evaluation/metrics.py — Ragas 0.4.x 评估指标封装

使用 GLM-4-Flash API 作为 LLM Judge，计算四项 RAG 核心指标：
    faithfulness       — 回答是否忠实于检索上下文（核心幻觉指标）
    answer_relevancy   — 回答与问题的相关性
    context_recall     — 上下文对 ground truth 的覆盖率
    context_precision  — 上下文中相关段落的精确率

公开接口：
    build_ragas_llm()                  -> LangchainLLMWrapper
    build_ragas_embeddings()           -> LangchainEmbeddingsWrapper
    evaluate_rag(records, llm, emb)    -> dict[metric_name, score]

说明：
    no_rag 模式（retrieved_contexts 为空列表）仅计算 answer_relevancy，
    其余三项指标需要上下文，自动跳过。

注意（Ragas 0.4.x）：
    evaluate() 只接受 ragas.metrics.base.Metric 子类（旧式 API）。
    使用 ragas.metrics._faithfulness 等私有模块，不使用 ragas.metrics.collections。
    AnswerRelevancy 须设 strictness=1，否则报 n>1 不支持的错误。
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)


def build_ragas_llm():
    """
    构建以 GLM-4-Flash 为后端的 Ragas LLM Judge（LangchainLLMWrapper）。

    Returns:
        LangchainLLMWrapper 实例
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if not cfg.GLM_API_KEY:
        raise EnvironmentError(
            "未设置 GLM_API_KEY 环境变量。\n"
            "请在 .env 文件中添加: GLM_API_KEY=your_key"
        )

    lc_llm = ChatOpenAI(
        model=cfg.GLM_MODEL,
        api_key=cfg.GLM_API_KEY,
        base_url=cfg.GLM_BASE_URL,
        temperature=0,
    )
    logger.info(f"[Ragas] LLM Judge: {cfg.GLM_MODEL} @ {cfg.GLM_BASE_URL}")
    return LangchainLLMWrapper(lc_llm)


def build_ragas_embeddings():
    """
    构建本地 bge-m3 Embedding（供 AnswerRelevancy 使用）。

    Returns:
        LangchainEmbeddingsWrapper 实例
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    lc_emb = HuggingFaceEmbeddings(
        model_name=cfg.EMBED_MODEL,
        cache_folder=cfg.HF_CACHE_DIR,
        model_kwargs={"device": cfg.EMBED_GPU},
    )
    logger.info(f"[Ragas] Embeddings: {cfg.EMBED_MODEL} (local)")
    return LangchainEmbeddingsWrapper(lc_emb)


def evaluate_rag(
    records: list[dict],
    ragas_llm=None,
    ragas_embeddings=None,
) -> dict:
    """
    对一批 RAG 实验结果执行 Ragas 评估。

    使用旧式 Metric API（ragas.metrics._*），兼容 ragas.evaluate()。

    Args:
        records:   list of dict，每条须含：
                     question      — 问题文本
                     answer        — LLM 生成的回答
                     contexts      — list[str]，检索段落（空列表 = no_rag）
                     ground_truth  — 参考答案文本
        ragas_llm: LangchainLLMWrapper（None 则自动初始化 DeepSeek）
        ragas_embeddings: LangchainEmbeddingsWrapper（None 则自动初始化 bge-m3）

    Returns:
        dict: {metric_name: float}，含 NaN 的指标表示不适用
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    if ragas_llm is None:
        ragas_llm = build_ragas_llm()
    if ragas_embeddings is None:
        ragas_embeddings = build_ragas_embeddings()

    # ── 判断是否有上下文（决定可用指标集）──────────────────────
    has_contexts = any(bool(r.get("contexts")) for r in records)

    if has_contexts:
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
            ContextRecall(llm=ragas_llm),
            ContextPrecision(llm=ragas_llm),
        ]
        logger.info("[Ragas] 指标: faithfulness + answer_relevancy + context_recall + context_precision")
    else:
        metrics = [AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)]
        logger.info("[Ragas] no_rag 模式，仅计算: answer_relevancy")

    # ── 构建 EvaluationDataset ─────────────────────────────────
    samples = []
    for r in records:
        ctxs = r.get("contexts") or []
        if not ctxs:
            ctxs = [""]   # 占位符，Ragas 要求非空
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=ctxs,
                reference=r["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    logger.info(f"[Ragas] 开始评估 {len(samples)} 条样本 …")

    # ── 执行评估 ───────────────────────────────────────────────
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    # ── 提取得分 ───────────────────────────────────────────────
    scores = {}
    result_df = result.to_pandas()
    for m in metrics:
        col = m.name
        if col in result_df.columns:
            scores[col] = float(result_df[col].mean(skipna=True))
        else:
            scores[col] = float("nan")

    logger.info(f"[Ragas] 评估完成: {scores}")
    return scores
