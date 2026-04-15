"""
evaluate.py — Ragas 评估入口

读取 run_baseline.py 输出的 CSV，计算 Ragas 四项指标，
输出结果到终端和 JSON 文件。

用法:
    python evaluate.py --input results/naive_rag_dev_20260404_120000.csv
    python evaluate.py --input results/no_rag_dev_20260404_120000.csv
    python evaluate.py --input results/naive_rag_dev_*.csv --sample 50

前置条件:
    在 .env 文件中配置 GLM_RAGAS_API_KEY（Ragas 评测专用）

输出:
    终端打印：指标摘要表
    results/{stem}_ragas.json：完整结果（含逐样本得分）
"""
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

from evaluation.metrics import build_ragas_embeddings, build_ragas_llm, evaluate_rag

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────

def load_results_csv(csv_path: str, sample: int | None = None) -> list[dict]:
    """
    读取 run_baseline.py 生成的 CSV，还原 contexts 为 list[str]。
    contexts 以多列形式存储：context_0, context_1, ...

    Args:
        csv_path: CSV 文件路径
        sample:   随机采样条数（None = 全量）

    Returns:
        list of dict: [{question, answer, contexts: list[str], ground_truth}]
    """
    df = pd.read_csv(csv_path)
    required = {"query_id", "question", "answer", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}  (实际列: {list(df.columns)})")

    # 空答案不过滤，生成失败也计入评估（得分自然为低）
    empty_count = (df["answer"].str.strip() == "").sum()
    if empty_count > 0:
        logger.warning(f"  含 {empty_count} 条空 answer（生成失败），仍计入评估")

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
        logger.info(f"  随机采样 {sample} 条（seed={cfg.RANDOM_SEED}）")

    ctx_cols = sorted(
        [c for c in df.columns if c.startswith("context_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )

    records = []
    for _, row in df.iterrows():
        ctxs = [str(row[c]) for c in ctx_cols if pd.notna(row[c]) and str(row[c]).strip()]
        records.append({
            "query_id":          str(row["query_id"]),
            "question":          str(row["question"]),
            "answer":            str(row["answer"]),
            "contexts":          ctxs,
            "ground_truth":      str(row["ground_truth"]),
            "is_impossible":     bool(row.get("is_impossible", False)),
            "relevant_context":  str(row.get("relevant_context", "")),
        })

    logger.info(
        f"  加载 {len(records)} 条记录，"
        f"有上下文: {sum(1 for r in records if r['contexts'])} 条"
    )
    return records


# ──────────────────────────────────────────────────────────
# 主评估流程
# ──────────────────────────────────────────────────────────

def run_evaluation(csv_path: str, sample: int | None = None) -> str:
    """
    执行完整评估流程，返回输出 JSON 路径。

    Args:
        csv_path: run_baseline.py 输出的 CSV 文件路径
        sample:   评估条数上限（None = 全量）

    Returns:
        str: 输出 JSON 的绝对路径
    """
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    logger.info("=" * 60)
    logger.info(f" 评估配置")
    logger.info(f"   输入文件  : {csv_path}")
    logger.info(f"   评估条数  : {sample or '全量'}")
    logger.info(f"   LLM Judge : {cfg.GLM_MODEL}")
    logger.info("=" * 60)

    # ── 1. 加载数据 ────────────────────────────────────────────
    logger.info("[Step 1] 加载结果 CSV …")
    records = load_results_csv(csv_path, sample=sample)

    # ── 按可回答/不可回答分组 ──────────────────────────────────
    answerable = [r for r in records if not r["is_impossible"]]
    unanswerable = [r for r in records if r["is_impossible"]]
    logger.info(f"  可回答: {len(answerable)} 条，不可回答: {len(unanswerable)} 条")

    scores = {}

    # ── 2. 可回答问题：Ragas 完整评测 ─────────────────────────
    if answerable:
        logger.info("[Step 2] 初始化 LLM Judge …")
        ragas_llm = build_ragas_llm()
        ragas_embeddings = build_ragas_embeddings()

        logger.info(f"[Step 3a] 可回答问题 Ragas 评估（{len(answerable)} 条）…")
        ans_scores = evaluate_rag(answerable, ragas_llm=ragas_llm, ragas_embeddings=ragas_embeddings)
        # 加前缀区分
        for k, v in ans_scores.items():
            scores[f"answerable_{k}"] = v

    # ── 3. 不可回答问题：计算幻觉率 ────────────────────────────
    if unanswerable:
        from evaluation.metrics import _compute_abstention_rate
        logger.info(f"[Step 3b] 不可回答问题幻觉率评估（{len(unanswerable)} 条）…")
        abstention_rate = _compute_abstention_rate(unanswerable)
        hallucination_rate = 1.0 - abstention_rate
        scores["hallucination_rate"] = hallucination_rate
        scores["unanswerable_abstention_rate"] = abstention_rate
        scores["unanswerable_total"] = len(unanswerable)

    # ── 4. 打印摘要 ────────────────────────────────────────────
    _print_summary(scores, csv_path, len(records))

    # ── 5. 保存 JSON ───────────────────────────────────────────
    stem     = Path(csv_path).stem
    out_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), f"{stem}_ragas.json")

    output = {
        "source_file":  csv_path,
        "evaluated_at": datetime.now().isoformat(),
        "num_samples":  len(records),
        "scores":       scores,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"[Step 5] 结果已保存: {out_path}")
    return out_path


def _print_summary(scores: dict, csv_path: str, n: int) -> None:
    """在终端打印格式化的评估结果。"""
    SEP = "=" * 50
    METRIC_LABELS = {
        "answerable_faithfulness":       "Faithfulness       (可回答, 幻觉↓)",
        "answerable_answer_correctness": "Answer Correctness (可回答, 准确性)",
        "answerable_context_recall":     "Context Recall     (可回答, 覆盖率)",
        "answerable_context_precision":  "Context Precision  (可回答, 精确率)",
        "answerable_abstention_rate":    "Abstention Rate    (可回答, 拒答率)",
        "hallucination_rate":            "Hallucination Rate (不可回答, 幻觉↑)",
        "unanswerable_abstention_rate":  "Abstention Rate    (不可回答, 拒答↓)",
    }
    print(f"\n{SEP}")
    print(f"  Ragas 评估结果 (SQuAD 2.0)")
    print(f"  文件: {os.path.basename(csv_path)}")
    print(f"  样本数: {n}")
    print(SEP)
    for key, label in METRIC_LABELS.items():
        val = scores.get(key, float("nan"))
        if isinstance(val, (int, float)) and math.isnan(val):
            print(f"  {label:<48} N/A")
        elif isinstance(val, (int, float)):
            filled = int(val * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {label:<48} {val:.4f}  [{bar}]")
        else:
            print(f"  {label:<48} {val}")
    print(SEP)


# ──────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG results with Ragas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --input results/naive_rag_dev_20260404_120000.csv
  python evaluate.py --input results/naive_rag_dev_20260404_120000.csv --sample 50
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to run_baseline.py output CSV",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Evaluate only N randomly sampled rows (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args   = _parse_args()
    out    = run_evaluation(csv_path=args.input, sample=args.sample)
    print(f"\n完整结果 JSON: {out}")
