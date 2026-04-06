"""
evaluate.py — Ragas 评估入口

读取 run_baseline.py 输出的 CSV，计算 Ragas 四项指标，
输出结果到终端和 JSON 文件。

用法:
    python evaluate.py --input results/naive_rag_dev_20260404_120000.csv
    python evaluate.py --input results/no_rag_dev_20260404_120000.csv
    python evaluate.py --input results/naive_rag_dev_*.csv --sample 50

前置条件:
    export DEEPSEEK_API_KEY='sk-...'   # DeepSeek API Key

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

    # 过滤掉生成失败的行（answer 为空）
    before = len(df)
    df = df[df["answer"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        logger.warning(f"  过滤掉 {before - len(df)} 条空 answer 行，剩余 {len(df)} 条")

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
            "query_id":    str(row["query_id"]),
            "question":    str(row["question"]),
            "answer":      str(row["answer"]),
            "contexts":    ctxs,
            "ground_truth": str(row["ground_truth"]),
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

    # ── 2. 初始化 LLM Judge ───────────────────────────────────
    logger.info("[Step 2] 初始化 DeepSeek LLM Judge …")
    ragas_llm = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()

    # ── 3. Ragas 评估 ──────────────────────────────────────────
    logger.info(f"[Step 3] 执行 Ragas 评估（{len(records)} 条）…")
    scores = evaluate_rag(records, ragas_llm=ragas_llm, ragas_embeddings=ragas_embeddings)

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
        "faithfulness":       "Faithfulness       (幻觉↓)",
        "answer_correctness": "Answer Correctness (准确性)",
        "context_recall":     "Context Recall     (覆盖率)",
        "context_precision":  "Context Precision  (精确率)",
        "abstention_rate":    "Abstention Rate    (拒答率)",
    }
    print(f"\n{SEP}")
    print(f"  Ragas 评估结果")
    print(f"  文件: {os.path.basename(csv_path)}")
    print(f"  样本数: {n}")
    print(SEP)
    for key, label in METRIC_LABELS.items():
        val = scores.get(key, float("nan"))
        if math.isnan(val):   # NaN
            bar = "N/A (no_rag 模式)"
            print(f"  {label:<38} N/A")
        else:
            filled = int(val * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {label:<38} {val:.4f}  [{bar}]")
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
