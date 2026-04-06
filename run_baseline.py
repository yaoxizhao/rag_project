"""
run_baseline.py — 实验主脚本

将 Retriever / Augmenter / Generator 串联成完整 RAG 流水线，
支持两种对照实验模式：

    no_rag    : LLM 直接作答，无任何检索上下文（幻觉上界基准）
    naive_rag : 标准 RAG，先检索 Top-K 段落再生成回答

用法:
    # 开发集（200 queries × 10k 语料，默认）
    python run_baseline.py --mode no_rag
    python run_baseline.py --mode naive_rag

    # 正式评测集（500 queries × 50k 语料）
    python run_baseline.py --mode no_rag    --dataset eval
    python run_baseline.py --mode naive_rag --dataset eval

    # 快速冒烟测试（只跑 N 条 query，验证流程）
    python run_baseline.py --mode naive_rag --num-queries 5

输出:
    results/{mode}_{dataset}_{timestamp}.csv
    列: query_id | question | answer | contexts | ground_truth
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402 — 须最先导入，触发 HF 镜像设置

from data.loader import load_queries_with_qrels
from rag.augmenter import Augmenter
from rag.generator import Generator
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

# dataset 模式 → (query 数量, ChromaDB 集合名)
DATASET_CONFIG = {
    "dev":  (cfg.DEV_QUERY_NUM,  "nq_dev"),
    "eval": (cfg.EVAL_QUERY_NUM, "nq_eval"),
}


# ──────────────────────────────────────────────────────────
# 核心流水线
# ──────────────────────────────────────────────────────────

def run_experiment(
    mode: str,
    dataset: str = "dev",
    num_queries: int | None = None,
) -> str:
    """
    执行一次完整的 RAG 实验，返回输出 CSV 路径。

    Args:
        mode:        "no_rag" 或 "naive_rag"
        dataset:     "dev" 或 "eval"
        num_queries: 覆盖默认 query 数（None = 使用 config 默认值）

    Returns:
        str: 结果 CSV 的绝对路径
    """
    if mode not in ("no_rag", "naive_rag"):
        raise ValueError(f"mode 须为 'no_rag' 或 'naive_rag'，收到: {mode!r}")
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"dataset 须为 'dev' 或 'eval'，收到: {dataset!r}")

    default_nq, collection_name = DATASET_CONFIG[dataset]
    nq = num_queries if num_queries is not None else default_nq

    logger.info("=" * 60)
    logger.info(f" 实验配置")
    logger.info(f"   mode       : {mode}")
    logger.info(f"   dataset    : {dataset}  ({nq} queries)")
    logger.info(f"   collection : {collection_name}  (naive_rag 时使用)")
    logger.info("=" * 60)

    # ── 1. 加载 Queries ──────────────────────────────────────
    logger.info("[Step 1] 加载测试集 …")
    queries = load_queries_with_qrels(num_queries=nq)
    logger.info(f"  已加载 {len(queries)} 条 query")

    # ── 2. 初始化组件 ─────────────────────────────────────────
    logger.info("[Step 2] 初始化 RAG 组件 …")
    augmenter = Augmenter()
    generator = Generator()

    retriever = None
    if mode == "naive_rag":
        retriever = Retriever(collection_name=collection_name)
        if retriever.count() == 0:
            raise RuntimeError(
                f"ChromaDB 集合 '{collection_name}' 为空！\n"
                f"请先运行: python build_index.py --mode {dataset}"
            )
        logger.info(f"  Retriever 就绪，索引文档数: {retriever.count():,}")

    # ── 3. 运行流水线 ─────────────────────────────────────────
    logger.info(f"[Step 3] 开始推理（mode={mode}，共 {len(queries)} 条）…")
    results = []
    errors  = 0
    t_start = time.time()

    for item in tqdm(queries, desc=f"[{mode}]", unit="query"):
        qid       = item["query_id"]
        question  = item["question"]
        gt        = item["ground_truth"]
        contexts  = []

        # 检索（仅 naive_rag）
        if mode == "naive_rag":
            hits     = retriever.retrieve(question, top_k=cfg.TOP_K)
            contexts = [h["text"] for h in hits]

        # 构建 Prompt
        prompt = augmenter.build_prompt(question, contexts)

        # LLM 生成（带错误恢复）
        try:
            answer = generator.generate(prompt)
        except Exception as e:
            logger.warning(f"  [WARN] query {qid} 生成失败: {e}")
            answer = ""
            errors += 1

        result = {
            "query_id":     qid,
            "question":     question,
            "answer":       answer,
            "ground_truth": gt,
        }
        # contexts 存为独立列 context_0 / context_1 / ...（避免 CSV 内嵌 JSON）
        for i, ctx in enumerate(contexts):
            result[f"context_{i}"] = ctx
        results.append(result)

    elapsed = time.time() - t_start
    logger.info(
        f"[Step 3] 推理完成: {len(results)} 条，"
        f"失败 {errors} 条，耗时 {elapsed:.1f}s "
        f"({elapsed / len(results):.2f}s/query)"
    )

    # ── 4. 保存 CSV ───────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    run_dir  = os.path.join(cfg.RESULTS_DIR, f"run_{date_str}", mode)
    os.makedirs(run_dir, exist_ok=True)
    filename  = f"{mode}_{dataset}_{ts}.csv"
    out_path  = os.path.join(run_dir, filename)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"[Step 4] 结果已保存: {out_path}  ({len(df)} 行 × {len(df.columns)} 列)")

    # ── 5. 简要统计 ───────────────────────────────────────────
    answered = df[df["answer"].str.strip() != ""]
    logger.info(f"  有效回答率: {len(answered)}/{len(df)} ({100*len(answered)/len(df):.1f}%)")
    logger.info(f"  平均回答长度: {df['answer'].str.split().apply(len).mean():.0f} 词")
    if mode == "naive_rag":
        ctx_cols = [c for c in df.columns if c.startswith("context_")]
        logger.info(f"  平均检索段落数: {len(ctx_cols)}")

    return out_path


# ──────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline.py --mode no_rag
  python run_baseline.py --mode naive_rag --dataset eval
  python run_baseline.py --mode naive_rag --num-queries 5
        """,
    )
    parser.add_argument(
        "--mode", required=True, choices=["no_rag", "naive_rag"],
        help="no_rag: LLM only  |  naive_rag: Retrieve then Generate",
    )
    parser.add_argument(
        "--dataset", default="dev", choices=["dev", "eval"],
        help="dev=200q/10k docs  |  eval=500q/50k docs  (default: dev)",
    )
    parser.add_argument(
        "--num-queries", type=int, default=None, metavar="N",
        help="Override query count (e.g. 5 for smoke test)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    out  = run_experiment(
        mode=args.mode,
        dataset=args.dataset,
        num_queries=args.num_queries,
    )
    print(f"\n结果文件: {out}")
    print("下一步评估: python evaluate.py --input " + out)
