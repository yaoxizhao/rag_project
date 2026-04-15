"""
run_baseline.py — 实验主脚本

通过 rag/pipeline.py 的注册表机制选择流水线模式，
支持无限制扩展新的 RAG 策略，无需修改本文件。

内置模式:
    no_rag    : LLM 直接作答，无检索上下文
    naive_rag : 标准 RAG，先检索 Top-K 再生成

扩展模式:
    只需在 rag/pipeline.py 中 @register_pipeline("新名称") 即可，
    本文件通过 --mode 自动路由，不需要改。

用法:
    python run_baseline.py --mode no_rag
    python run_baseline.py --mode naive_rag --dataset eval
    python run_baseline.py --mode naive_rag --num-queries 5
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402 — 须最先导入，触发 HF 镜像设置

from data.loader import load_queries_with_qrels
from rag.pipeline import available_modes, create_pipeline

logger = logging.getLogger(__name__)

# dataset 模式 → (query 数量, ChromaDB 集合名)
_dataset_slug = cfg.DATASET_NAME
DATASET_CONFIG = {
    "dev":  (cfg.DEV_QUERY_NUM,  f"{_dataset_slug}_dev"),
    "eval": (cfg.EVAL_QUERY_NUM, f"{_dataset_slug}_eval"),
}


# ──────────────────────────────────────────────────────────
# 核心实验流程
# ──────────────────────────────────────────────────────────

def run_experiment(
    mode: str,
    dataset: str = "dev",
    num_queries: int | None = None,
) -> str:
    """
    执行一次完整的 RAG 实验，返回输出 CSV 路径。
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"dataset 须为 'dev' 或 'eval'，收到: {dataset!r}")

    default_nq, collection_name = DATASET_CONFIG[dataset]
    nq = num_queries if num_queries is not None else default_nq

    logger.info("=" * 60)
    logger.info(f" 实验配置")
    logger.info(f"   mode       : {mode}")
    logger.info(f"   dataset    : {dataset}  ({nq} queries)")
    logger.info(f"   collection : {collection_name}")
    logger.info("=" * 60)

    # ── 1. 加载 Queries ──────────────────────────────────────
    logger.info("[Step 1] 加载测试集 …")
    queries = load_queries_with_qrels(num_queries=nq)
    logger.info(f"  已加载 {len(queries)} 条 query")

    # ── 2. 创建 Pipeline（通过注册表自动路由）───────────────
    logger.info(f"[Step 2] 创建 Pipeline (mode={mode}) …")
    pipeline = create_pipeline(mode, collection_name=collection_name)

    # ── 准备输出目录和断点文件 ─────────────────────────────────
    # 先扫描是否已有断点（支持跨天续跑）
    ckpt_path = None
    run_dir   = None
    if os.path.isdir(cfg.RESULTS_DIR):
        for entry in sorted(os.listdir(cfg.RESULTS_DIR)):
            candidate = os.path.join(cfg.RESULTS_DIR, entry, mode)
            candidate_ckpt = os.path.join(candidate, f"{mode}_{dataset}_checkpoint.csv")
            if os.path.exists(candidate_ckpt):
                ckpt_path = candidate_ckpt
                run_dir   = candidate
                break

    # 没找到断点，用当天日期创建新目录
    if run_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        run_dir  = os.path.join(cfg.RESULTS_DIR, f"run_{date_str}", mode)
    os.makedirs(run_dir, exist_ok=True)
    if ckpt_path is None:
        ckpt_path = os.path.join(run_dir, f"{mode}_{dataset}_checkpoint.csv")

    # ── 2.5 加载断点（如果存在）─────────────────────────────────
    done_ids = set()
    results  = []
    if os.path.exists(ckpt_path):
        existing_df = pd.read_csv(ckpt_path)
        done_ids = set(existing_df["query_id"].astype(str))
        results  = existing_df.to_dict("records")
        logger.info(f"  从断点恢复: 已有 {len(done_ids)} 条结果，跳过")

    pending = [q for q in queries if str(q["query_id"]) not in done_ids]
    logger.info(f"  待处理: {len(pending)} 条（已完成: {len(done_ids)} 条）")

    if not pending:
        logger.info("  所有 query 已完成，跳过推理")
    else:
        # ── 3. 并发推理 ─────────────────────────────────────────
        n_workers = min(cfg.CONCURRENT_REQUESTS, len(pending))
        logger.info(
            f"[Step 3] 开始推理（mode={mode}，{len(pending)} 条，"
            f"并发={n_workers}）…"
        )
        errors  = 0
        t_start = time.time()

        def _process_one(item):
            try:
                output = pipeline.process(item["question"])
                answer = output["answer"]
                contexts = output["contexts"]
            except Exception as e:
                logger.warning(f"  [WARN] query {item['query_id']} 失败: {e}")
                return item, "", [], True
            return item, answer, contexts, False

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_one, item): item for item in pending}
            with tqdm(total=len(pending), desc=f"[{mode}]", unit="query") as pbar:
                for future in as_completed(futures):
                    item, answer, contexts, is_error = future.result()
                    if is_error:
                        errors += 1

                    result = {
                        "query_id":         item["query_id"],
                        "question":         item["question"],
                        "answer":           answer,
                        "ground_truth":     item["ground_truth"],
                        "is_impossible":    item["is_impossible"],
                        "relevant_context": item["relevant_context"],
                    }
                    for i, ctx in enumerate(contexts):
                        result[f"context_{i}"] = ctx
                    results.append(result)
                    done_ids.add(str(item["query_id"]))
                    pbar.update(1)

                    # 定期保存断点
                    if len(done_ids) % cfg.CHECKPOINT_INTERVAL == 0:
                        pd.DataFrame(results).to_csv(
                            ckpt_path, index=False, encoding="utf-8"
                        )

        elapsed = time.time() - t_start
        logger.info(
            f"[Step 3] 推理完成: {len(results)} 条，"
            f"失败 {errors} 条，耗时 {elapsed:.1f}s "
            f"({elapsed / max(len(pending), 1):.2f}s/query)"
        )

    # ── 4. 保存最终 CSV ──────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode}_{dataset}_{ts}.csv"
    out_path = os.path.join(run_dir, filename)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"[Step 4] 结果已保存: {out_path}  ({len(df)} 行 × {len(df.columns)} 列)")

    # 删除断点文件
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("  断点文件已清理")

    # ── 5. 简要统计 ───────────────────────────────────────────
    answered = df[df["answer"].str.strip() != ""]
    logger.info(f"  有效回答率: {len(answered)}/{len(df)} ({100*len(answered)/len(df):.1f}%)")
    logger.info(f"  平均回答长度: {df['answer'].str.split().apply(len).mean():.0f} 词")
    if "is_impossible" in df.columns:
        ans_df = df[~df["is_impossible"]]
        unans_df = df[df["is_impossible"]]
        logger.info(f"  可回答: {len(ans_df)} 条，不可回答: {len(unans_df)} 条")
    ctx_cols = [c for c in df.columns if c.startswith("context_")]
    if ctx_cols:
        logger.info(f"  检索段落数: {len(ctx_cols)}")

    return out_path


# ──────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline.py --mode no_rag
  python run_baseline.py --mode naive_rag --dataset eval
  python run_baseline.py --mode naive_rag --num-queries 5
        """,
    )
    parser.add_argument(
        "--mode", required=True,
        help=f"Pipeline mode, available: {', '.join(available_modes())}",
    )
    parser.add_argument(
        "--dataset", default="dev", choices=["dev", "eval"],
        help="dev=50q  |  eval=full  (default: dev)",
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
