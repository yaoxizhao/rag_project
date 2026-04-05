"""
build_index.py — ChromaDB 索引构建脚本（一次性执行）

用法:
    # 开发模式（10k 文档，约 10k chunks，快速迭代用）
    python build_index.py --mode dev

    # 正式评测模式（50k 文档，约 50k chunks，论文实验用）
    python build_index.py --mode eval

    # 强制重建（清空已有索引后重新构建）
    python build_index.py --mode dev --force
"""
import argparse
import logging
import os
import sys
import time

# 确保项目根目录在 path 中（从任意位置都能运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402 — 须最先导入，触发 HF 镜像设置

from data.loader import chunk_corpus, load_corpus
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

# 实验模式 → (ChromaDB 集合名, 语料库文档数)
MODE_CONFIG = {
    "dev":  ("nq_dev",  cfg.DEV_CORPUS_NUM),
    "eval": ("nq_eval", cfg.EVAL_CORPUS_NUM),
}


def build_index(mode: str = "dev", force: bool = False) -> Retriever:
    """
    加载语料库、切块并写入 ChromaDB。

    Args:
        mode:  "dev"（10k docs）或 "eval"（50k docs）
        force: True 则先清空集合再重建

    Returns:
        已就绪的 Retriever 实例
    """
    if mode not in MODE_CONFIG:
        raise ValueError(f"mode 须为 'dev' 或 'eval'，收到: {mode!r}")

    collection_name, num_docs = MODE_CONFIG[mode]
    logger.info(
        f"[BuildIndex] 模式={mode} | 目标文档数={num_docs:,} | "
        f"集合={collection_name} | force={force}"
    )

    retriever = Retriever(collection_name=collection_name)

    if force:
        logger.info("[BuildIndex] --force: 清空旧索引 …")
        retriever.reset()

    if retriever.count() > 0:
        logger.info(
            f"[BuildIndex] 集合已有 {retriever.count():,} 条，"
            "跳过重建（使用 --force 重新构建）"
        )
        return retriever

    # ── 加载语料库并切块 ──────────────────────────────────────
    t0 = time.time()
    logger.info(f"[BuildIndex] 加载语料库（{num_docs:,} 篇）…")
    corpus = load_corpus(num_docs=num_docs)
    chunks = chunk_corpus(corpus)
    logger.info(
        f"[BuildIndex] 语料库就绪: {len(corpus):,} 篇 → {len(chunks):,} chunks "
        f"（耗时 {time.time() - t0:.1f}s）"
    )

    # ── 写入 ChromaDB ─────────────────────────────────────────
    t1 = time.time()
    retriever.add(chunks)
    logger.info(f"[BuildIndex] 索引写入完成，耗时 {time.time() - t1:.1f}s")

    return retriever


# ──────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for RAG baseline")
    parser.add_argument(
        "--mode", choices=["dev", "eval"], default="dev",
        help="dev=10k docs, eval=50k docs (default: dev)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Drop and rebuild the collection from scratch",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# 本地测试入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 若携带 CLI 参数则走正式构建流程
    if len(sys.argv) > 1:
        args = _parse_args()
        build_index(mode=args.mode, force=args.force)
        sys.exit(0)

    # ── 无参数：执行本地测试 ────────────────────────────────────
    SEP = "=" * 60
    TEST_COLLECTION = "nq_sanity_test"   # 独立测试集合，不污染正式数据

    print(f"\n{SEP}")
    print("build_index.py — 本地测试")
    print("流程：50 篇文档 → 切块 → 写入 ChromaDB → Top-3 检索验证")
    print(SEP)

    # ── Step 1: 加载少量语料并切块 ────────────────────────────
    print("\n[Step 1] 加载 50 篇文档并切块 …")
    corpus = load_corpus(num_docs=50)
    chunks = chunk_corpus(corpus)
    print(f"  文档数  : {len(corpus)}")
    print(f"  Chunk 数: {len(chunks)}")
    assert len(chunks) >= len(corpus), "切块数不应少于文档数"

    # ── Step 2: 初始化 Retriever 并写入 ─────────────────────
    print(f"\n[Step 2] 初始化 Retriever，写入集合 '{TEST_COLLECTION}' …")
    retriever = Retriever(collection_name=TEST_COLLECTION)
    retriever.reset()          # 确保干净的测试环境
    retriever.add(chunks)
    indexed = retriever.count()
    print(f"  索引文档数: {indexed}")
    assert indexed == len(chunks), \
        f"索引数量不符: 期望 {len(chunks)}, 实际 {indexed}"
    print("  [PASS] 写入数量验证通过")

    # ── Step 3: 向量检索测试 ──────────────────────────────────
    print(f"\n[Step 3] 检索测试（Top-{cfg.TOP_K}）…")
    test_queries = [
        "who wrote hallelujah i just love her so",        # NQ 真实问题（数据中有答案）
        "when did the flash first appear on arrow",        # NQ 真实问题
        "what is the capital of France",                   # 通用知识
    ]

    for query in test_queries:
        print(f"\n  Query : {query!r}")
        hits = retriever.retrieve(query, top_k=cfg.TOP_K)
        assert len(hits) > 0, f"检索结果为空！query={query!r}"
        for rank, hit in enumerate(hits, 1):
            score_bar = "█" * int(hit["score"] * 20)
            print(
                f"  Rank {rank} | score={hit['score']:.4f} [{score_bar:<20}] "
                f"| doc_id={hit['doc_id']}"
            )
            preview = hit["text"][:120].replace("\n", " ")
            print(f"         | {preview}…")

    print("\n  [PASS] 检索结果非空验证通过")

    # ── Step 4: 清理测试集合 ──────────────────────────────────
    retriever.reset()
    print(f"\n[Step 4] 测试集合 '{TEST_COLLECTION}' 已清空")

    print(f"\n{SEP}")
    print("build_index.py 测试全部通过！")
    print(f"{SEP}")
    print("\n下一步：运行完整索引构建：")
    print("  python build_index.py --mode dev    # 开发模式 (10k docs)")
    print("  python build_index.py --mode eval   # 正式模式 (50k docs)")
