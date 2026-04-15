"""
data/loader.py — 数据加载门面（Facade）+ 公共工具 + 数据集注册表

公开接口（所有数据集通用，切换数据集只需改 config.py）：
    chunk_text(text, chunk_size, overlap)    -> list[str]
    load_corpus(num_docs)                    -> dict[doc_id, {title, text}]
    chunk_corpus(corpus)                     -> list[{chunk_id, doc_id, text}]
    load_queries_with_qrels(num_queries)     -> list[dict]
    register_dataset(name, load_corpus_fn, load_queries_fn)
    available_datasets()                     -> list[str]

扩展方式（添加新数据集）：
    1. 在 data/ 下新建 xxx_dataset.py，实现 load_corpus() 和 load_queries()
    2. 调用 register_dataset("xxx", load_corpus_fn, load_queries_fn)
    3. config.py 中改 DATASET_NAME = "xxx"
"""
import logging
import os
import sys

# 确保从项目根目录导入 config（无论从哪里运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 数据集注册表
# ──────────────────────────────────────────────────────────

_DATASET_REGISTRY: dict[str, dict] = {}


def register_dataset(name: str, load_corpus_fn, load_queries_fn):
    """
    注册数据集。添加新数据集时调用此函数。

    Args:
        name:            数据集名称（对应 config.DATASET_NAME）
        load_corpus_fn:  函数 (num_docs=None) -> dict[doc_id, {title, text}]
        load_queries_fn: 函数 (num_queries=None) -> list[dict]
    """
    _DATASET_REGISTRY[name] = {
        "load_corpus": load_corpus_fn,
        "load_queries": load_queries_fn,
    }
    logger.debug(f"[Registry] 注册数据集: {name}")


def available_datasets() -> list[str]:
    """返回所有已注册的数据集名称。"""
    return sorted(_DATASET_REGISTRY.keys())


# ──────────────────────────────────────────────────────────
# 文本切块（公共工具，所有数据集通用）
# ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = cfg.CHUNK_SIZE,
    overlap: int = cfg.CHUNK_OVERLAP,
) -> list[str]:
    """
    将文本按词切分为等长重叠块（word-based chunking）。
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    stride = chunk_size - overlap
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride
    return chunks


# ──────────────────────────────────────────────────────────
# 语料库切块（公共工具，所有数据集通用）
# ──────────────────────────────────────────────────────────

def chunk_corpus(corpus: dict[str, dict]) -> list[dict]:
    """
    对语料库中每篇文档做切块处理（title 拼接到 text 前面）。

    Returns:
        list of dicts: [{"chunk_id": str, "doc_id": str, "text": str}]
    """
    chunks = []
    for doc_id, doc in corpus.items():
        title    = doc.get("title", "").strip()
        body     = doc["text"].strip()
        full_text = f"{title}. {body}" if title else body

        for i, chunk in enumerate(chunk_text(full_text)):
            chunks.append({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id":   doc_id,
                "text":     chunk,
            })
    return chunks


# ──────────────────────────────────────────────────────────
# 公共 API（门面函数，自动路由到对应数据集）
# ──────────────────────────────────────────────────────────

def load_corpus(num_docs=None) -> dict[str, dict]:
    """
    加载语料库。根据 config.DATASET_NAME 自动选择数据集。
    """
    name = cfg.DATASET_NAME
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"未注册的数据集: {name!r}，已注册: {available_datasets()}\n"
            f"请先调用 register_dataset() 注册该数据集"
        )
    return _DATASET_REGISTRY[name]["load_corpus"](num_docs=num_docs)


def load_queries_with_qrels(num_queries: int = cfg.DEV_QUERY_NUM) -> list[dict]:
    """
    加载测试集 queries。根据 config.DATASET_NAME 自动选择数据集。
    """
    name = cfg.DATASET_NAME
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"未注册的数据集: {name!r}，已注册: {available_datasets()}\n"
            f"请先调用 register_dataset() 注册该数据集"
        )
    return _DATASET_REGISTRY[name]["load_queries"](num_queries=num_queries)


# ──────────────────────────────────────────────────────────
# 注册内置数据集：SQuAD 2.0
# ──────────────────────────────────────────────────────────

def _squad_v2_load_corpus(num_docs=None) -> dict[str, dict]:
    """从 SQuAD 2.0 的 train + validation 中提取去重的 context 段落。"""
    from datasets import load_dataset as hf_load

    logger.info("[Corpus] 从 SQuAD 2.0 train+val 提取唯一 context 段落 …")
    seen: set[str] = set()
    corpus: dict[str, dict] = {}

    for split_name in ["train", "validation"]:
        ds = hf_load("rajpurkar/squad_v2", split=split_name)
        for item in ds:
            ctx = item["context"]
            if ctx in seen:
                continue
            seen.add(ctx)
            doc_id = f"doc_{len(corpus)}"
            corpus[doc_id] = {"title": item["title"], "text": ctx}
            if num_docs and len(corpus) >= num_docs:
                logger.info(f"[Corpus] 达到上限 {num_docs}，停止加载")
                return corpus

    logger.info(f"[Corpus] 加载完成: {len(corpus):,} 篇唯一段落")
    return corpus


def _squad_v2_load_queries(num_queries=None) -> list[dict]:
    """加载 SQuAD 2.0 验证集问题。"""
    import random
    from datasets import load_dataset as hf_load

    logger.info("[Queries] 加载 SQuAD 2.0 验证集 …")
    ds = hf_load("rajpurkar/squad_v2", split="validation")
    total = len(ds)

    rng = random.Random(cfg.RANDOM_SEED)
    indices = list(range(total))
    rng.shuffle(indices)
    if num_queries:
        indices = indices[:num_queries]

    records = []
    answerable = 0
    unanswerable = 0
    for i in indices:
        item = ds[int(i)]
        is_impossible = len(item["answers"]["text"]) == 0
        ground_truth = item["answers"]["text"][0] if not is_impossible else ""

        if is_impossible:
            unanswerable += 1
        else:
            answerable += 1

        records.append({
            "query_id":          item["id"],
            "question":          item["question"],
            "ground_truth":      ground_truth,
            "is_impossible":     is_impossible,
            "relevant_context":  item["context"],
        })

    logger.info(
        f"[Queries] 采样 {len(records)} 条 "
        f"(可回答: {answerable}, 不可回答: {unanswerable})"
    )
    return records


# 注册 SQuAD 2.0
register_dataset("squad_v2", _squad_v2_load_corpus, _squad_v2_load_queries)


# ──────────────────────────────────────────────────────────
# 扩展示例（未来数据集添加模板）
# ──────────────────────────────────────────────────────────

# 添加新数据集（如 fiqa）的步骤：
#
# 方式一：在本文件中添加（适合简单数据集）
#   def _fiqa_load_corpus(num_docs=None):
#       ...
#   def _fiqa_load_queries(num_queries=None):
#       ...
#   register_dataset("fiqa", _fiqa_load_corpus, _fiqa_load_queries)
#
# 方式二：在单独文件中添加（适合复杂数据集）
#   # data/fiqa_dataset.py
#   from data.loader import register_dataset
#   def load_corpus(num_docs=None): ...
#   def load_queries(num_queries=None): ...
#   register_dataset("fiqa", load_corpus, load_queries)
#
#   # 然后在 run_baseline.py 开头: import data.fiqa_dataset  # noqa


# ──────────────────────────────────────────────────────────
# 本地测试入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    SEP = "=" * 60

    # ── Test 0: 注册表 ──────────────────────────────────────
    print(f"\n{SEP}")
    print("Test 0 / 数据集注册表")
    print(SEP)
    print(f"  已注册数据集: {available_datasets()}")
    print(f"  当前使用: {cfg.DATASET_NAME}")
    assert cfg.DATASET_NAME in available_datasets()
    print("  [PASS] 注册表验证通过")

    # ── Test 1: chunk_text ──────────────────────────────────
    print(f"\n{SEP}")
    print("Test 1 / chunk_text()")
    print(SEP)
    sample_text = " ".join([f"word{i}" for i in range(300)])
    result_chunks = chunk_text(sample_text)
    assert len(result_chunks) >= 2
    assert all(len(c.split()) <= cfg.CHUNK_SIZE for c in result_chunks)
    print("  [PASS] chunk_text 验证通过")

    # ── Test 2: load_corpus ─────────────────────────────────
    print(f"\n{SEP}")
    print("Test 2 / load_corpus(num_docs=50)")
    print(SEP)
    corpus = load_corpus(num_docs=50)
    assert len(corpus) == 50
    sample = next(iter(corpus.values()))
    print(f"  示例 title: {sample['title'][:60]}")
    print("  [PASS] load_corpus 验证通过")

    # ── Test 3: chunk_corpus ────────────────────────────────
    print(f"\n{SEP}")
    print("Test 3 / chunk_corpus()")
    print(SEP)
    all_chunks = chunk_corpus(corpus)
    assert all("chunk_id" in c for c in all_chunks)
    print(f"  {len(corpus)} 篇 → {len(all_chunks)} chunks")
    print("  [PASS] chunk_corpus 验证通过")

    # ── Test 4: load_queries_with_qrels ─────────────────────
    print(f"\n{SEP}")
    print("Test 4 / load_queries_with_qrels(num_queries=10)")
    print(SEP)
    queries = load_queries_with_qrels(num_queries=10)
    assert len(queries) == 10
    ans = sum(1 for q in queries if not q["is_impossible"])
    print(f"  可回答: {ans}, 不可回答: {10 - ans}")
    for q in queries[:2]:
        tag = "UNANS" if q["is_impossible"] else "ANS"
        print(f"  [{tag}] {q['question'][:60]}  GT={q['ground_truth'][:30]}")
    print("  [PASS] load_queries_with_qrels 验证通过")

    print(f"\n{SEP}")
    print("全部测试通过！")
    print(SEP)
