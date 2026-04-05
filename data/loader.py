"""
data/loader.py — BeIR/NQ 数据加载、采样与切块

数据源：通过 beir 库下载原始 JSONL 文件（绕开 HuggingFace 加载脚本限制）。
下载后缓存在 cfg.HF_CACHE_DIR/beir_datasets/nq/，后续实验直接读本地文件。

公开接口：
    chunk_text(text, chunk_size, overlap) -> list[str]
    load_corpus(num_docs)                -> dict[doc_id, {title, text}]
    chunk_corpus(corpus)                 -> list[{chunk_id, doc_id, text}]
    load_queries_with_qrels(num_queries) -> list[{query_id, question,
                                                   ground_truth,
                                                   relevant_doc_ids}]
"""
import gzip
import json
import logging
import os
import random
import sys

# 确保从项目根目录导入 config（无论从哪里运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)

# 本地缓存目录（与 HuggingFace 缓存并列，避免污染）
_BEIR_NQ_DIR = os.path.join(cfg.HF_CACHE_DIR, "beir_nq")


# ──────────────────────────────────────────────────────────
# 内部：数据集下载与文件路径
# ──────────────────────────────────────────────────────────

def _hf_download(repo_id: str, filename: str, local_path: str) -> None:
    """
    通过 hf-mirror.com 下载单个文件到 local_path。
    若文件已存在则跳过。
    """
    if os.path.isfile(local_path):
        logger.info(f"[Cache] 命中: {local_path}")
        return
    from huggingface_hub import hf_hub_download
    logger.info(f"[HF] 下载 {repo_id}/{filename} → {local_path} …")
    tmp = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cfg.HF_CACHE_DIR,
        endpoint=cfg.HF_ENDPOINT,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # hf_hub_download 返回缓存内副本，软链或拷贝到目标位置
    if not os.path.isfile(local_path):
        import shutil
        shutil.copy2(tmp, local_path)
    logger.info(f"[HF] 下载完成: {local_path}")


def _ensure_beir_nq() -> str:
    """
    确保 BeIR/NQ 三个核心文件已下载并解压到 _BEIR_NQ_DIR：
        corpus.jsonl      — 来自 BeIR/nq :: corpus.jsonl.gz
        queries.jsonl     — 来自 BeIR/nq :: queries.jsonl.gz
        qrels/test.tsv    — 来自 BeIR/nq-qrels :: test.tsv
    下载仅执行一次，后续使用本地缓存。
    """
    os.makedirs(_BEIR_NQ_DIR, exist_ok=True)
    qrels_dir = os.path.join(_BEIR_NQ_DIR, "qrels")
    os.makedirs(qrels_dir, exist_ok=True)

    corpus_jsonl  = os.path.join(_BEIR_NQ_DIR, "corpus.jsonl")
    queries_jsonl = os.path.join(_BEIR_NQ_DIR, "queries.jsonl")
    qrels_tsv     = os.path.join(qrels_dir, "test.tsv")

    def _download_and_decompress(repo_id: str, gz_filename: str, out_jsonl: str):
        gz_path = out_jsonl + ".gz"
        _hf_download(repo_id, gz_filename, gz_path)
        if not os.path.isfile(out_jsonl):
            logger.info(f"[Decompress] {gz_path} → {out_jsonl} …")
            with gzip.open(gz_path, "rb") as fin, open(out_jsonl, "wb") as fout:
                import shutil
                shutil.copyfileobj(fin, fout)
            logger.info(f"[Decompress] 完成: {out_jsonl}")

    # corpus
    if not os.path.isfile(corpus_jsonl):
        _download_and_decompress("BeIR/nq", "corpus.jsonl.gz", corpus_jsonl)
    else:
        logger.info(f"[Cache] corpus.jsonl 已存在，跳过下载")

    # queries
    if not os.path.isfile(queries_jsonl):
        _download_and_decompress("BeIR/nq", "queries.jsonl.gz", queries_jsonl)
    else:
        logger.info(f"[Cache] queries.jsonl 已存在，跳过下载")

    # qrels (plain TSV, not gzipped)
    if not os.path.isfile(qrels_tsv):
        _hf_download("BeIR/nq-qrels", "test.tsv", qrels_tsv)
    else:
        logger.info(f"[Cache] qrels/test.tsv 已存在，跳过下载")

    return _BEIR_NQ_DIR


def _nq_corpus_file()  -> str: return os.path.join(_ensure_beir_nq(), "corpus.jsonl")
def _nq_queries_file() -> str: return os.path.join(_ensure_beir_nq(), "queries.jsonl")
def _nq_qrels_file()   -> str: return os.path.join(_ensure_beir_nq(), "qrels", "test.tsv")


# ──────────────────────────────────────────────────────────
# 内部：蓄水池采样（Reservoir Sampling）
# ──────────────────────────────────────────────────────────

def _reservoir_sample_jsonl(
    filepath: str,
    n: int,
    seed: int = cfg.RANDOM_SEED,
) -> list[dict]:
    """
    对大型 JSONL 文件做蓄水池采样，内存复杂度 O(n)，时间复杂度 O(文件行数)。
    保证相同 seed 下结果完全一致。
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    i = -1
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            row = json.loads(line)
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = row
    return reservoir


# ──────────────────────────────────────────────────────────
# 文本切块
# ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = cfg.CHUNK_SIZE,
    overlap: int = cfg.CHUNK_OVERLAP,
) -> list[str]:
    """
    将文本按词切分为等长重叠块（word-based chunking）。

    Args:
        text:       原始文本字符串
        chunk_size: 每块的词数（默认 256 词 ≈ 340 tokens）
        overlap:    相邻块之间的词重叠数（默认 32 词）

    Returns:
        list of chunk strings（至少返回 1 个元素）
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
# 语料库加载
# ──────────────────────────────────────────────────────────

def load_corpus(num_docs: int = cfg.DEV_CORPUS_NUM) -> dict[str, dict]:
    """
    从 BeIR/NQ corpus.jsonl 中随机采样固定子集（蓄水池算法）。

    Args:
        num_docs: 采样文档数，默认 DEV_CORPUS_NUM=10_000

    Returns:
        dict: {doc_id: {"title": str, "text": str}}
    """
    corpus_file = _nq_corpus_file()
    logger.info(f"[Corpus] 蓄水池采样 {num_docs:,} 篇（seed={cfg.RANDOM_SEED}）…")
    logger.info(f"[Corpus] 来源: {corpus_file}")

    sampled = _reservoir_sample_jsonl(corpus_file, num_docs, cfg.RANDOM_SEED)

    corpus = {
        row["_id"]: {
            "title": row.get("title", ""),
            "text":  row["text"],
        }
        for row in sampled
    }
    logger.info(f"[Corpus] 采样完成: {len(corpus):,} 篇")
    return corpus


# ──────────────────────────────────────────────────────────
# 语料库切块
# ──────────────────────────────────────────────────────────

def chunk_corpus(corpus: dict[str, dict]) -> list[dict]:
    """
    对语料库中每篇文档做切块处理（title 拼接到 text 前面）。

    Returns:
        list of dicts: [{"chunk_id": str, "doc_id": str, "text": str}]
        chunk_id 格式: "{doc_id}_{chunk_index}"
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
# Query + Qrels 加载
# ──────────────────────────────────────────────────────────

def load_queries_with_qrels(num_queries: int = cfg.DEV_QUERY_NUM) -> list[dict]:
    """
    加载 BeIR/NQ 测试集 queries，并通过 qrels 关联 ground-truth 文档文本。

    ground_truth 取该 query 第一条相关文档（score>0）的文本，
    供 Ragas 的 ground_truths 字段使用。

    Args:
        num_queries: 采样 query 数，默认 DEV_QUERY_NUM=200

    Returns:
        list of dicts:
            query_id        — query 原始 ID（字符串）
            question        — 问题文本
            ground_truth    — 第一条相关文档的文本
            relevant_doc_ids— 所有相关文档 ID 列表
    """
    # 1. 读取 qrels（仅保留正相关，score > 0）
    qrels_file = _nq_qrels_file()
    logger.info(f"[Qrels] 读取 {qrels_file} …")
    qid_to_cids: dict[str, list[str]] = {}
    with open(qrels_file, encoding="utf-8") as f:
        header = f.readline()  # 跳过表头 (query-id  corpus-id  score)
        logger.debug(f"[Qrels] 表头: {header.strip()}")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qid_to_cids.setdefault(qid, []).append(cid)
    logger.info(f"[Qrels] 正相关条目 query 数: {len(qid_to_cids):,}")

    # 2. 读取 queries
    queries_file = _nq_queries_file()
    logger.info(f"[Queries] 读取 {queries_file} …")
    qid_to_question: dict[str, str] = {}
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid_to_question[row["_id"]] = row["text"]
    logger.info(f"[Queries] 全量 queries: {len(qid_to_question):,} 条")

    # 3. 可复现采样
    valid_qids = [qid for qid in qid_to_cids if qid in qid_to_question]
    rng = random.Random(cfg.RANDOM_SEED)
    sampled_qids = rng.sample(valid_qids, min(num_queries, len(valid_qids)))

    # 4. 找到所有需要的语料库文档（用于 ground truth 文本）
    needed_cids: set[str] = set()
    for qid in sampled_qids:
        needed_cids.update(qid_to_cids[qid])
    logger.info(
        f"[Corpus] 需要获取 ground-truth 的文档: {len(needed_cids)} 篇，"
        "流式扫描 corpus.jsonl …"
    )

    # 5. 流式扫描 corpus.jsonl，找到 needed_cids（找齐后提前退出）
    corpus_file = _nq_corpus_file()
    cid_to_text: dict[str, str] = {}
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["_id"] in needed_cids:
                cid_to_text[row["_id"]] = row["text"]
            if len(cid_to_text) >= len(needed_cids):
                break  # 找齐了，提前终止
    logger.info(f"[Corpus] ground-truth 命中: {len(cid_to_text)}/{len(needed_cids)}")

    # 6. 组装 records
    records = []
    for qid in sampled_qids:
        relevant_cids = qid_to_cids[qid]
        ground_truth  = next(
            (cid_to_text[cid] for cid in relevant_cids if cid in cid_to_text),
            "",  # 极少数情况：相关文档未出现在 corpus.jsonl 中
        )
        records.append({
            "query_id":         qid,
            "question":         qid_to_question[qid],
            "ground_truth":     ground_truth,
            "relevant_doc_ids": relevant_cids,
        })

    logger.info(f"[Queries] 最终样本数: {len(records)}")
    return records


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

    # ── Test 1: chunk_text（无需下载）──────────────────────
    print(f"\n{SEP}")
    print("Test 1 / chunk_text()  [无需下载]")
    print(SEP)
    sample_text = " ".join([f"word{i}" for i in range(300)])
    result_chunks = chunk_text(sample_text)
    print(f"  输入: 300 词")
    print(f"  输出: {len(result_chunks)} 个 chunk")
    for i, c in enumerate(result_chunks):
        wc = len(c.split())
        overlap_ok = True  # 验证重叠
        print(f"  chunk[{i}]: {wc} 词  首10词→ {' '.join(c.split()[:10])}…")
    assert all(len(c.split()) <= cfg.CHUNK_SIZE for c in result_chunks)
    # 验证重叠：chunk[1] 的前 overlap 词应与 chunk[0] 的后 overlap 词相同
    if len(result_chunks) >= 2:
        tail0 = result_chunks[0].split()[-cfg.CHUNK_OVERLAP:]
        head1 = result_chunks[1].split()[:cfg.CHUNK_OVERLAP]
        assert tail0 == head1, f"Overlap 不正确: {tail0} != {head1}"
    print("  [PASS] chunk_text 验证通过")

    # ── Test 2: load_corpus（会触发下载，首次较慢）─────────
    print(f"\n{SEP}")
    print("Test 2 / load_corpus(num_docs=50)  [首次运行会下载数据集]")
    print(SEP)
    corpus = load_corpus(num_docs=50)
    print(f"  加载文档数: {len(corpus)}")
    assert len(corpus) == 50, f"期望50篇，实际{len(corpus)}篇"
    sample_id  = next(iter(corpus))
    sample_doc = corpus[sample_id]
    print(f"  示例 doc_id : {sample_id}")
    print(f"  示例 title  : {sample_doc['title'][:80]}")
    print(f"  示例 text   : {sample_doc['text'][:120]}…")
    print("  [PASS] load_corpus 验证通过")

    # ── Test 3: chunk_corpus ────────────────────────────────
    print(f"\n{SEP}")
    print("Test 3 / chunk_corpus()")
    print(SEP)
    all_chunks = chunk_corpus(corpus)
    print(f"  {len(corpus)} 篇文档 → {len(all_chunks)} 个 chunk")
    print(f"  平均每篇 chunk 数: {len(all_chunks)/len(corpus):.1f}")
    print(f"  chunk[0] chunk_id : {all_chunks[0]['chunk_id']}")
    print(f"  chunk[0] text     : {all_chunks[0]['text'][:120]}…")
    assert all("chunk_id" in c and "doc_id" in c and "text" in c for c in all_chunks)
    print("  [PASS] chunk_corpus 验证通过")

    # ── Test 4: load_queries_with_qrels ────────────────────
    print(f"\n{SEP}")
    print("Test 4 / load_queries_with_qrels(num_queries=5)")
    print(SEP)
    queries = load_queries_with_qrels(num_queries=5)
    print(f"  加载 query 数: {len(queries)}")
    assert len(queries) == 5
    for q in queries:
        print(f"\n  Q  [{q['query_id']}]: {q['question']}")
        gt = q['ground_truth'][:100].replace('\n', ' ')
        print(f"  GT (前100字符)  : {gt}…")
        print(f"  相关文档数      : {len(q['relevant_doc_ids'])}")
    assert all(q["question"] for q in queries), "存在空 question！"
    print("\n  [PASS] load_queries_with_qrels 验证通过")

    print(f"\n{SEP}")
    print("全部测试通过！data/loader.py 工作正常。")
    print(SEP)
