"""
rag/retriever.py — 向量检索模块

封装 ChromaDB 初始化、文档插入（add）和向量检索（retrieve）。
Embedding 模型：BAAI/bge-m3（via sentence-transformers，dense 检索模式）
向量数据库：ChromaDB（本地持久化，cosine 空间）

公开接口（严格遵循架构文档规范）：
    Retriever(collection_name)
        .add(chunks)
        .retrieve(query, top_k) -> list[{"doc_id", "text", "score"}]
        .count() -> int
        .reset()

── 架构插拔点 ────────────────────────────────────────────────
后续可替换为 HybridRetriever（BM25+Dense）、RerankerRetriever 等，
只需保持 retrieve() 的参数和返回格式不变。
─────────────────────────────────────────────────────────────
"""
import logging
import os
import sys

# config 须在 sentence_transformers 之前导入，以触发 HF 镜像环境变量设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Retriever:
    """
    基于 ChromaDB + BAAI/bge-m3 的密集向量检索器。

    使用方法：
        retriever = Retriever("nq_dev")
        retriever.add(chunks)           # 建索引
        hits = retriever.retrieve(q)    # 检索
    """

    def __init__(self, collection_name: str):
        """
        初始化 Embedding 模型和 ChromaDB 集合。

        Args:
            collection_name: ChromaDB 集合名称（建议按实验版本命名）
        """
        # ── Embedding 模型 ──────────────────────────────────────
        logger.info(f"[Retriever] 加载 Embedding 模型: {cfg.EMBED_MODEL} …")
        import torch
        device = cfg.EMBED_GPU if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(
            cfg.EMBED_MODEL,
            cache_folder=cfg.HF_CACHE_DIR,
            device=device,
        )
        logger.info(f"[Retriever] 模型加载完成，设备: {device}")

        # ── ChromaDB 客户端 ──────────────────────────────────────
        os.makedirs(cfg.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
        # cosine 空间：distance = 1 − cosine_similarity（越小越相似）
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"[Retriever] 集合 '{collection_name}' 就绪，"
            f"已有文档数: {self._collection.count()}"
        )

    # ──────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────

    def add(
        self,
        chunks: list[dict],
        batch_size: int = cfg.EMBED_BATCH_SIZE,
    ) -> None:
        """
        将切块文档嵌入并存入 ChromaDB。

        Args:
            chunks:     list of {"chunk_id": str, "doc_id": str, "text": str}
            batch_size: 单次 encode 批大小（RTX 3090 适配）

        说明：
            - 若集合已有 >= len(chunks) 条记录，视为已完成，直接跳过。
            - 若集合有部分数据（断点续传场景），过滤掉已存在的 chunk_id。
            - 强制重建请先调用 reset()。
        """
        if not chunks:
            logger.warning("[Retriever.add] chunks 为空，跳过")
            return

        existing_count = self._collection.count()

        # 全量已存在 → 跳过
        if existing_count >= len(chunks):
            logger.info(
                f"[Retriever.add] 集合已有 {existing_count} 条（≥ {len(chunks)}），跳过"
            )
            return

        # 断点续传：找出尚未写入的 chunk_id
        if existing_count > 0:
            logger.info(f"[Retriever.add] 检测到部分索引（{existing_count} 条），续传 …")
            existing_ids = set(self._collection.get(include=[])["ids"])
            new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        else:
            new_chunks = chunks

        logger.info(f"[Retriever.add] 待写入: {len(new_chunks)} chunks …")

        # 分批 Encode → Add
        for i in tqdm(
            range(0, len(new_chunks), batch_size),
            desc="Embedding & indexing",
            unit="batch",
        ):
            batch      = new_chunks[i : i + batch_size]
            texts      = [c["text"]     for c in batch]
            ids        = [c["chunk_id"] for c in batch]
            metadatas  = [{"doc_id": c["doc_id"]} for c in batch]

            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # L2 归一化，配合 cosine 空间
            ).tolist()

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        logger.info(
            f"[Retriever.add] 完成，集合总文档数: {self._collection.count()}"
        )

    def retrieve(self, query: str, top_k: int = cfg.TOP_K) -> list[dict]:
        """
        对单条 query 进行向量检索，返回 Top-K 结果。

        Args:
            query:  查询字符串
            top_k:  返回结果数量

        Returns:
            list of dict:
                doc_id — 原始文档 ID（来自 corpus）
                text   — chunk 文本
                score  — 余弦相似度 ∈ [0, 1]（越大越相关）
        """
        n = min(top_k, self._collection.count())
        if n == 0:
            logger.warning("[Retriever.retrieve] 集合为空，返回空列表")
            return []

        query_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "doc_id": meta["doc_id"],
                "text":   doc,
                "score":  round(1.0 - float(dist), 4),  # cosine dist → similarity
            })
        return hits

    # ──────────────────────────────────────────────────────────
    # 辅助接口
    # ──────────────────────────────────────────────────────────

    def count(self) -> int:
        """返回集合中当前文档数量。"""
        return self._collection.count()

    def reset(self) -> None:
        """清空集合（用于重建索引或测试清理）。"""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[Retriever.reset] 集合 '{name}' 已清空")
