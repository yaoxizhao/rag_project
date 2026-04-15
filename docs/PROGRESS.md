# Research Progress Log - 科研代码进度记录

按时间倒序记录每次代码改动。

---

## 2026-04-15 — CRAG-Lite 实现

**内容**：
- 实现 CRAG 论文（Yan et al. 2024）的轻量版本 CRAG-Lite
- 零训练路径：直接复用 bge-m3 余弦相似度分数作为检索评估指标

**新增文件**：
- `rag/crag/__init__.py` — 包标记
- `rag/crag/evaluator.py` — 检索评估器（三档动作：correct/incorrect/ambiguous）
- `rag/crag/refiner.py` — 知识精炼（decompose-then-recompose）
- `rag/crag/pipeline.py` — CRAGPipeline 类（继承 Pipeline，重写 process）

**注册 mode**：`crag_lite`（在 rag/pipeline.py 追加注册）

**核心逻辑**：
```
检索 top-5 → 评估器判断（阈值 upper=0.50, lower=0.30）
  ├─ correct    → 知识精炼（分句→过滤→重组）→ RAG prompt → 生成
  ├─ incorrect  → 拒绝 prompt → "I cannot answer..."
  └─ ambiguous  → 知识精炼 + 不确定性提示 → 生成
```

**状态**：代码已完成，待实验验证

---



## 2026-04-14 — Baseline 实验完成

**内容**：
- 完成 `no_rag` 和 `naive_rag` 两组 baseline 实验（SQuAD v2, 600 queries）
- 评测指标通过 Ragas + GLM-4-Flash 计算

**关键结果**：

| 指标 | no_rag | naive_rag |
|------|--------|-----------|
| faithfulness | N/A | 0.901 |
| answer_correctness | 0.503 | 0.638 |
| context_recall | N/A | 0.899 |
| context_precision | N/A | 0.808 |
| hallucination_rate | 1.000 | 0.936 |
| unanswerable_abstention_rate | 0.000 | 0.064 |

**分析**：
- naive_rag 检索质量高（context_precision=0.81, context_recall=0.90）
- 但幻觉率仍然很高（93.6%），模型几乎不拒绝回答不可答问题
- **核心瓶颈**：缺乏"自知之明"——检索到的上下文无法回答问题时，模型仍强行作答
- **改进方向**：需要引入判断机制（如 CRAG 的检索评估器、Self-RAG 的自反思token）

**实验配置**：
- 数据集：SQuAD v2（600 queries，其中 313 不可答）
- 检索：BAAI/bge-m3 + ChromaDB，top_k=5, chunk_size=256, chunk_overlap=32
- 生成：Qwen2.5-7B-Instruct (vLLM, TP=2)
- Judge：GLM-4-Flash（Ragas）
