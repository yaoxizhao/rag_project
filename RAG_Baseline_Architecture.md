# RAG Baseline Architecture

> **用途**：本文档是项目的唯一技术准则，所有代码实现必须与此保持一致。
> **版本**：v1.0 | **日期**：2026-04-04

---

## 1. 项目目标

构建一个 **Vanilla RAG（标准检索增强生成）Baseline** 系统，用于研究幻觉降低（Hallucination Reduction）。系统需满足：

- **高度模块化**：核心组件解耦，支持后续插拔微创新模块（Reranker、Query Rewriting 等）
- **高可复现性**：固定随机种子，版本锁定，结果可追溯
- **算力适配**：2x RTX 3090（48GB 总显存），优先本地部署

---

## 2. 技术选型

| 组件 | 方案 | 版本/规格 | 说明 |
|------|------|----------|------|
| **RAG 框架** | Native Python | — | 零黑盒，接口完全自定义 |
| **向量数据库** | ChromaDB | latest | 本地持久化，无需独立服务器 |
| **切块策略** | Fixed-size Chunking | size=256, overlap=32 | 适配 NQ 短问题特征 |
| **生成模型** | Qwen2.5-7B-Instruct | BF16 | vLLM 服务，OpenAI 兼容接口 |
| **部署方式** | vLLM | TP=2, port=8000 | 张量并行，双卡吞吐最优 |
| **嵌入模型** | BAAI/bge-m3 | — | 本地运行，MTEB 顶级性能 |
| **评测数据集** | BeIR/Natural Questions | 子集采样 | seed=42 保证可复现 |
| **评估框架** | Ragas | latest | Faithfulness 等核心指标 |
| **Ragas Judge** | DeepSeek API | deepseek-chat | 高性价比评测模型 |

### 采样策略

| 阶段 | Query 数量 | 语料库大小 | 用途 |
|------|-----------|-----------|------|
| 开发/调试 | 200 | 10,000 docs | 快速迭代验证 |
| 正式评测 | 500 | 50,000 docs | 论文最终结果 |

---

## 3. 目录结构

```
/data/zhaoyaoxi/rag_project/
│
├── config.py                   # [核心] 全局配置，唯一超参来源
├── serve_model.sh              # vLLM 启动脚本（TP=2，port=8000）
├── build_index.py              # [一次性] ChromaDB 索引构建
├── run_baseline.py             # [入口] 实验主脚本
├── evaluate.py                 # [入口] Ragas 评估脚本
│
├── rag/                        # ★ 核心 RAG 模块（可插拔）
│   ├── __init__.py
│   ├── retriever.py            # 检索器：ChromaDB + bge-m3
│   ├── augmenter.py            # 增强器：Prompt 构建与上下文注入
│   └── generator.py            # 生成器：vLLM API 调用封装
│
├── data/
│   ├── __init__.py
│   └── loader.py               # BeIR/NQ 加载 + 可复现子集采样
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # Ragas 指标封装
│
├── results/                    # 实验输出（CSV，按时间戳命名）
│   └── .gitkeep
│
└── RAG_Baseline_Architecture.md
```

---

## 4. 核心模块规范

### 4.1 `config.py` — 全局配置

所有超参数、路径、模型名称**必须**在此文件集中定义，其他模块通过 `from config import cfg` 读取，禁止硬编码。

```python
# 关键配置项（示意）
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
TOP_K = 3
EMBED_MODEL = "BAAI/bge-m3"
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen2.5-7B-Instruct"
CHROMA_DIR = "./chroma_db"
HF_CACHE_DIR = "/data/zhaoyaoxi/huggingface_cache"
RANDOM_SEED = 42
DEV_QUERY_NUM = 200
DEV_CORPUS_NUM = 10_000
EVAL_QUERY_NUM = 500
EVAL_CORPUS_NUM = 50_000
```

### 4.2 `rag/retriever.py` — 检索器接口

```python
class Retriever:
    def __init__(self, collection_name: str): ...
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...
    # 返回格式：[{"doc_id": str, "text": str, "score": float}]
```

> 插拔点：后续可替换为 `HybridRetriever`、`RerankerRetriever` 等，接口不变。

### 4.3 `rag/augmenter.py` — 增强器接口

```python
class Augmenter:
    def build_prompt(self, query: str, contexts: list[str]) -> str: ...
    # no_rag 模式：contexts=[]，直接返回裸 query prompt
    # naive_rag 模式：将 contexts 注入标准 RAG prompt 模板
```

> 插拔点：后续可替换为 `CoTAugmenter`、`SelfRAGAugmenter` 等。

### 4.4 `rag/generator.py` — 生成器接口

```python
class Generator:
    def __init__(self, base_url: str, model: str): ...
    def generate(self, prompt: str) -> str: ...
    # 通过 OpenAI 兼容接口调用 vLLM，返回纯文本答案
```

### 4.5 `run_baseline.py` — 实验入口

```bash
# 支持两种模式
python run_baseline.py --mode no_rag    # 纯 LLM，无检索
python run_baseline.py --mode naive_rag # 标准 RAG
```

输出：`results/{mode}_{timestamp}.csv`，包含列：`[query_id, question, answer, contexts, ground_truth]`

### 4.6 `evaluate.py` — 评估入口

```bash
python evaluate.py --input results/naive_rag_20260404.csv
```

输出：终端打印 + `results/{input_name}_ragas.json`，包含：
- `faithfulness`（忠实度，核心幻觉指标）
- `answer_relevancy`（答案相关性）
- `context_recall`（上下文召回率）
- `context_precision`（上下文精确率）

---

## 5. 数据流

```
                    ┌──────────────────────────────────────────────┐
                    │               run_baseline.py                 │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
                    │              data/loader.py                   │
                    │   BeIR/NQ  →  固定种子采样  →  (query, gt)   │
                    └───────────────────┬──────────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
    ┌──────────▼─────────┐  ┌───────────▼──────────┐            │
    │  rag/retriever.py  │  │  rag/augmenter.py    │            │
    │  ChromaDB + bge-m3 │→ │  Prompt 构建         │            │
    └────────────────────┘  └───────────┬──────────┘            │
                                        │                        │
                            ┌───────────▼──────────┐            │
                            │  rag/generator.py    │            │
                            │  vLLM / Qwen2.5-7B   │            │
                            └───────────┬──────────┘            │
                                        │                        │
                    ┌───────────────────▼──────────────────────  │
                    │         results/{mode}_{ts}.csv  ◄─────────┘
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
                    │              evaluate.py                      │
                    │         Ragas → DeepSeek Judge                │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
                    │       results/{name}_ragas.json               │
                    │  faithfulness / answer_relevancy / recall...  │
                    └──────────────────────────────────────────────┘
```

---

## 6. 可复现性规范

| 措施 | 实现方式 |
|------|---------|
| 随机种子 | 所有采样统一使用 `RANDOM_SEED=42` |
| 模型版本 | `requirements.txt` 锁定所有包版本 |
| 数据集版本 | HuggingFace datasets 指定 revision |
| 结果追溯 | 输出 CSV 文件名包含时间戳 + 模式标识 |
| 环境隔离 | Python venv，路径写入 `config.py` |

---

## 7. 后续创新模块插拔示例

本 Baseline 完成后，可通过以下方式插拔新模块，**无需修改主流程**：

```python
# run_baseline.py 中
if args.mode == "naive_rag":
    retriever = Retriever(...)           # 替换为 HybridRetriever(...)
    augmenter = Augmenter(...)           # 替换为 QueryRewriteAugmenter(...)
    generator = Generator(...)           # 接口不变
```

---

## 8. 环境依赖（待生成 requirements.txt）

```
chromadb
sentence-transformers      # bge-m3 本地推理
openai                     # vLLM OpenAI 兼容接口
datasets                   # BeIR/NQ 加载
ragas
deepseek                   # 或 openai 兼容，用于 Ragas judge
pandas
tqdm
```
