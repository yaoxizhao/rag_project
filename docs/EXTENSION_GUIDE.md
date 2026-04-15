# 扩展指南

> 核心原则：**只加文件（子文件夹），不改旧代码**

## 架构总览

```
data/loader.py          ← 数据集注册表，切换数据集改 config.py
rag/pipeline.py         ← 流水线注册表，--mode 自动路由
rag/retriever.py        ← 检索器基类（DenseRetriever）
rag/augmenter.py        ← Prompt 构建器
rag/generator.py        ← LLM 生成器
run_baseline.py         ← 实验主脚本（不需要改）
```

## 文件组织规则

**每个新创新点的代码必须放在 `rag/` 下的独立子文件夹中**，不要在 `rag/` 根目录下创建散文件。

```
rag/
├── __init__.py
├── pipeline.py         ← 基类 + 注册表（不改动已有代码，只追加注册）
├── retriever.py        ← baseline 检索器（不改动）
├── augmenter.py        ← baseline prompt 构建器（不改动）
├── generator.py        ← baseline 生成器（不改动）
│
├── crag/               ← CRAG 论文方法
│   ├── __init__.py
│   ├── evaluator.py    ← 检索评估器
│   ├── refiner.py      ← 知识精炼
│   └── pipeline.py     ← CRAGPipeline（继承 rag.pipeline.Pipeline，重写 process()）
│
├── self_rag/           ← Self-RAG 论文方法
│   ├── __init__.py
│   └── pipeline.py     ← SelfRAGPipeline
│
└── crag_self_rag/      ← 缝合方案：CRAG + Self-RAG
    ├── __init__.py
    └── pipeline.py     ← CRAGSelfRAGPipeline（编排两个子文件夹的模块）
```

**关系说明**：
- `rag/pipeline.py`：基类 `Pipeline`（默认流程）+ 注册表 `_REGISTRY`（所有 mode 入口）
- 子文件夹的 `pipeline.py`：继承基类，重写 `process()` 实现论文特有的流程
- 注册在 `rag/pipeline.py` 底部追加，import 路径为 `from rag.crag.pipeline import CRAGPipeline`

## 一、添加新 RAG 策略（创新点）

### 方式 A：换检索器（如 Hybrid Retrieval、Reranker）

1. 新建子文件夹 `rag/hybrid/`，内含 `__init__.py` 和 `retriever.py`，保持 `retrieve(query, top_k)` 接口：

```python
# rag/hybrid/retriever.py
class HybridRetriever:
    def __init__(self, collection_name):
        self.dense = Retriever(collection_name)
        self.bm25 = ...  # 你的 BM25 实现

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        # 返回 [{"doc_id": str, "text": str, "score": float}, ...]
        ...
```

2. 在 `rag/pipeline.py` 底部注册：

```python
@register_pipeline("hybrid_rag")
def _create_hybrid_rag(collection_name=None, **kwargs):
    from rag.hybrid.retriever import HybridRetriever
    from rag.augmenter import Augmenter
    from rag.generator import Generator

    return Pipeline(
        retriever=HybridRetriever(collection_name),
        augmenter=Augmenter(),
        generator=Generator(),
    )
```

3. 运行：`python run_baseline.py --mode hybrid_rag --dataset eval`

### 方式 B：换 Prompt 策略（如 Self-RAG）

1. 新建子文件夹 `rag/self_rag/`，内含 `__init__.py` 和 `augmenter.py`，保持 `build_prompt(query, contexts)` 接口

2. 注册新的 pipeline，换掉 augmenter：

```python
@register_pipeline("self_rag")
def _create_self_rag(collection_name=None, **kwargs):
    from rag.self_rag.augmenter import SelfRAGAugmenter
    ...
    return Pipeline(retriever=..., augmenter=SelfRAGAugmenter(), generator=...)
```

### 方式 C：换整个流程（如 CRAG、多轮检索）

新建子文件夹 `rag/crag/`，继承 `Pipeline` 并重写 `process()`：

```python
# rag/crag/pipeline.py
from rag.pipeline import Pipeline

class CRAGPipeline(Pipeline):
    def process(self, query):
        # 自定义流程：检索 → 评估 → 决定是否重新检索 → 生成
        ...
```

然后在 `rag/pipeline.py` 中注册：

```python
@register_pipeline("crag")
def _create_crag(collection_name=None, **kwargs):
    from rag.crag.pipeline import CRAGPipeline
    return CRAGPipeline(collection_name=collection_name)
```

### 方式 D：缝合两个已有方法

新建子文件夹 `rag/crag_self_rag/`，组合调用两个已有方法的模块：

```python
# rag/crag_self_rag/pipeline.py
from rag.pipeline import Pipeline
from rag.crag.evaluator import CRAGEvaluator
from rag.self_rag.augmenter import SelfRAGAugmenter

class CRAGSelfRAGPipeline(Pipeline):
    def process(self, query):
        # 先用 CRAG 评估检索质量，再用 Self-RAG 的 prompt 策略生成
        ...
```

然后在 `rag/pipeline.py` 中注册。

## 二、添加新数据集

1. 在 `data/` 下新建文件，实现两个函数：

```python
# data/fiqa_dataset.py
from data.loader import register_dataset

def load_corpus(num_docs=None):
    # 返回 dict: {doc_id: {"title": str, "text": str}}
    ...

def load_queries(num_queries=None):
    # 返回 list[dict]，必须含以下字段：
    # query_id, question, ground_truth, is_impossible, relevant_context
    ...

register_dataset("fiqa", load_corpus, load_queries)
```

2. `config.py` 改 `DATASET_NAME = "fiqa"`

3. 在 `run_baseline.py` 和 `build_index.py` 开头加一行触发注册：

```python
import data.fiqa_dataset  # noqa
```

## 三、关键接口约定

| 组件 | 必须实现的方法 | 返回格式 |
|------|--------------|---------|
| Retriever | `retrieve(query, top_k)` | `[{"doc_id", "text", "score"}]` |
| Augmenter | `build_prompt(query, contexts)` | `str` |
| Generator | `generate(prompt)` | `str` |
| Pipeline | `process(query)` | `{"answer", "contexts"}` |
| 数据集 | `load_corpus(num_docs)` | `{doc_id: {"title", "text"}}` |
| 数据集 | `load_queries(num_queries)` | `[{"query_id", "question", "ground_truth", "is_impossible", "relevant_context"}]` |
