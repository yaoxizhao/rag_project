"""
rag/pipeline.py — RAG 流水线抽象层

通过 Pipeline 类组合 Retriever / Augmenter / Generator，
支持通过注册表模式扩展新的实验模式（--mode）。

公开接口：
    create_pipeline(mode, collection_name)  -> Pipeline
    register_pipeline(name)(factory_fn)     # 装饰器，注册新 pipeline
    available_modes()                       -> list[str]

扩展方式（添加新创新点）：
    1. 在 rag/ 下新建检索器/增强器/生成器（如 hybrid_retriever.py）
    2. 在本文件中注册新的 pipeline 工厂函数（见底部示例）
    3. run_baseline.py --mode=新名称 自动生效，无需改旧代码

    也可在 rag/ 的其他文件中注册（导入 pipeline 模块后调用 register_pipeline）。
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 注册表
# ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {}


def register_pipeline(name: str):
    """
    装饰器：注册一个 pipeline 工厂函数。

    用法:
        @register_pipeline("hybrid_rag")
        def _create_hybrid_rag(collection_name=None, **kwargs):
            return Pipeline(
                retriever=HybridRetriever(...),
                augmenter=Augmenter(),
                generator=Generator(),
            )
    """
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def create_pipeline(mode: str, collection_name: str = None) -> "Pipeline":
    """根据 mode 名称创建对应的 Pipeline 实例。"""
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知 mode: {mode!r}，可用: {available}")
    return _REGISTRY[mode](collection_name=collection_name)


def available_modes() -> list[str]:
    """返回所有已注册的 mode 名称。"""
    return sorted(_REGISTRY.keys())


# ──────────────────────────────────────────────────────────
# Pipeline 基类
# ──────────────────────────────────────────────────────────

class Pipeline:
    """
    可扩展的 RAG 流水线。

    默认流程：检索（可选）→ Prompt 构建 → 生成
    子类可重写 process() 实现自定义流程（如 Self-RAG、CRAG）。

    使用方法：
        pipe = Pipeline(retriever=r, augmenter=a, generator=g)
        result = pipe.process("What is X?")
        # result = {"answer": "...", "contexts": ["...", "..."]}
    """

    def __init__(self, retriever=None, augmenter=None, generator=None):
        self.retriever = retriever
        self.augmenter = augmenter
        self.generator = generator

    def process(self, query: str) -> dict:
        """
        处理单条 query。

        Returns:
            {"answer": str, "contexts": list[str]}
        """
        # 1. 检索（可选）
        contexts = []
        if self.retriever:
            hits = self.retriever.retrieve(query, top_k=cfg.TOP_K)
            contexts = [h["text"] for h in hits]

        # 2. 构建 Prompt
        prompt = self.augmenter.build_prompt(query, contexts)

        # 3. 生成
        answer = self.generator.generate(prompt)

        return {"answer": answer, "contexts": contexts}


# ──────────────────────────────────────────────────────────
# 内置 pipeline 注册
# ──────────────────────────────────────────────────────────

@register_pipeline("no_rag")
def _create_no_rag(collection_name=None, **kwargs):
    """LLM 直接作答，无检索上下文。"""
    from rag.augmenter import Augmenter
    from rag.generator import Generator

    return Pipeline(
        retriever=None,
        augmenter=Augmenter(),
        generator=Generator(),
    )


@register_pipeline("naive_rag")
def _create_naive_rag(collection_name=None, **kwargs):
    """标准 RAG：检索 Top-K → 注入 Prompt → 生成。"""
    from rag.augmenter import Augmenter
    from rag.generator import Generator
    from rag.retriever import Retriever

    if not collection_name:
        raise ValueError("naive_rag 需要 collection_name 参数")

    retriever = Retriever(collection_name=collection_name)
    if retriever.count() == 0:
        raise RuntimeError(
            f"ChromaDB 集合 '{collection_name}' 为空！\n"
            f"请先运行: python build_index.py --mode eval"
        )

    return Pipeline(
        retriever=retriever,
        augmenter=Augmenter(),
        generator=Generator(),
    )


# ──────────────────────────────────────────────────────────
# 扩展示例（未来创新点添加模板）
# ──────────────────────────────────────────────────────────

# 每个创新点的代码放在 rag/<方法名>/ 子文件夹中，不要在 rag/ 根目录创建散文件。
# 以下是添加新创新点的模板，取消注释并修改即可使用：
#
# @register_pipeline("crag")
# def _create_crag(collection_name=None, **kwargs):
#     from rag.crag.pipeline import CRAGPipeline   # 新建 rag/crag/ 子文件夹
#     return CRAGPipeline(collection_name=collection_name)
#
# @register_pipeline("self_rag")
# def _create_self_rag(collection_name=None, **kwargs):
#     from rag.self_rag.pipeline import SelfRAGPipeline   # 新建 rag/self_rag/ 子文件夹
#     return SelfRAGPipeline(collection_name=collection_name)


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

    print(f"\n{SEP}")
    print("rag/pipeline.py — 注册表测试")
    print(SEP)
    print(f"  已注册 mode: {available_modes()}")

    # 测试 no_rag 创建
    print("\n[Test 1] 创建 no_rag pipeline …")
    pipe = create_pipeline("no_rag")
    assert pipe.retriever is None
    assert pipe.augmenter is not None
    assert pipe.generator is not None
    print("  [PASS] no_rag pipeline 创建成功")

    # 测试未知 mode
    print("\n[Test 2] 测试未知 mode 报错 …")
    try:
        create_pipeline("nonexistent")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"  [PASS] 正确报错: {e}")

    print(f"\n{SEP}")
    print("rag/pipeline.py 测试通过！")
    print(f"当前可用 mode: {available_modes()}")
    print("添加新 mode: 在本文件底部用 @register_pipeline 注册即可")
    print(SEP)
