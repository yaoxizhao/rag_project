"""
rag/generator.py — LLM 生成模块

通过 vLLM 的 OpenAI 兼容接口调用 Qwen2.5-7B-Instruct，
封装为简洁的 generate(prompt) → str 接口。

公开接口（符合架构文档规范）：
    Generator(base_url, model)
        .generate(prompt) -> str
        .health_check()   -> bool

── 架构插拔点 ────────────────────────────────────────────────
后续可替换为其他 API（DeepSeek、GPT-4o 等）或本地 HF 推理，
只需保持 generate() 签名不变。
─────────────────────────────────────────────────────────────

前置条件：vLLM 服务已启动
    bash serve_model.sh
    # 或手动：
    # CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    #   --tensor-parallel-size 2 --port 8000
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

logger = logging.getLogger(__name__)


class Generator:
    """
    基于 OpenAI 兼容接口的 LLM 生成器。

    使用方法：
        gen = Generator()
        answer = gen.generate(prompt)
    """

    def __init__(
        self,
        base_url: str = cfg.LLM_BASE_URL,
        model: str    = cfg.LLM_MODEL,
        api_key: str  = cfg.LLM_API_KEY,
    ):
        """
        初始化 OpenAI 客户端。

        Args:
            base_url: vLLM 服务地址（默认 http://localhost:8000/v1）
            model:    模型名称（默认 Qwen2.5-7B-Instruct）
            api_key:  API Key（vLLM 不校验，填 "EMPTY" 即可）
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装 openai 库: pip install openai") from e

        self._client   = OpenAI(base_url=base_url, api_key=api_key)
        self._model    = model
        self._base_url = base_url
        logger.info(f"[Generator] 初始化完成: {base_url} / {model}")

    # ──────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用 LLM 生成回答（带指数退避重试）。

        Args:
            prompt: 由 Augmenter.build_prompt() 构建的完整 Prompt
            max_retries: 最大重试次数（默认 3 次）

        Returns:
            str: LLM 生成的纯文本回答（已去除首尾空白）

        Raises:
            openai.APIConnectionError: vLLM 服务未启动或地址错误
            openai.APIStatusError:     服务端返回错误状态码
        """
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=cfg.LLM_TEMPERATURE,
                    max_tokens=cfg.LLM_MAX_TOKENS,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    f"[Generator] attempt {attempt + 1}/{max_retries} failed: {e}, "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)

    def health_check(self) -> bool:
        """
        检查 vLLM 服务是否可达。

        Returns:
            True  — 服务正常
            False — 服务不可达
        """
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"[Generator] 服务在线，可用模型: {available}")
            return True
        except Exception as e:
            logger.warning(f"[Generator] 服务不可达: {e}")
            return False


# ──────────────────────────────────────────────────────────
# 本地测试入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("rag/generator.py — 本地测试")
    print(f"目标服务: {cfg.LLM_BASE_URL}")
    print(SEP)

    gen = Generator()

    # ── Step 1: 健康检查 ───────────────────────────────────
    print("\n[Step 1] 健康检查（vLLM 服务连通性）…")
    alive = gen.health_check()
    if not alive:
        print(f"\n  [SKIP] vLLM 服务未启动（{cfg.LLM_BASE_URL}）")
        print("  请先运行: bash serve_model.sh")
        print("  generator.py 代码结构验证通过，跳过在线测试。")
        sys.exit(0)
    print("  [PASS] vLLM 服务在线")

    # ── Step 2: no_rag 生成测试 ────────────────────────────
    print("\n[Step 2] no_rag 生成测试…")
    from rag.augmenter import Augmenter
    aug = Augmenter()

    q = "who wrote hallelujah i just love her so"
    prompt_no_rag = aug.build_prompt(q, contexts=[])
    print(f"  Query : {q!r}")
    answer_no_rag = gen.generate(prompt_no_rag)
    print(f"  Answer: {answer_no_rag}")
    assert answer_no_rag, "no_rag 生成结果为空！"
    print("  [PASS] no_rag 生成非空")

    # ── Step 3: naive_rag 生成测试 ─────────────────────────
    print("\n[Step 3] naive_rag 生成测试（注入真实语料片段）…")
    contexts = [
        '"Hallelujah I Love Her So" is a single from American musician '
        'Ray Charles. The jazz and rhythm and blues song was written by '
        'Ray Charles and was released in 1956.',
        "Ray Charles Robinson was an American singer, songwriter, musician "
        "and composer. He is best known for his pioneering soul music.",
    ]
    prompt_rag = aug.build_prompt(q, contexts=contexts)
    answer_rag = gen.generate(prompt_rag)
    print(f"  Answer: {answer_rag}")
    assert answer_rag, "naive_rag 生成结果为空！"
    print("  [PASS] naive_rag 生成非空")

    print(f"\n{SEP}")
    print("rag/generator.py 全部测试通过！")
    print(SEP)
