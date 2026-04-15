"""
config.py — 全局配置，唯一超参来源
所有模块通过 `import config as cfg` 读取，禁止硬编码。
"""
import os

# 加载 .env 文件（本地机密，不进 git）
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ──────────────────────────────────────────────────────────
# HuggingFace 镜像与缓存（须在 datasets/transformers 导入前生效）
# ──────────────────────────────────────────────────────────
HF_ENDPOINT  = "https://hf-mirror.com"
HF_CACHE_DIR = "/data/zhaoyaoxi/huggingface_cache"

CUDA_VISIBLE_DEVICES = "2"       # 指定使用的 GPU 编号（bge-m3 及 Embedding 相关）
EMBED_GPU = "cuda:0"             # torch 内部设备号（CUDA_VISIBLE_DEVICES 映射后始终为 cuda:0）

os.environ["HF_ENDPOINT"]         = HF_ENDPOINT
os.environ["HUGGINGFACE_HUB_URL"] = HF_ENDPOINT
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ──────────────────────────────────────────────────────────
# 路径
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR   = os.path.join(PROJECT_ROOT, "chroma_db")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

# ──────────────────────────────────────────────────────────
# 嵌入模型
# ──────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 128          # 单次编码批大小（RTX 3090 显存适配）

# ──────────────────────────────────────────────────────────
# 生成模型（vLLM OpenAI 兼容接口）
# ──────────────────────────────────────────────────────────
LLM_BASE_URL    = "http://localhost:8000/v1"
LLM_MODEL       = "Qwen2.5-7B-Instruct"
LLM_API_KEY     = "EMPTY"     # vLLM 不校验 key
LLM_TEMPERATURE = 0.0         # 确定性输出，保证可复现
LLM_MAX_TOKENS  = 128          # SQuAD 短答案（平均 3 词），128 tokens 足够

# ──────────────────────────────────────────────────────────
# 检索参数
# ──────────────────────────────────────────────────────────
TOP_K = 5                      # 每次检索返回的段落数

# ──────────────────────────────────────────────────────────
# 并发与断点续传
# ──────────────────────────────────────────────────────────
CONCURRENT_REQUESTS = 8          # vLLM 并发请求数（vLLM 自动 batch）
CHECKPOINT_INTERVAL = 10         # 每处理 N 条保存一次断点

# ──────────────────────────────────────────────────────────
# 切块策略（word-based，英文约 1.3 word/token）
# chunk_size=256 words ≈ 340 tokens；overlap=32 words ≈ 42 tokens
# ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 256            # words per chunk
CHUNK_OVERLAP = 32             # words overlap between adjacent chunks

# ──────────────────────────────────────────────────────────
# 数据集与采样
# ──────────────────────────────────────────────────────────
RANDOM_SEED  = 42
DATASET_NAME = "squad_v2"           # 用于 ChromaDB 集合命名

DEV_QUERY_NUM  = 50                  # 快速调试
EVAL_QUERY_NUM = 600                 # 评测 query 数

# ──────────────────────────────────────────────────────────
# Ragas / LLM Judge（GLM-4-Flash，免费）
# ──────────────────────────────────────────────────────────
GLM_API_KEY        = os.environ.get("GLM_API_KEY", "")
GLM_RAGAS_API_KEY  = os.environ.get("GLM_RAGAS_API_KEY", "")
GLM_BASE_URL       = "https://open.bigmodel.cn/api/paas/v4"
GLM_MODEL          = "glm-4-flashx"
RAGAS_MAX_WORKERS  = 30
