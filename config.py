"""
config.py — 全局配置，唯一超参来源
所有模块通过 `import config as cfg` 读取，禁止硬编码。
"""
import os

# ──────────────────────────────────────────────────────────
# HuggingFace 镜像与缓存（须在 datasets/transformers 导入前生效）
# ──────────────────────────────────────────────────────────
HF_ENDPOINT  = "https://hf-mirror.com"
HF_CACHE_DIR = "/data/zhaoyaoxi/huggingface_cache"

CUDA_VISIBLE_DEVICES = "1"       # 指定使用的 GPU 编号

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
EMBED_BATCH_SIZE = 64          # 单次编码批大小（RTX 3090 显存适配）

# ──────────────────────────────────────────────────────────
# 生成模型（vLLM OpenAI 兼容接口）
# ──────────────────────────────────────────────────────────
LLM_BASE_URL    = "http://localhost:8000/v1"
LLM_MODEL       = "Qwen2.5-7B-Instruct"
LLM_API_KEY     = "EMPTY"     # vLLM 不校验 key
LLM_TEMPERATURE = 0.0         # 确定性输出，保证可复现
LLM_MAX_TOKENS  = 512

# ──────────────────────────────────────────────────────────
# 检索参数
# ──────────────────────────────────────────────────────────
TOP_K = 3                      # 每次检索返回的段落数

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
DATASET_NAME = "BeIR/nq"
QRELS_NAME   = "BeIR/nq-qrels"

DEV_QUERY_NUM  = 200           # 开发/调试阶段
DEV_CORPUS_NUM = 10_000

EVAL_QUERY_NUM  = 500          # 论文正式评测
EVAL_CORPUS_NUM = 50_000

# ──────────────────────────────────────────────────────────
# Ragas / DeepSeek Judge
# ──────────────────────────────────────────────────────────
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL    = "deepseek-chat"
