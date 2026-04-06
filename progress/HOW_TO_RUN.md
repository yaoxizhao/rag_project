# HOW TO RUN — RAG Baseline 操作手册

> 本文档说明如何通过命令行手动执行各步骤，以及如何让 Claude Code 代劳。

---

## 目录

1. [环境准备](#1-环境准备)
2. [Step 1：启动 vLLM 服务](#2-step-1启动-vllm-服务)
3. [Step 2：构建向量索引](#3-step-2构建向量索引)
4. [Step 3：跑 Baseline 实验](#4-step-3跑-baseline-实验)
5. [Step 4：Ragas 评估](#5-step-4ragas-评估)
6. [完整实验流程一览](#6-完整实验流程一览)
7. [让 Claude Code 执行](#7-让-claude-code-执行)

---

## 1. 环境准备

```bash
# 激活 conda 环境
conda activate rag_project

# 设置 DeepSeek API Key（每次新开终端都要执行）
export DEEPSEEK_API_KEY='sk-xxxxxxxxxxxxxxxx'

# 进入项目根目录
cd /data/zhaoyaoxi/rag_project
```

---

## 2. Step 1：启动 vLLM 服务

vLLM 服务需要**独占一个终端**持续运行，其他步骤在另一个终端执行。

```bash
# 启动服务（会一直阻塞，Ctrl+C 停止）
bash serve_model.sh
```

**验证服务已就绪：**

```bash
curl http://localhost:8000/v1/models
# 预期返回 JSON，包含 "Qwen2.5-7B-Instruct"
```

**配置说明：**

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | `Qwen/Qwen2.5-7B-Instruct` | BF16 精度 |
| 端口 | `8000` | OpenAI 兼容接口 |
| GPU | `1, 2`（CUDA_VISIBLE_DEVICES=1,2） | 避开其他用户占用的卡 |
| TP | `2` | 张量并行，两卡各承担一半 |

> 如需修改 GPU 编号，编辑 `serve_model.sh` 中的 `GPUS="1,2"`。

---

## 3. Step 2：构建向量索引

索引只需构建一次，后续实验复用。

### 开发模式（Dev）— 快速调试

```bash
# 10,000 篇文档，约 10k chunks，构建约 5-10 分钟
python build_index.py --mode dev
```

### 正式评测模式（Eval）— 论文实验

```bash
# 50,000 篇文档，约 50k chunks，构建约 30-60 分钟
python build_index.py --mode eval
```

### 强制重建（清空后重建）

```bash
python build_index.py --mode dev --force
python build_index.py --mode eval --force
```

**输出位置：** `chroma_db/`（持久化在本地磁盘）

**两个集合名：**

| 模式 | 集合名 | 文档数 |
|------|--------|--------|
| dev  | `nq_dev`  | ~10k docs / ~10k chunks |
| eval | `nq_eval` | ~50k docs / ~50k chunks |

> 若集合已存在，脚本会跳过重建（无需 `--force`）。

---

## 4. Step 3：跑 Baseline 实验

**前置条件：** vLLM 服务运行中，对应索引已构建（`naive_rag` 模式需要）。

### 开发集（200 queries × 10k corpus）

```bash
# 纯 LLM，无检索（幻觉上界基准）
python run_baseline.py --mode no_rag

# RAG 模式：先检索 Top-3 段落再生成
python run_baseline.py --mode naive_rag
```

### 正式评测集（500 queries × 50k corpus）

```bash
python run_baseline.py --mode no_rag    --dataset eval
python run_baseline.py --mode naive_rag --dataset eval
```

### 快速冒烟测试（只跑 N 条 query）

```bash
# 验证流程通路，约 1-2 分钟
python run_baseline.py --mode naive_rag --num-queries 5
```

**输出文件：** `results/{mode}_{dataset}_{timestamp}.csv`

| 列名 | 说明 |
|------|------|
| `query_id` | 问题 ID |
| `question` | 问题文本 |
| `answer` | LLM 生成的回答 |
| `contexts` | JSON 字符串，检索到的段落列表（no_rag 为 `[]`） |
| `ground_truth` | 标准答案 |

---

## 5. Step 4：Ragas 评估

**前置条件：** 已设置 `DEEPSEEK_API_KEY`，有 `results/*.csv` 文件。

### 评估指定 CSV 文件（全量）

```bash
python evaluate.py --input results/naive_rag_dev_20260404_152931.csv
```

### 评估并采样（快速测试）

```bash
# 只评估 50 条，约 5 分钟
python evaluate.py --input results/naive_rag_dev_20260404_152931.csv --sample 50
```

### 批量评估多个文件

```bash
for f in results/no_rag_eval_*.csv results/naive_rag_eval_*.csv; do
    python evaluate.py --input "$f"
done
```

**输出文件：** `results/{stem}_ragas.json`

**终端打印示例：**

```
==================================================
  Ragas 评估结果
  文件: naive_rag_dev_20260404_152931.csv
  样本数: 200
==================================================
  Faithfulness       (幻觉↓)       0.2976  [█████░░░░░░░░░░░░░░░]
  Answer Relevancy   (相关性)      0.0530  [█░░░░░░░░░░░░░░░░░░░]
  Context Recall     (覆盖率)      0.0174  [░░░░░░░░░░░░░░░░░░░░]
  Context Precision  (精确率)      0.0671  [█░░░░░░░░░░░░░░░░░░░]
==================================================
```

---

## 6. 完整实验流程一览

### Dev 模式（快速验证，约 20 分钟）

```bash
# 终端 A：启动服务
bash serve_model.sh

# 终端 B：依次执行
export DEEPSEEK_API_KEY='sk-xxxxxxxxxxxxxxxx'
cd /data/zhaoyaoxi/rag_project

python build_index.py --mode dev
python run_baseline.py --mode no_rag
python run_baseline.py --mode naive_rag
python evaluate.py --input results/no_rag_dev_*.csv
python evaluate.py --input results/naive_rag_dev_*.csv
```

### Eval 模式（论文正式实验，约 2-3 小时）

```bash
# 终端 A：启动服务
bash serve_model.sh

# 终端 B：依次执行
export DEEPSEEK_API_KEY='sk-xxxxxxxxxxxxxxxx'
cd /data/zhaoyaoxi/rag_project

python build_index.py --mode eval              # ~30-60 分钟
python run_baseline.py --mode no_rag    --dataset eval   # ~500 × 0.5s ≈ 250s
python run_baseline.py --mode naive_rag --dataset eval   # ~500 × 0.5s ≈ 250s
python evaluate.py --input results/no_rag_eval_*.csv
python evaluate.py --input results/naive_rag_eval_*.csv
```

---

## 7. 让 Claude Code 执行

Claude Code 可以读取代码、修改代码、并通过 Bash 工具在终端执行命令。以下是常见的指令示例。

### 7.1 委托执行单个步骤

直接用自然语言告诉 Claude Code 要做什么，它会自动选择正确命令并执行：

```
帮我构建 eval 模式的向量索引
```

```
运行 naive_rag 正式评测（eval 集，500 queries）
```

```
评估最新的 naive_rag_eval 结果文件
```

### 7.2 委托执行完整流程

```
按顺序执行以下步骤：
1. 构建 eval 索引（50k corpus）
2. 跑 no_rag eval 实验
3. 跑 naive_rag eval 实验
4. 分别评估两个结果文件
```

### 7.3 查看日志或结果

```
查看最新的 naive_rag 结果 CSV，统计一下回答长度分布
```

```
读取最新的 Ragas JSON 评估文件，告诉我各项指标
```

### 7.4 调试问题

```
运行冒烟测试（5 条 query），看看流程有没有报错
```

```
vLLM 健康检查，看服务是否正常
```

### 7.5 修改超参数重新实验

```
把 TOP_K 改成 5，重新跑 dev 集的 naive_rag 实验
```

```
把 CHUNK_SIZE 改成 128，重建 dev 索引，再跑一次 naive_rag
```

### 7.6 注意事项

| 事项 | 说明 |
|------|------|
| **vLLM 需手动启动** | Claude Code 无法在后台持久化进程，需要你在另一个终端先 `bash serve_model.sh` |
| **API Key** | 在终端执行 `export DEEPSEEK_API_KEY='sk-...'` 后，Claude Code 的 Bash 工具会继承该环境变量 |
| **长耗时任务** | `build_index.py --mode eval` 约 30-60 分钟，Claude Code 执行时窗口不能关闭 |
| **告诉 Claude 当前状态** | 例如"vLLM 已经启动了""dev 索引已经有了，直接跑 eval"，避免重复操作 |
