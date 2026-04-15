# 运行指南

> 3 步完成一次实验：改配置 → 跑推理 → 跑评测

## 0. 前置条件

```bash
conda activate rag_project
```

### 启动 vLLM（后台）
```bash
# 新建 tmux session 启动 vLLM（不占用当前终端）
tmux new-session -d -s vllm_server "bash serve_model.sh"

# 查看输出日志
tmux attach -t vllm_server    # 按 Ctrl+B 再按 D 退出（不会中断程序）

# 检查是否启动成功
curl http://localhost:8000/v1/models
```

### tmux 常用操作

```bash
tmux ls                        # 列出所有 session
tmux attach -t <session名>     # 进入 session
# Ctrl+B, D                    # 退出 session（程序继续运行）
tmux kill-session -t <session名>  # 终止 session
```

### 长时间任务用 tmux 跑

```bash
# 推理和评测也可以放 tmux 里，断开 SSH 也不会中断
tmux new-session -d -s rag_run "python run_baseline.py --mode no_rag --dataset eval && python run_baseline.py --mode naive_rag --dataset eval"
tmux attach -t rag_run        # 查看进度
```

## 1. 选择数据集和方法

编辑 `config.py`：

```python
DATASET_NAME = "squad_v2"      # 数据集（见下方可选列表）
TOP_K        = 5               # 检索段落数
CHUNK_SIZE   = 256             # 切块大小
```

可选数据集：`squad_v2`（新增数据集见 `EXTENSION_GUIDE.md`）

## 2. 构建索引（首次或换数据集时）

```bash
python build_index.py --mode eval           # 首次构建
python build_index.py --mode eval --force   # 强制重建
```

## 3. 运行推理

```bash
# 冒烟测试
python run_baseline.py --mode naive_rag --num-queries 5

# 正式实验
python run_baseline.py --mode no_rag    --dataset eval
python run_baseline.py --mode naive_rag --dataset eval
```

可选 mode：`no_rag`、`naive_rag`（新增 mode 见 `EXTENSION_GUIDE.md`）

## 4. 评测

```bash
python evaluate.py --input results/run_YYYYMMDD/naive_rag/naive_rag_eval_*.csv
python evaluate.py --input results/run_YYYYMMDD/no_rag/no_rag_eval_*.csv
```

## 5. 结果

| 输出 | 位置 |
|------|------|
| LLM 回答 CSV | `results/run_{date}/{mode}/{mode}_{dataset}_{ts}.csv` |
| Ragas 评测 JSON | 同目录下 `*_ragas.json` |

## 常用参数速查

| 参数 | 含义 | 示例 |
|------|------|------|
| `--mode` | RAG 策略 | `no_rag`, `naive_rag` |
| `--dataset` | 数据规模 | `dev`(50q), `eval`(全量) |
| `--num-queries` | 覆盖 query 数 | `5` 冒烟测试 |
| `--force` | 强制重建索引 | `build_index.py` 专用 |
| `--sample` | 评测采样数 | `evaluate.py` 专用 |
