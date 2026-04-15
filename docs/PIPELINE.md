# Pipeline Execution Order - 实验执行流程
> 最后更新：2026-04-13

## 核心执行链路（随创新点动态迭代）

### [按需执行] Step 1: 构建/更新向量索引
- 脚本：`build_index.py`
- 说明：加载 BeIR/fiqa corpus，切块后用 BAAI/bge-m3 编码，存入 ChromaDB
- 触发条件：**仅在**首次运行、更换数据集、或修改 Chunk/Embedding 策略时才需要执行。平常跑实验**无需重复执行**。
- 产出：`chroma_db/` 目录

### [每次必执行] Step 1.5: 启动 vLLM 推理服务
- 脚本：`serve_model.sh`
- 说明：启动 Qwen2.5-7B-Instruct 的 vLLM 服务（tensor-parallel=2, port=8000）
- 触发条件：**每次实验前**必须确保服务已启动
- 检查：`curl http://localhost:8000/v1/models`

### [高频执行] Step 2: 模型推理与生成
- 脚本：`run_baseline.py`
- 说明：运行当前的 RAG 策略（`--mode no_rag` 或 `--mode naive_rag`等）并生成预测答案 CSV
- 前置依赖：Step 1 的索引库已存在，且 vLLM 服务已在后台启动
- 产出：`results/run_YYYYMMDD/{mode}/` 下的 CSV 文件

### [高频执行] Step 3: Ragas 评估
- 脚本：`evaluate.py`
- 说明：对 Step 2 生成的最新 CSV 计算 faithfulness, answer_correctness, context_recall, context_precision 等指标
- 前置依赖：Step 2 生成的 CSV 文件
- 产出：同目录下生成 `*_ragas.json` 评估结果

## 常见问题与依赖关系
- vLLM 服务需要 2x RTX 3090，确保 `serve_model.sh` 已执行
- ChromaDB 索引只需构建一次，除非更改 embedding 模型或切块策略
- evaluate.py 依赖 大模型 API 作为 Ragas judge，确保 API key 已配置
