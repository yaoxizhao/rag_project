# Innovation Ideas - 创新点孵化器

## 2026-04-15 Corrective Retrieval Augmented Generation (CRAG)

**创新点**：
1. 轻量级检索评估器（T5-large 0.77B 微调），对 query-document 对打分判断相关性，准确率 84.3%
2. 三动作触发机制 {Correct, Incorrect, Ambiguous}：基于置信度阈值触发不同纠正策略
3. 知识精炼（Decompose-then-Recompose）：将文档分解为知识条，过滤无关信息后重组

**方法关键词**：retrieval evaluation, corrective action, knowledge refinement, decompose-then-recompose, web search fallback, plug-and-play

**源码**：论文称"will publish later"，2024年论文，社区有非官方实现

**可行的融合方案**：
- CRAG-Lite：只取检索评估器+拒绝机制，不做网络搜索。零训练路径用 bge-m3 相似度，轻量训练路径用 bge-reranker。直接解决 hallucination_rate=0.936 的瓶颈。预计半天实现。**→ 已实现（rag/crag/），mode: crag_lite，待评测**
- CRAG-Full：完整复现 CRAG（评估器+知识精炼+网络搜索）。在 CRAG-Lite 基础上加搜索 API。预计 2-3 天实现。

**可与以下已实现创新点缝合**：
- 暂无其他已实现创新点（当前仅有 no_rag, naive_rag baseline）

**已分析的论文列表**：
- CRAG (Yan et al. 2024)：retrieval evaluation, corrective action, knowledge refinement — 源码：待确认
