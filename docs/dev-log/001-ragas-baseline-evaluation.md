# RAGAS 基线评估：派聪明论文知识库检索与生成质量

- **日期**: 2026-06-02
- **场景**: 科研论文知识库问答（全色锐化领域，5 篇遥感论文）

---

## 背景问题

派聪明从通用 RAG 系统改造为科研论文知识库后，需要建立可量化的评估基准，以衡量当前 RAG 流水线的检索和生成质量，并为后续优化提供对比基线。

## 实验设计

### QA 数据集

- **来源**: 人工构造的 55 条问答对，覆盖 5 篇论文 + 通用概念
- **论文**: WFANet (AAAI25)、PSCSC、SFIT (AAAI22)、DCPNet、DFCFN
- **问题类型**: concept / problem / innovation / architecture / methodology / experiment / comparison
- **每条包含**: question, ground_truth, contexts（检索到的原文段落）

### 评估方法

- **评判 LLM**: DeepSeek Chat（通过 OpenAI 兼容接口调用，temperature=0）
- **指标**: faithfulness（忠实度）、answer_relevancy（回答相关性）、context_precision（上下文精度）、context_recall（上下文召回）

## 实验结果

| 指标 | 均值 | 最小值 | 最大值 | 说明 |
|------|:---:|:-----:|:-----:|------|
| Faithfulness | **0.052** | 0.000 | 1.000 | 答案与上下文的事实一致性 |
| Answer Relevancy | **0.982** | 0.000 | 1.000 | 答案与问题的相关性 |
| Context Precision | **0.110** | 0.000 | 1.000 | 检索片段中相关部分的比例 |
| Context Recall | **0.102** | 0.000 | 1.000 | 上下文覆盖标准答案所需信息的程度 |

### 按问题类型分组

| 类型 | 条数 | Faithfulness | Precision | Recall |
|------|:---:|:----------:|:---------:|:-----:|
| methodology | 20 | 0.092 | 0.258 | 0.195 |
| experiment | 12 | 0.083 | 0.042 | 0.100 |
| architecture | 6 | 0.000 | 0.000 | 0.000 |
| concept | 6 | 0.000 | 0.033 | 0.083 |
| comparison | 5 | 0.000 | 0.040 | 0.000 |
| problem | 3 | 0.000 | 0.000 | 0.000 |
| innovation | 3 | 0.000 | 0.000 | 0.000 |

## 问题示例

### 检索失败的典型例子

**Q（问题类型: innovation）**："SFIT 的核心贡献是什么？"
- **检索命中**: 5 个片段（来自其他论文的 general 内容）
- **系统回答**: "暂无相关信息"
- **分析**: 搜索引擎未将 "SFIT" 名称与论文内容关联，5 个检索结果均来自不相关段落
- **根因**: 512 字符的固定分块可能截断了论文摘要中的关键信息

### 检索命中但 Faithfulness 低的例子

**Q（问题类型: methodology）**："WFANet 包含哪两个子模块？"
- **Ground Truth**: "Multi-Frequency Fusion Attention (MFFA) 和 Spatial Detail Enhancement Module (SDEM)"
- **系统回答**: "Frequency Attention Triplet (FATG) 和 Adaptive Detail Fusion (ADFR)"
- **Faithfulness**: 0.0（答案中的模块名称在上下文中不存在）
- **分析**: LLM 生成了一对假的模块名称，听起来合理但事实错误。根本原因是检索到的上下文中包含 WFANet 的碎片化描述，但缺少模块名称的精确引用

### 检索完全命中的成功案例

**Q（问题类型: experiment）**："DFCFN 相比第二优方法在 SAM 指标上提升了多少？"
- **检索命中**: 包含精确数字的段落
- **系统回答**: "6.4%、6.2% 和 5.3%"
- **Faithfulness**: 1.0, Recall: 1.0
- **分析**: 该问题涉及的具体数字在原文中有明确表述，检索命中了包含数字的段落，LLM 直接提取了准确值

## 问题分析

### 检索环节（Context Precision / Recall 低）

1. **分块策略不匹配论文结构**
   - 当前: 固定 512 字符按段落/句子分割
   - 问题: 论文的 Method section 通常含连贯的多段描述（问题陈述→方法动机→具体实现），被截断后单块信息不完整
   - 实验数据（表格、数值）和结论性语句可能被分散到不同 chunk

2. **缺少论文元数据索引**
   - chunk 不携带论文标题、章节名、作者等元信息
   - 如 "SFIT" 作为论文名称，在 chunk 中可能仅以 "our method" 出现，导致关键词匹配失败

3. **混合搜索权重可能不平衡**
   - KNN 向量检索 + BM25 文本检索的权重由 rescore 决定（queryWeight=0.2, rescoreQueryWeight=1.0）
   - BM25 主导时，对专业术语（如 "仿射耦合层"、"可逆神经网络"）的分词匹配效果差

### 生成环节（Faithfulness 低）

1. **Prompt 中"参考信息优先"的约束不够强**
   - 当检索上下文不包含答案时，LLM 倾向于用自身知识"补全"
   - 论文特定问题（如模块名称、数据集名称）的系统幻觉率高

2. **缺乏"无法回答"的保底机制**
   - 即使部分问题触发 "暂无相关信息"，仍有大量问题生成了错误信息
   - 需要更严格的上下文利用检测

## 初步结论

1. **当前 RAG 流水线在论文知识库场景下检索召回严重不足**，这是最大的瓶颈
2. **检索质量低直接导致生成质量不可控**，faithfulness 的根因在 retrieval 而非 generation
3. **方法论（methodology）类问题相对表现较好**，因为这类描述在论文正文中占比最大
4. **有明确数值/指标的问题容易命中**（如 DFCFN 的 SAM 数据），因为数字关键词的 BM25 匹配效果好

## 改进方向（待讨论）

1. **(高优先级) 分块策略优化**: 基于论文章节结构的分块（Introduction / Method / Experiment / Conclusion），保留段落完整性
2. **(高优先级) 论文元数据丰富**: 每个 chunk 关联论文标题、章节名称，支持按论文/章节过滤检索
3. **(中优先级) 检索增强**: 引入查询重写（Query Rewrite），对论文特定术语做同义词扩展
4. **(中优先级) 重排序**: 在 KNN+BM25 后加 Cross-encoder 重排序
5. **(低优先级) Prompt 优化**: 强化 "必须基于上下文回答，否则如实说不知道" 的约束
