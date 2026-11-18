# AlphaBench

> **AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining**
> Haochen Luo, Ho Tin Ko, Jiandong Chen, David Sun, Yuan Zhang, Chen Liu（香港城市大学）｜ ICLR 2026｜ 代码：github.com/CityU-MLO/AlphaBench｜ 主页：<https://alphabench.cc/>

## 摘要

公式化 alpha 因子挖掘（Formulaic Alpha Factor Mining, FAFM）是量化投资中的核心问题，其目标是设计可解释的公式从历史金融序列中提取预测信号。随着大语言模型（LLM）的出现，近期研究开始探索它们在 FAFM 中的角色，但其在不同任务和配置下的能力仍不清楚。本文介绍 AlphaBench，首个系统评估 LLM 在 FAFM 中表现的基准。AlphaBench 覆盖三个核心任务：因子生成（factor generation）、因子评估（factor evaluation）和因子搜索（factor searching），这些都是量化研究工作流中常见的任务。除任务级评估外，还进一步分析不同 LLM 设置（包括模型类型、提示范式、推理策略）如何影响性能。在一系列开源和闭源模型上的实验揭示，LLM 在自动化因子挖掘方面具有强大潜力，同时在鲁棒性、搜索效率和实用性方面也面临持续挑战。

## Introduction

LLM 在代码生成、数学推理和符号合成方面展现出强大能力，自然地被研究者尝试应用于量化金融的 alpha 因子挖掘。然而，学术界和工业界长期缺乏一个统一、系统的基准来回答关键问题：不同 LLM 在生成可执行 alpha 公式上的表现如何？LLM 能否在没有回测的情况下准确评估因子质量？LLM 在迭代搜索更优因子时的效率和效果如何？AlphaBench 正是为填补这一空白而设计。

### 相关工作：现有基准的不足

| 现有基准 | 覆盖范围 | 与 AlphaBench 的差距 |
| --- | --- | --- |
| FinQA / ConvFinQA / FinTextQA | 金融数值推理、问答 | 仅测试阅读理解，不测试生成可执行 alpha 公式 |
| FinBen | 42 个金融数据集，覆盖信息提取、预测、决策 | 无专门评估 LLM 生成可执行 alpha 因子 DSL 表达式的任务 |
| Pixiu | 金融指令微调数据 + FinMA 模型 + 8 任务基准 | 混合 NLP 和预测任务，但无 alpha 因子生成与评估 |
| EvoCodeBench / Text-to-SQL / VerilogEval | 代码/DSL 生成 | 面向通用编程或硬件描述，非金融特有的 alpha 表达式语法和语义 |
| FinEval / FLUE | 金融语言理解 | 不涉及可执行公式生成和回测验证 |

现有基准的普遍局限：要么测试通用代码生成而不涉及金融语义，要么测试金融 NLP 而不涉及可执行表达式，无法系统衡量 LLM 在"将金融概念翻译为可回测公式"这一核心能力上的表现。

核心假设：FAFM 可以被自然地建模为代码生成问题，其中 LLM 必须生成在特定 DSL 下语法正确且语义符合金融直觉的可执行表达式。因此，评估 LLM 在 FAFM 中的表现需要同时考察三个维度：（1）生成能力（能否正确翻译描述为公式）；（2）评估能力（能否在零样本情况下判断因子质量）；（3）搜索能力（能否通过迭代策略高效发现更优因子）。

## 基准设计

### 三个任务的设计逻辑与数据流

AlphaBench 的整体设计遵循 alpha 因子研究的真实工作流：研究员首先根据金融直觉**生成**候选因子，然后**评估**这些因子的质量，最后通过系统**搜索**迭代改进。

- **因子生成**：输入自然语言描述或方向性主题 → LLM 根据指令库（允许的变量和算子）生成 JSON 格式的因子表达式 → 通过 Qlib 回测引擎检查语法有效性和可执行性；
- **因子评估**：输入候选因子池或单个因子表达式 → LLM 在零样本情况下对因子进行排名或打分 → 与回测得到的真实 IC/RankIC/Sharpe 对比；
- **因子搜索**：输入种子因子 + 搜索算法模板 → LLM 按模板进行多轮迭代（CoE/ToT/EA）→ 比较最终因子与初始种子的性能提升。

### 子任务定义

**因子生成**：

1. **Text2Alpha Generation**：模型接收宽泛描述（如"动量"或"均值回归"），翻译为完整公式表达式。例如 "Return the rolling max of close minus current close, divided by volume SMA (default window = 25)"，期望输出包含 Div、Sub、Max、Mean 等算子的 JSON 表达式；
2. **Directional Mining**：模型被提供特定主题（如"基于波动率"或"成交量驱动"），生成该主题下的一组不同因子。指令分为三级难度。

**因子评估**：

1. **Ranking**：给定 $N$ 个候选因子，模型选择表现最好的 $k$ 个；评估指标为 precision@k 和与回测真值的秩相关；
2. **Scoring**：模型为单个因子分配绝对分数（估计 IC、Sharpe 或定性评级）；评估指标为信号准确率（Signal ACC）和 MAE；
3. **Signal Classification（原子任务）**：给定因子表达式，判断该因子是有意义信号还是噪声；
4. **Pairwise Selection（原子任务）**：面对两个候选因子，比较并选择预期表现更好的一个。

**因子搜索**：

1. **Chain-of-Experience (CoE)**：从种子因子出发进行序列精炼，每轮基于历史尝试的最佳结果生成下一个改进版本；
2. **Tree-of-Thought (ToT)**：带剪枝的分支探索，在每一深度层级上生成多个候选，评估后保留 top-N 继续扩展；
3. **Evolutionary Algorithm (EA)**：基于种群的迭代，通过 LLM 驱动的突变和交叉操作生成新一代因子。

### 评估指标设计

- **生成任务**：Reliability（产生有效代码的比例）、Stability（多次采样输出一致的程度）、Accuracy（生成表达式匹配用户意图的程度）；
- **评估任务**：Ranking Precision、Scoring Signal ACC、Scoring MAE、Overall；
- **搜索任务**：Search Cost（成功找到有效因子所需的 token 数和轮数）、Search Quality（最终因子与初始种子的改进）、Diversity、ImprovedRate / ImprovedRatio。

## 实现与结果

### 数据集与模型

**指令规模**：因子生成 687 条指令（含 Text2Alpha 和 Directional Mining）；因子评估 1170 条指令（含 Ranking 和 Scoring）；因子搜索 27 条搜索指令，覆盖 3 种算法。

**数据与市场**：回测框架 Microsoft Qlib；因子库 Alpha158 作为种子池；市场数据 CSI300 成分股，2020–2025 年日数据。

**模型列表**：商业模型（Gemini-2.5-Pro、Gemini-2.5-Flash、Gemini-1.5-Flash-8B、GPT-4.1-Mini、GPT-5）；开源模型（DeepSeek-R1-Distill-Qwen-32B、DeepSeek-V3、LLaMA3.1-70B-Instruct、LLaMA3.1-8B-Instruct、Qwen-2.5-14B-Instruct）；代码专用模型（Qwen2.5-Coder-7B、CodeLlama-70B、Qwen3-Coder-480B）。

**Token 消耗**（以 DeepSeek-V3 为代表）：全部任务总消耗约 5.5M tokens（其中 prompt 4.2M）；支持 prompt caching 的模型 cache hit ratio 达 85%。

### 因子生成结果

| 模型 | 设置 | Reliability | Stability | Accuracy | Overall |
| --- | --- | --- | --- | --- | --- |
| GPT-5 | Vanilla | 1.00 | 0.62 | 0.56 | **0.72** |
| GPT-5 | CoT | 0.99 | 0.58 | 0.44 | 0.69 |
| Gemini-2.5-Flash | Vanilla | 0.99 | 0.57 | 0.57 | **0.71** |
| Gemini-2.5-Flash | CoT | 0.99 | 0.49 | 0.58 | 0.69 |
| DeepSeek-V3 | Vanilla | 0.91 | 0.35 | 0.31 | 0.52 |
| LLaMA3.1-70B | Vanilla | 0.95 | 0.52 | 0.38 | 0.62 |
| DeepSeek-R1-Distill-Qwen-32B | Vanilla | 0.35 | 0.19 | 0.14 | **0.23** |
| Qwen3-Coder-480B | Vanilla | 0.96 | 0.56 | 0.43 | 0.65 |

- GPT-5 Overall 最高（0.72），Gemini-2.5-Flash 紧随其后（0.71）——写公式这个活不需要最贵的模型；
- CoT 对生成任务仅有边际收益，有时降低 Stability；
- DeepSeek-R1-Distill-Qwen-32B 表现最差（0.23），说明蒸馏模型在此专门任务上能力严重受限；
- 代码专用 LLM 中 Qwen3-Coder-480B 领先（0.65），但仍低于通用大模型，显示 coder-oriented 模型在金融语义理解上的差距。

### 因子评估结果

| 模型 | Ranking Precision | Scoring Signal ACC | Scoring MAE | Overall |
| --- | --- | --- | --- | --- |
| GPT-5 | 0.24 / 0.32 / 0.25 | 0.32 / 0.36 | 1.67 | 0.47 / 0.48 |
| Gemini-2.5-Pro | 0.23 / 0.24 | 0.36 / 0.32 | 1.66 | 0.48 / 0.47 |
| Gemini-2.5-Flash | 0.25 / 0.14 | 0.32 / 0.33 | 1.67 / 1.64 | 0.47 / 0.44 |
| DeepSeek-V3 | 0.19 / 0.17 | 0.16 / 0.17 | 1.60 / 1.57 | 0.40 / 0.40 |

所有模型评估能力均非常弱，最佳 Overall 约 0.48，远低于生成任务。原子任务：Signal Classification 准确率约 0.39–0.46（接近随机），Pairwise Selection 约 0.44–0.57（略好于随机）。评估是所有 LLM 的短板，原因有二：

1. **因子挖掘太小众**：大模型训练时几乎没见过"因子公式 → 回测结果"这种配对数据，无法建立公式结构和预期效果之间的映射；
2. **光看公式太缺上下文**：因子好坏与回测时间窗口、调仓频率、市场环境都有关系，但光看一行公式这些信息全丢了。

**SFT 效果**：CSI300 Pairwise Selection 从 0.48 提升至 0.83（域内微调），但 Signal Classification 仅 0.39 → 0.48；跨市场迁移上，CSI300 微调后的模型在 S&P500 Pairwise Selection 上仍有提升（0.44 → 0.64）。

### 因子搜索结果

| 模型 | Search Quality | Search Cost |
| --- | --- | --- |
| DeepSeek-V3 | 0.494 | 0.800 |
| Gemini-1.5-Flash-8b | 0.622 | 0.802 |
| Gemini-2.5-Flash | 0.646 | 0.850 |
| Gemini-2.5-Pro | 0.632 | 0.808 |
| GPT-4.1-Mini | 0.608 | 0.904 |
| GPT-5 | **0.656** | 0.940 |
| LLaMA3.1-70b | 0.624 | 0.850 |

- GPT-5 质量最高但成本也最高；**Gemini-2.5-Flash 性价比最优**（质量 0.646 仅次于 GPT-5，成本更低）——论文原话说轻量商业模型是大规模搜索的实用执行器；
- EA（进化算法）在 CSI300 和 S&P500 上均优于 CoE 和 ToT——进化算法天然适合"生成、评估、改进"的循环；
- 温度 1.5 提高多样性但增加成本；温度 0.75 更高效可靠但多样性略降；EA 种群大小 20 在成本、多样性和改进间达到最佳平衡。

### 模型间差异总结

1. **规模效应一致**：更大模型在各任务上均优于较小模型，但评估任务上差距缩小；
2. **商业模型 > 开源模型**：在生成和搜索任务上 GPT-5 和 Gemini 系列显著领先；
3. **CoT 双刃剑**：对生成有边际帮助（小模型更明显），对评估几乎无益甚至有害；
4. **代码专用模型的局限**：Qwen3-Coder-480B Reliability 高（0.96）但 Accuracy（0.43）明显低于同规模通用模型；
5. **评估是所有 LLM 的短板**：无论规模大小，所有模型在零样本因子评估上表现接近随机。

### 局限性

1. **模型覆盖有限**：仅测试了 10 余个模型；
2. **市场范围局限**：仅使用中国 A 股 CSI300 日数据，未覆盖美股、加密货币、日内数据；
3. **提示策略有限**：仅对比 Vanilla 和 CoT，未测试 ReAct、RAG 等范式；
4. **因子类型单一**：仅测试公式化 alpha，未覆盖机器学习因子、另类数据因子；
5. **SFT 过拟合风险**：模型倾向于坍缩到预测单一噪声标签；
6. **回测真实性**：使用 Qlib 标准设置，未考虑滑点、市场冲击、流动性限制等真实交易摩擦。

### 使用指南与扩展潜力

使用流程：选择待测 LLM（支持 OpenAI/Gemini/本地模型 API）→ 选择任务指令集 → 调用 LLM 生成 alpha 表达式（Qlib 兼容格式）→ Qlib 回测执行获取 IC/RankIC/Sharpe → 对比预测与真值计算指标 → 搜索任务按 CoE/ToT/EA 模板迭代。成本参考：以 DeepSeek-V3 完整跑一遍全部任务约需 5.5M tokens，prompt caching 可降低约 85% 的重复成本。

FFO（Factor Feature Oracle）评估引擎：输入一个因子公式自动跑回测输出效果指标，如 `ppo eval "Rank($close, 20)"`、`ppo eval "Mean($volume, 5) / Std($volume, 20)" --market csi500`，输出 IC、RankIC、ICIR、胜率、年化收益。FFO 基于 Qlib 构建，支持沪深 300、中证 500、中证 1000 和标普 500 四个市场（2020–2025），并支持 MCP 协议被 AI Agent 调用。

扩展潜力：跨资产扩展（替换数据即可适配其他市场）、跨 DSL 扩展（替换因子语言可应用于符号回归、约束求解等）、集成 RAG、连接组合优化、更复杂搜索算法（MCTS、贝叶斯优化）。

## 总结

AlphaBench 是第一个系统测试大语言模型在量化金融公式化 alpha 因子挖掘中生成、评估和搜索三方面能力的标准化基准。

1. **生成**：给 LLM 一段文字描述（如"动量因子"），看它能否写出正确的数学公式；
2. **评估**：给 LLM 几个公式，不跑回测，看它能不能猜出哪个更好；
3. **搜索**：给 LLM 一个种子公式，让它通过多轮修改逐步找到更优版本。

两大贡献：**给了标准**——以后谁再说"我的大模型挖因子很牛"，先在 AlphaBench 上跑一跑；**给了工具**——FFO 让因子评估从写半天代码变成一行命令。
