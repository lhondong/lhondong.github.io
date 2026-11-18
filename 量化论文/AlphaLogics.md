# AlphaLogics

> **AlphaLogics: A Market Logic-Driven Multi-Agent System for Scalable and Interpretable Alpha Factor Generation**
> Zhangyuhua Weng, Shengli Zhang, Taotao Wang, Yihan Xia（深圳大学）｜ arXiv 2603.20247，2026-03-10

## 摘要

"因子投资最终根植于市场逻辑（Market Logic）——即隐藏在观测到的 alpha 因子背后的潜在机制，它解释了为什么这些因子能够在不同资产和不同市场制度下持续有效。"然而，近期的因子挖掘工作优先追求因子发现而非逻辑发现，产生了复杂但原理不清的 alpha 因子，而市场逻辑本身仍然主要依赖手工构建，难以规模化。本文提出 AlphaLogics，一个由市场逻辑驱动的多智能体因子挖掘系统，包含三个核心组件：**市场逻辑挖掘（Market Logic Mining）**，从历史因子库中反向提取市场逻辑，构建初始市场逻辑库；**引导式因子生成（Guided Factor Generation）**，利用给定的市场逻辑指导新因子的生成与优化，并结合回测反馈进行迭代；**市场逻辑生成（Market Logic Generation）**，以初始市场逻辑库为条件生成新的市场逻辑，并通过聚合其引导因子的回测结果来精炼每一条市场逻辑，持续更新逻辑库。在 CSI 500 和 S&P 500 上的实验表明，AlphaLogics 在代表性基线方法上持续提升预测指标和风险调整后收益，同时生成的市场逻辑库在指导后续因子发现方面保持经验上的有效性。

## Introduction

Alpha mining 的现有方法存在一个共同盲区：只优化因子、不优化市场逻辑。可解释的经济机制仍主要靠手工构建，难以扩展。作者认为，市场逻辑不应是模糊的文本描述，而应被结构化为"条件—预期"对，成为显式可优化的研究对象。

### 相关工作

本文方法与现有方法最根本的不同在于：**将"市场逻辑"从隐含的先验假设提升为显式、可挖掘、可生成、可迭代优化的中间表示**。传统手工方法直接设计因子，数据驱动方法直接优化因子，现有 LLM 方法用语言描述辅助因子优化——但它们都没有将"市场逻辑"本身作为独立的研究对象进行系统化管理和进化。

创新点：

1. **构建市场逻辑库**：从历史因子库中提取底层市场逻辑，将隐式假设转化为显式、可重用的知识单元；
2. **市场逻辑的自动化生成与优化**：通过外层循环持续生成和精炼市场逻辑，实现自动化扩展；
3. **广泛的多市场验证**：跨多个市场的实证实验表明，AlphaLogics 框架下生成的因子取得了更优的表现。

### 与代表性方法的对比

| 方法 | 核心思路 | 优点 | 缺点 | 改进点 |
| --- | --- | --- | --- | --- |
| **AlphaLogics（本文）** | 双循环架构：外层生成/优化市场逻辑，内层在逻辑约束下生成因子 | 可解释性强；逻辑库可累积扩展；跨市场表现稳定；IC/IR 和收益指标全面领先 | 依赖 LLM 能力；计算开销较高（多轮 Agent 交互+回测）；目前仅覆盖技术因子 | 将市场逻辑显式化并自动化进化 |
| AlphaAgent | LLM 驱动的因子生成与优化，通过自然语言语义约束指导因子构建 | 利用 LLM 注入领域知识；自然语言接口降低使用门槛 | 市场逻辑隐式且手工输入；未将逻辑本身作为优化对象 | AlphaLogics 将逻辑从"输入条件"升级为"优化目标" |
| LightGBM / XGBoost | 传统梯度提升树，直接从原始特征学习预测模型 | 训练速度快；工程成熟 | 因子可解释性差；缺乏经济理论支撑；容易过拟合；在本文实验中美股表现甚至为负 | AlphaLogics 通过市场逻辑约束将经济直觉注入生成过程 |

## 方法设计

AlphaLogics 的整体框架是一个**双循环结构**：外层循环优化市场逻辑（Market Logic），内层循环在固定市场逻辑下优化因子表达式（Factor Expression）。

### 阶段一：市场逻辑挖掘（Market Logic Mining）——初始化

输入为历史公开因子库（Alpha101、Alpha191、Alpha158、Alpha360）中的因子表达式。通过三个智能体串联工作，从数学表达式反向抽象出人类可读的市场逻辑：

1. **FormulaStructureAgent**：分析因子表达式的运算结构和形式化属性（如用了哪些算子、时间窗口多长、是否含截面操作等）；
2. **FinancialSemanticsMappingAgent**：将数学操作映射为规范的金融学和行为金融学语义（例如将 `TS_CORR(RANK(open), RANK(volume))` 映射为"开盘价排名与成交量排名的时序相关性"）；
3. **MarketLogicAbstractionAgent**：将金融语义进一步抽象为显式的市场逻辑陈述（例如"缺乏成交量确认的价格变动更可能由短暂情绪驱动并均值回归"）。

输出为初始市场逻辑库 $H_{\text{init}}$，其中每条市场逻辑表示为 $H = \langle \mathcal{C}, \mathcal{B} \rangle$。

### 阶段二：引导式因子生成与优化（内层循环）

输入为一条给定的市场逻辑 $H$、训练/验证数据集、优化目标 $J(\cdot)$、早停阈值 $T_{\text{early}}$。处理步骤（Algorithm 1）：

1. **L2FC-Agent** 将人类可读的市场逻辑 $H$ 转换为规范化结构化形式 $H^{\text{struct}}$，并编译为可执行逻辑约束 $\Gamma = \text{Compile}(H^{\text{struct}})$；
2. **FEG-Agent** 在约束 $\Gamma$ 下生成满足市场逻辑语义的数学因子表达式 $F$；若存在上一轮反馈 $fb^{\text{factor}}$，则将其作为条件输入以指导生成方向；
3. **BacktestEngine** 对生成的因子在训练集和验证集上进行回测，得到指标 $(R_{\text{train}}, R_{\text{val}})$；
4. 若验证指标 $J(R_{\text{val}})$ 优于历史最优，则更新最优指标并重置早停计数器；否则计数器加一；
5. 若未触发早停，**FPF-Agent** 基于 $H^{\text{struct}}$、当前因子 $F$ 和验证指标 $R_{\text{val}}$ 生成因子级反馈，用于下一轮生成。

输出为逻辑级证据 $E^{\text{Logic}} = \langle H^{\text{struct}}, R_{\text{val}}^* \rangle$ 和最优验证指标 $R_{\text{val}}^*$。

### 阶段三：市场逻辑生成与优化（外层循环）

输入为初始逻辑库 $H_{\text{init}}$、数据集、最大迭代轮数 $T$、目标函数 $J(\cdot)$。处理步骤（Algorithm 2）：

1. **MLGA-Agent** 从当前逻辑库中生成一条新的市场逻辑 $H_{\text{current}}$（首轮）或基于历史逻辑、历史证据和逻辑级反馈生成改进逻辑 $H_{\text{new}}$；
2. 将 $H_{\text{current}}$ 送入内层循环，得到该逻辑下的最优因子表现 $R_{\text{val}}^*$；
3. **MLRD-Agent** 从跨因子视角总结和反思市场逻辑，生成逻辑级精炼方向 $fb^{\text{Logic}}$；
4. 若当前逻辑的表现优于历史最佳，则更新 $H_{\text{best}}$ 和 $E_{\text{best}}$；
5. 将新生成的逻辑加入逻辑库和历史记录，进入下一轮，直到达到最大轮数 $T$。

### 系统组成与公式

系统由 6 个核心智能体和 1 个回测引擎组成，形成"挖掘—生成—反馈—精炼"的闭环：

| 智能体 | 所属阶段 | 功能 |
| --- | --- | --- |
| FormulaStructureAgent | 挖掘 | 解析因子表达式的形式结构 |
| FinancialSemanticsMappingAgent | 挖掘 | 将数学操作映射为金融语义 |
| MarketLogicAbstractionAgent | 挖掘 | 将语义抽象为显式市场逻辑 |
| LogicToFinanceConstraintAgent (L2FC-Agent) | 内层循环 | 将市场逻辑编译为可执行约束 $\Gamma$ |
| FactorExpressionGeneratorAgent (FEG-Agent) | 内层循环 | 在约束下生成因子表达式 |
| FactorPerformanceFeedbackAgent (FPF-Agent) | 内层循环 | 基于表现生成因子级反馈 |
| MarketLogicGeneratorAgent (MLGA-Agent) | 外层循环 | 生成新市场逻辑 |
| MarketLogicRefinementDirectionAgent (MLRD-Agent) | 外层循环 | 跨因子反思并给出逻辑精炼方向 |

拓扑关系：挖掘阶段的三智能体为串联流水线；内层循环中 L2FC → FEG → Backtest → FPF 形成闭环；外层循环中 MLGA → 内层循环 → MLRD 形成更高层级的闭环。两个循环均带有早停/最优更新机制。挖掘阶段的三智能体为串联流水线；内层循环中 L2FC → FEG → Backtest → FPF 形成闭环；外层循环中 MLGA → 内层循环 → MLRD 形成更高层级的闭环。

**公式 1：市场逻辑表示**

$$
H = \langle \mathcal{C}, \mathcal{B} \rangle
$$

- $H$：一条市场逻辑；$\mathcal{C}$：一个谓词的合取/析取，每个谓词 $c_i = (v, \text{op}, \theta, w)$ 包含市场变量 $v$、操作符、阈值 $\theta$ 和窗口 $w$，描述"在什么条件下"某种市场现象成立；$\mathcal{B} = (y, d, h)$：目标变量 $y$、方向 $d$（正/负相关）、预测 horizon $h$，描述"预期会发生什么"。

**公式 2：因子表达式**

$$
f_t = \mathcal{O}(x_{t-k:t})
$$

因子是从历史市场数据到当前因子值的映射。AlphaLogics 通过约束算子组合 $\mathcal{O}$ 的搜索空间来确保生成的 $f_t$ 符合市场逻辑。

**评估指标**：$\text{IC}_t = \text{corr}_{\text{Spearman}}(f_t, r_{t+1})$（信息系数）；$\text{ICIR} = \mathbb{E}[\text{IC}] / \sigma[\text{IC}]$（IC 信息比）；AR（年化收益）；IR（信息比）；MDD（最大回撤）。

## 实现与结果

### 实验设置

- **数据集**：CSI 500（OHLCV 数据来自 Baostock）、S&P 500（OHLCV 数据来自 Yahoo Finance）；
- **数据划分**：训练集 2015.01–2019.12，验证集 2020.01–2020.12，测试集 2021.01–2024.12；
- **平台**：Qlib；基础特征为日内收益、日收益、20 日相对成交量、标准化日振幅；预测模型 LightGBM（截面 Z-score 标准化）；交易策略 Top-outside（选取排名前 50 的股票，剔除排名后 5 的股票）；
- **交易成本**：CSI 500 买入 0.0005、卖出 0.0015；S&P 500 买入 0.0000、卖出 0.0005；
- **早停设置**：内层循环 $T_{\text{early}} = 3$；
- **基线**：LSTM、Transformer、LightGBM、TRA、GRU、XGBoost、MLP、O3-mini、Deepseek-V3.1、AlphaForge、RD-Agent、AlphaAgent。

### 关键结果

**CSI 500（2021.01–2024.12）**：

- AlphaLogics IC = **0.0251**，相比次优的 AlphaAgent（0.0221）提升 13.6%；ICIR = **0.2312**，提升 10.5%；
- AR = **16.72%**，相比 AlphaAgent（12.46%）提升 34.2%；IR = **1.5266**，提升 24.8%；
- MDD = **-5.31%**，为所有方法中最低（AlphaAgent 为 -6.65%）。

**S&P 500（2021.01–2024.12）**：

- IC = **0.0093**，相比次优的 AlphaAgent（0.0060）提升 55.0%；ICIR = **0.0878**，提升 70.5%；
- AR = **13.75%**，相比 AlphaAgent（8.57%）提升 60.4%；IR = **1.2658**，提升 31.1%；
- MDD = **-9.06%**，接近 AlphaAgent（-9.44%）但略优。

**市场逻辑挖掘有效性（Table 2）**：对 Alpha101/Alpha158/Alpha360/Alpha191 的反向提取中，数学解释一致性达 94.9%–100.0%，金融解释一致性达 92.7%–98.8%。

**消融实验（Table 3，Persistent vs Transient Logic）**：第 5 轮时，Persistent Logic 的 IC = 0.0232、ICIR = 0.2137；Transient Logic 的 IC = 0.0165、ICIR = 0.1320。Persistent Logic 随轮次稳步提升（IC 从 0.0199 提升至 0.0232），Transient Logic 则逐渐退化——逻辑库的累积效应在长期迭代中价值显著。

### 优势场景

- **美股市场**：优势尤为突出，IC 提升 55%、ICIR 提升 70.5%、AR 提升 60.4%，说明市场逻辑作为高层次语义抽象在跨市场迁移时具有显著优势；
- **多轮迭代场景**：Persistent Logic 持续改进，表明逻辑库的累积效应在长期迭代中价值显著；
- **风险调整后收益**：在 IR 和 MDD 两个指标上均表现最优，说明其不仅提升了收益预测能力，还改善了策略的稳健性。

### 局限性

- **LLM 依赖**：论文未明确披露具体使用的 LLM 型号，仅将 O3-mini 和 Deepseek-V3.1 列为基线对比方法；计算开销较高（多轮 Agent 交互 + 回测）；
- **因子覆盖**：目前仅覆盖基于 OHLCV 和价格-成交量模式的技术因子，对基本面因子未做深入探讨；
- **无开源**：论文未提及代码开源地址。

### 实现细节

因子 DSL（附录 A.2）给出算子族和参数范围：算术运算（+、-、×、//）、截面运算（rank、zscore）、时序聚合（ts_mean、ts_std、ts_min、ts_max、ts_sum，窗口 $w \in \mathbb{N}^+$）、时序变化（ts_delta，滞后 $\ell \in \mathbb{N}^+$）、时序关系（ts_corr、ts_cov）、平滑/衰减（ts_decay、ts_wma）。数据预处理为截面 Z-score 标准化。FPF-Agent 维护最近 $M$ 个候选因子及其验证指标缓冲区。

### 迁移潜力

- **跨资产类别**：技术市场逻辑（动量、反转、量价背离）在股票、期货、加密货币等市场中具有一定普适性，但需重新验证；
- **跨频率**：本文使用日频数据，但市场逻辑本身可适配到小时频、周频等不同频率，只需调整 horizon $h$ 和窗口 $w$；
- **基本面因子扩展**：若将基本面变量（PE、PB、ROE）纳入 DSL 和逻辑表示 $\mathcal{C}$，理论上可扩展至基本面逻辑挖掘；
- **其他预测任务**：市场逻辑的显式表示思路可迁移至其他需要"可解释中间层"的预测任务，如信用风险评估、保险理赔预测。

## 总结

让机器先学会"为什么有效"的市场逻辑，再用这些逻辑去指导生成"怎么算有效"的因子。

1. **反向提炼**：从历史因子库中自动提取人类可读的市场逻辑，建立初始逻辑库；
2. **逻辑编译**：将市场逻辑转化为机器可执行的因子生成约束；
3. **约束生成**：在逻辑约束下让 LLM 生成因子表达式，并回测验证；
4. **因子反馈**：根据回测结果反馈调整因子生成方向，直到早停；
5. **逻辑进化**：聚合多条因子的回测证据，反思并生成更优的市场逻辑，循环迭代。
