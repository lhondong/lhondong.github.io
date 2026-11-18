# AlphaAgentEvo

> **AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning**
> Ziyi Tang, Xuexiong Yin, Weixing Chen, Zechuan Chen, Yongsen Zheng, Wenxuan Ye, Keze Wang, Liang Lin（中山大学 / 南洋理工大学 / 港中大（深圳））｜ ICLR 2026｜ 链接：openreview.net/forum?id=INmZrawUMu｜ 前作：AlphaAgent（KDD 2025，同一团队）｜ 代码：论文承诺接收后开源

## 摘要

Alpha mining（alpha 因子挖掘）旨在从庞大且嘈杂的搜索空间中识别能够产生超额收益的预测性 alpha 因子；然而，现有的基于进化的方法难以促进 alpha 的系统化进化。传统方法如遗传编程（GP）无法解释自然语言指令，且往往无法从失败的尝试中提取有价值的见解，导致可解释性低和探索效率低下。类似地，缺乏系统化进化机制（如长期规划和反思）的现有多智能体方法容易陷入重复的进化套路，导致进化效率低下。本文引入 AlphaAgentEvo，一种用于 alpha mining 的自进化智能体强化学习（Agentic Reinforcement Learning, ARL）框架，将 alpha mining 从脆弱的"搜索-回测-重启"循环转向连续的进化轨迹。在分层奖励函数的引导下，智能体参与搜索空间的自我探索，逐步学习基本要求（如有效的工具调用），然后学习更复杂的目标（如持续性能提升）。通过这一过程，智能体获得高级行为，如长期规划（long-horizon planning）和反思推理（reflective reasoning），使其能够主动响应底层状态（如市场制度转变）。大量实验表明，AlphaAgentEvo 实现了更高效的 alpha 进化，生成多样化且可迁移的 alpha，持续超越广泛的基线。值得注意的是，仅 4B 参数，它就超越了配置有最先进闭源推理模型的 LLM 驱动进化方法，凸显了 ARL 在下一代 alpha mining 中的前景。

## Introduction

Alpha mining 的目标是从海量搜索空间中发现预测性因子。传统上，这一过程被视为"搜索-回测-重启"的循环：生成候选、回测评估、筛选保留、再从零开始。作者认为，这种离散的循环模式无法充分利用历史尝试中的经验，也无法实现因子的渐进式改进。本文将 alpha mining 重新定义为连续的"进化"过程：给定一个种子 alpha，通过多轮交互逐步将其转化为更优版本，同时保留其可解释的核心逻辑。

### 相关工作

| 方法类别 | 代表工作 | 具体局限 |
| --- | --- | --- |
| 遗传编程（GP） | Lin et al. 2019; Schmidt & Lipson 2010 | 依赖启发式搜索和随机突变，无法从失败尝试中学习；不能解释自然语言指令；进化过程缺乏人类可读性 |
| LLM 驱动多智能体 | AlphaAgent（Tang et al., KDD 2025） | 缺乏长期规划和反思机制，容易陷入重复的局部修改；本质上仍是提示驱动，无系统性进化 |
| 工具使用 RL | ToolRL（Qian et al., 2025） | 单轮评估、跨轮耦合弱，无法有效规划多轮进化轨迹；在多轮场景下泛化不足 |
| 推理 LLM 直接进化 | GPT-5-mini、DeepSeek-R1 | 虽具备推理能力，但未经专门训练用于 alpha 进化，缺乏方向感知和持续改进机制 |

核心假设：alpha mining 本质上是一个多轮工具在环的进化过程，而非单轮生成问题。（1）通过将 LLM 驱动的智能体与回测工具进行多轮交互，并让智能体条件于完整的历史轨迹进行反思，可以实现系统性的 alpha 进化；（2）分层奖励函数能够引导智能体从基本合规（有效工具调用）逐步习得高级目标（性能持续提升），避免一开始就陷入复杂的性能优化而忽略基础约束；（3）将 GRPO 扩展为多轮 ARL 框架，可以使小参数模型（1.7B–4B）通过自进化获得超越百亿级闭源大模型的 alpha 进化能力。

## 方法设计

### Pipeline

AlphaAgentEvo 的流程可分为五个阶段：

**阶段一：种子初始化**。输入为一个专家设计的种子 alpha $f_{\text{seed}}$，从分布 $\mathcal{D}_{\text{seed}}$ 中采样。种子作为进化的起点，策略 $\pi$ 需要在保留其可解释核心逻辑的前提下进行改进。

**阶段二：多轮进化轨迹生成**。策略 LLM 与外部回测工具 $\mathcal{U}$ 进行最多 $T$ 轮交互。每轮中，策略 LLM 首先分析历史 alpha 和回测反馈，进行反思推理，然后生成 $k_t$ 个并行的 alpha 候选（工具调用）。所有候选被工具 $\mathcal{U}$ 评估后，反馈被追加到轨迹中。策略 LLM 在生成第 $t$ 轮 token 时，条件于整个过去轨迹 $\tau_{1:t-1}$，实现跨轮反思。

**阶段三：分组优势估计与策略更新**。对于每个输入种子，从旧策略 $\pi_{\text{old}}$ 中 rollout $G$ 组完整轨迹。轨迹按种子分组，组内奖励归一化后估计相对优势 $\hat{A}$。使用 GRPO 目标函数更新策略，只让策略生成的 token 贡献梯度，工具响应 token 被掩码排除。

**阶段四：分层奖励评估**。每轮生成的候选 alpha 进入分层奖励模块，从五个维度评估：工具调用有效性、与种子的一致性、探索多样性、性能改进、连续进步 streak。

**阶段五：种子组合与回测**。进化完成后，从 top 候选中构建多因子组合，在样本外数据上回测验证泛化能力。

### 模块结构

**进化策略 LLM（Policy LLM）**：基础模型 Qwen3-1.7B 或 Qwen3-4B-thinking；输入为任务描述 + 种子 alpha + 历史轨迹；输出为推理 token + 工具调用 token。

**外部回测工具 $\mathcal{U}$（evaluate_factor）**：输入 factor_name、factor_expr、metric（默认 Information_Ratio_with_cost）；在 HS300/CSI500 市场上对 alpha 进行回测；输出回测成功与否、IR 值、AER 等性能指标。工具不披露任何市场或时间范围信息，确保策略 LLM 无法利用数据泄露。

**分层奖励函数**：由五个组件经分层组合而成——基础层：工具调用奖励；方向层：一致性奖励 + 探索奖励；目标层：性能奖励 × 连续奖励。

**GRPO 训练器**：将单轮 GRPO 扩展为多轮 ARL；组内奖励归一化作为基线；KL 散度惩罚防止策略偏离参考模型过远。

### 公式与算法解释

**进化策略学习目标**：

$$
\pi^* = \arg\max_\pi \mathbb{E}_{f_{\text{seed}} \sim \mathcal{D}_{\text{seed}}} \left[ \max_{f \in \mathcal{F}_\pi(f_{\text{seed}})} \left( \mathbb{E}_{\mathbf{X} \sim \mathcal{D}_{\text{evo}}} s(f; \mathbf{X}) + \lambda \mathbb{E}_{\mathbf{X} \sim \mathcal{D}_{\text{test}}} s(f; \mathbf{X}) \right) \right]
$$

$$
\text{s.t. } \text{sim}(f, f_{\text{seed}}) \leq \delta \quad \forall f \in \mathcal{F}_\pi(f_{\text{seed}})
$$

- $\pi$：进化策略（即策略 LLM）；$\mathcal{D}_{\text{test}}$：分布外市场（测试期数据）；$\lambda > 0$：平衡分布内拟合与分布外泛化的权衡系数；$\text{sim}(f, f_{\text{seed}})$：基于 AST 重叠的结构相似度；$\delta$：相似度上限，确保进化在种子邻域内进行。

直觉：不同于直接优化单个 alpha，本文优化的是"进化策略"本身，即从任意种子出发持续产生更优 alpha 的能力。约束 $\text{sim}(f, f_{\text{seed}}) \leq \delta$ 防止无约束全局优化导致的过拟合和不可解释性。

**ARL 形式的 GRPO 目标函数**：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim D,\ \mathcal{T}=\{\tau_1,\ldots,\tau_G\} \sim \pi_{\text{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{\sum_t M_{i,t}} \sum_t M_{i,t} \min\left( \frac{\pi_\theta(\tau_{i,t}|x,\tau_{i,<t},\mathcal{U})}{\pi_{\text{old}}(\tau_{i,t}|x,\tau_{i,<t},\mathcal{U})} \hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta(\tau_{i,t}|x,\tau_{i,<t},\mathcal{U})}{\pi_{\text{old}}(\tau_{i,t}|x,\tau_{i,<t},\mathcal{U})}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{i,t} \right) - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right]
$$

- $\tau_{i,t}$：第 $i$ 条轨迹的第 $t$ 个 token；$M_{i,t}$：掩码，排除工具响应 token（仅策略生成 token 贡献梯度）；$\frac{1}{\sum_t M_{i,t}}$：按有效 token 数归一化，处理变长轨迹；$\hat{A}_{i,t}$：组内归一化的相对优势；$\beta = 0.001$。

直觉：标准 GRPO 为单轮文本优化设计。本文将其扩展为多轮 alpha 进化：轨迹包含多轮推理 + 工具调用 + 工具响应，但只让策略 LLM 自己生成的 token 承担梯度责任。组内归一化用组平均奖励作基线，消除不同种子 alpha 绝对难度差异的影响。

**分层奖励函数**：

$$
R(\tau) = \frac{\min(R_{\text{cons}}(\tau), C_{\text{cons}}) + \min(R_{\text{expl}}(\tau), C_{\text{expl}})}{\min(R_{\text{tool}}(\tau), C_{\text{tool}})} + \min(R_{\text{perf}}(\tau), C_{\text{perf}}) \cdot \min(R_{\text{streak}}(\tau), C_{\text{streak}})
$$

五个组件：

1. **工具调用奖励** $R_{\text{tool}}(\tau) = \alpha_{\text{succ}} \cdot N_{\text{succ}} - \alpha_{\text{fail}} \cdot N_{\text{fail}}$（$\alpha_{\text{succ}}=0.1$、$\alpha_{\text{fail}}=0.2$）：将每次工具调用视为成本，惩罚无效调用；
2. **一致性奖励** $R_{\text{cons}}(\tau) = \sum_{f_i \in F_{\text{succ}}} \alpha_{\text{cons}} \cdot \mathbb{1}[\text{sim}(f_i, f_{\text{seed}}) > h_{\text{low}}]$，其中 $\text{sim}(f_i, f_{\text{seed}}) = \frac{|AST(f_i) \cap AST(f_{\text{seed}})|}{\max(|AST(f_i)|, |AST(f_{\text{seed}})|)}$，$h_{\text{low}} = 0.1$、$\alpha_{\text{cons}}=0.02$：防止随机漂移破坏可解释性；
3. **探索奖励** $R_{\text{expl}}(\tau) = \sum_{f_i \in F_{\text{succ}}} \alpha_{\text{exp}} \cdot (1 - \max_{f_j \in F_{<i}} \text{sim}(f_i, f_j))$（$\alpha_{\text{exp}}=0.02$）：鼓励生成与先前所有候选差异大的 alpha，避免重复探索同一局部最优；
4. **性能奖励** $R_{\text{perf}} = \alpha_{\text{perf}} \cdot \log(1 + \exp(s(f^*) - \max(0, s(f_{\text{seed}}))))$（$\alpha_{\text{perf}}=0.1$，$f^*$ 为轨迹中最佳 alpha）：使用对数缩放处理噪声指标，只有当性能超过种子且为正时才给予显著奖励；
5. **连续奖励** $R_{\text{streak}} = \alpha_{\text{streak}} \cdot N_{\text{streak}}$（$\alpha_{\text{streak}}=0.15$，$N_{\text{streak}}$ 为最长连续进步序列长度）：鼓励稳定的进化策略。

总奖励的组合逻辑：一致性 + 探索被工具使用归一化（避免用大量调用 brute-force 搜索），性能与连续奖励相乘（两者缺一不可）。每个组件被上限封顶：$C_{\text{tool}}=1$、$C_{\text{cons}}=0.2$、$C_{\text{expl}}=0.3$、$C_{\text{perf}}=0.5$、$C_{\text{streak}}=0.6$。

## 实现与结果

### 实验设置

- **数据集**：AlphaEvo500（本文构建的专家整理数据集）：350 训练种子 / 50 验证 / 100 测试；Alpha158 作为额外测试集评估泛化；
- **回测设置**：市场 HS300 和 CSI500；时间 2023 年 1 月至 2025 年 11 月（训练数据仅 2023-01-01 至 2024-01-01 一年，加速迭代）；评估期 1 为熊市（2023），评估期 2 为牛市（2024–2025）；协议为单因子评估，每轮 long top 10% 股票，每 5 个交易日再平衡；
- **评估指标**：VR（Valid Ratio，语法有效且可回测的 alpha 百分比）、pass@3 / pass@5（第 3 轮/第 5 轮成功率）、AER（年化超额收益）、IR（信息比率）；
- **基线**：GP（4/20/50 后代）、AlphaAgent、GEPA、推理 LLM（Qwen3-1.7B、Qwen3-4B-thinking、GPT-5-mini、DeepSeek-R1）、工具 RL（ToolRL-1.7B/4B）；多因子组合对比 LightGBM、Stock-Mixer、AlphaQCM；
- **训练配置**：10 × RTX 4090 GPU；框架 Verl；模型 Qwen3（1.7B / 4B）；150 步；批次 20 种子 alpha × 3 rollouts/种子，最多 3 轮/轨迹，最多 4 工具调用/轮；学习率 1e-6，warmup ratio 0.1，Adam，KL 系数 $\beta = 0.001$；测试 checkpoint 为第 80 步。

### 主结果：AlphaEvo500 测试集

| 方法 | HS300 VR | HS300 Pass@3 | HS300 Pass@5 | CSI500 VR | CSI500 Pass@3 | CSI500 Pass@5 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | 0.676 | 0.08 | 0.11 | 0.657 | 0.35 | 0.43 |
| Qwen3-4B-thinking | 0.942 | 0.36 | 0.47 | 0.951 | 0.68 | 0.78 |
| GPT-5-mini | 0.970 | 0.75 | 0.88 | 0.972 | 0.73 | 0.82 |
| DeepSeek-R1 | 0.872 | 0.68 | 0.71 | 0.886 | 0.71 | 0.86 |
| ToolRL-1.7B | 0.864 | 0.74 | 0.78 | 0.851 | 0.66 | 0.74 |
| ToolRL-4B | 0.954 | 0.75 | 0.81 | 0.961 | 0.73 | 0.76 |
| GEPA (GPT-5-mini) | 0.992 | 0.87 | 0.90 | 0.971 | 0.86 | 0.91 |
| GEPA (DeepSeek-R1) | 0.977 | 0.83 | 0.87 | 0.978 | 0.82 | 0.88 |
| **AlphaAgentEvo-1.7B** | 0.940 | 0.77 | 0.90 | 0.923 | 0.76 | 0.78 |
| **AlphaAgentEvo-4B** | **0.979** | **0.97** | **0.97** | **0.977** | **0.93** | **0.95** |

- AlphaAgentEvo-4B 在 HS300 上 Pass@3 达到 0.97，显著超越最强基线 GEPA(GPT-5-mini) 的 0.87；在 CSI500 上 Pass@5 达到 0.95；
- 仅 1.7B 的 AlphaAgentEvo 就在 HS300 Pass@3 上超越 GPT-5-mini（0.77 vs 0.75）。

### 外部测试集：Alpha158

| 方法 | 熊市 VR | Pass@3 | Pass@5 | 牛市 VR | Pass@3 | Pass@5 |
| --- | --- | --- | --- | --- | --- | --- |
| GP(4) | 0.766 | 0.000 | 0.074 | 0.823 | 0.003 | 0.003 |
| GP(50) | 0.619 | 0.022 | 0.024 | 0.633 | 0.094 | 0.107 |
| AlphaAgent (GPT-3.5) | 0.905 | 0.236 | 0.495 | 0.900 | 0.643 | 0.783 |
| AlphaAgent (DeepSeek-R1) | 0.975 | 0.294 | 0.550 | 0.966 | 0.750 | 0.848 |
| GPT-5-mini | 0.988 | 0.156 | 0.293 | 0.975 | 0.828 | 0.903 |
| **AlphaAgentEvo-1.7B** | 0.952 | 0.506 | 0.613 | 0.917 | 0.909 | 0.926 |
| **AlphaAgentEvo-4B** | **0.982** | **0.581** | **0.725** | **0.982** | **0.963** | **0.994** |

AlphaAgentEvo-4B 在 Alpha158 熊市 Pass@5 达到 0.725、牛市 Pass@5 达到 0.994，接近饱和；GP 方法 50 个后代才勉强达到牛市 Pass@5 = 0.107。

### 多因子组合性能（2024-01 至 2025-11）

| 方法 | 类别 | AER | IR | MDD |
| --- | --- | --- | --- | --- |
| LightGBM | TS model | -0.009 | 1.192 | -0.195 |
| Stock-Mixer | TS model | 0.013 | 1.977 | -0.182 |
| AlphaQCM | RL Framework | 0.027 | 1.815 | -0.192 |
| AlphaAgent | Multi-LLM-agent | 0.064 | 2.046 | -0.196 |
| GPT-5-mini | Single-LLM-agent | -0.158 | 0.587 | -0.213 |
| ToolRL-4B | Single-LLM-agent | -0.027 | 1.532 | -0.215 |
| **AlphaAgentEvo-4B** | Single-LLM-agent | **0.129** | **2.442** | **-0.176** |

相比 AlphaAgent，AER 提升 101.6%（0.064 → 0.129），IR 提升 19.3%（2.046 → 2.442）。

### 进化分析与消融

- **进化轨迹**：AlphaAgentEvo 的 mean IR 增长速度持续快于 ToolRL，差距随轮次扩大；标准差更高说明探索更积极，但与种子的一致性保持稳定，证明是 agent 级策略进化而非随机漂移；
- **消融**：移除探索奖励后 AlphaEvo500 Pass@3 从 0.65 降至 0.54；移除一致性奖励后从 0.65 降至 0.51——两者均关键且互补；
- **多样性**：AlphaAgentEvo top 20 alpha 平均相似度 0.039、最大 0.263；DeepSeek-R1 平均 0.046、最大 0.583；Qwen3-4B 最大 0.600；AlphaAgent(GPT-4) 平均 0.058、最大 0.333。

### 案例分析：4B 模型的反思能力

种子因子 **CNTN60**（统计过去 60 天收盘价下跌的天数占比）。基座模型 Qwen3-4B-thinking 的进化方式是机械调参数（换窗口长度、加 Z-score 标准化）。AlphaAgentEvo-4B 先质疑种子因子的语义——下跌天数多代表看空信号，但直接用这个信号可能选到持续下跌的烂股票——然后提出将下跌天数和 RSI 动量指标结合，生成 **RSI_DOWN_COUNT** 因子，IR 从负值升至 0.94；第二轮加入波动率（ATR），IR 升至 1.19。这种"先质疑、再组合、再验证"的推理链路是强化学习训练出来的。

### 局限性

1. **种子依赖性**：进化质量依赖于初始种子的质量，论文未讨论如何从随机起点冷启动；
2. **训练成本**：需要 10 × RTX 4090 GPU 训练 150 步，对小团队门槛较高；
3. **市场覆盖有限**：仅在中国 A 股 HS300/CSI500 上验证，未覆盖美股、加密货币等资产类别；
4. **评估指标选择**：不使用 IC 等传统 alpha 指标（种子为布尔表达式，未选中股票值为 NaN），可能限制与文献的直接对比；
5. **工具不透明**：回测工具的具体实现（滑点、手续费模型）未完全披露；
6. **奖励超参数多**：五层奖励的系数与上限、一致性下限、KL 系数等缺乏选择的理论依据，也未做跨市场验证；
7. **任务特殊性**：因子公式是结构化的形式语言，恰是 RL 擅长的环境；4B 在此任务上超越大模型，不代表在其他金融 AI 任务上也能超越。

### 迁移潜力

1. **到其他因子类型**：当前框架使用布尔表达式种子，可扩展至连续值 alpha 或更复杂的组合策略；
2. **到其他市场**：只需替换回测工具中的市场数据即可应用于美股、期货、加密货币等；
3. **到其他 DSL 生成任务**：任何需要从种子程序出发通过多轮工具反馈进行改进的场景（如 SQL 查询优化）均可套用 ARL + 分层奖励框架；
4. **与更大模型结合**：若将训练好的 ARL 策略蒸馏到更小模型，可实现边缘设备部署。

## 总结

让小语言模型通过多轮反思和分层奖励，在回测工具的反馈下自学如何把一个种子因子逐步改得更好。

1. 输入一个种子 alpha 公式；
2. 小语言模型分析历史尝试和回测结果，进行反思推理；
3. 生成多个改进后的 alpha 候选，提交回测工具评估；
4. 根据工具反馈计算分层奖励（调用是否有效、是否偏离种子太远、是否重复、性能是否提升、是否持续进步）；
5. 用 GRPO 更新模型，使策略学会从任意种子出发系统性地进化出更好 alpha；
6. 重复多轮后，从 top 候选构建多因子组合输出策略。

本文的核心贡献不是 4B 模型本身，而是证明了一个思路——**在特定任务上，一个经过强化学习训练的小模型，可以超越没有训练过的大模型**。优势在于成本大幅降低（训练一次 4B 远低于持续调用闭源 API）、数据隐私有保障（小模型本地部署）、可持续进化（模型不是一锤子买卖）。对于量化从业者，最大的启发是这个问题：**你的因子挖掘系统，到底是在搜索还是在学习？** 如果只是用大模型反复生成因子然后筛选，那做的还是搜索；真正的学习，是系统本身在变强。
