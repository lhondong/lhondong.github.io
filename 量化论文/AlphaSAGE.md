# AlphaSAGE

> **AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration**
> Binqi Chen, Hongjun Ding, Ning Shen, Taian Guo 等（北京大学 / Zhengren Quant / Baruch / UBC）｜ arXiv 2509.25055｜ 代码：github.com/BerkinChen/AlphaSAGE

## 摘要

自动挖掘预测信号（alpha）是量化金融中的核心挑战。现有的基于强化学习（RL）的 alpha 挖掘框架面临三个相互关联的问题：奖励稀疏（reward sparsity），即只有在完整公式构建完成后才能获得有意义的反馈；语义不充分的序列表示（semantically inadequate sequential representations），无法捕捉数学表达式的组合本质；以及标准 RL 目标函数最大化期望回报，这本质上将策略推向单一最优模式，导致缺乏多样性。本文提出 AlphaSAGE，一个用于鲁棒探索的结构感知 alpha 挖掘框架，包含三项核心创新：（1）基于关系图卷积网络（RGCN）的结构感知编码器，在 alpha 的抽象语法树（AST）表示上运行；（2）基于生成流网络（GFlowNets）的全新生成框架，使采样概率与 alpha 质量成正比；（3）密集的多维度奖励结构，结合终端性能、结构对齐和新颖性。在中国和美国真实股票市场上的实验表明，AlphaSAGE 在信号质量和组合实现收益方面均优于现有方法，同时保持可解释性。

## Introduction

Alpha mining 旨在从历史金融数据中自动发现能够预测未来收益的数学表达式。近年来，RL 方法被广泛应用于该任务，但作者观察到现有 RL 框架存在三个根本性缺陷：

| 痛点 | 具体方法/现象 | 问题描述 |
| --- | --- | --- |
| 奖励稀疏 | AlphaGen、AlphaForge 等 RL 方法 | 只有在完整公式生成后才能通过回测获得单一标量奖励，中间步骤缺乏反馈，导致学习信号极弱 |
| 序列表示语义不足 | 基于 LSTM/Transformer 的 token 序列编码 | 将数学表达式视为线性 token 序列，忽略了运算符和操作数之间的层次结构关系，对语义等价但语法不同的表达式给出不同编码 |
| 模式坍缩 | 标准 RL 目标（最大化期望回报） | REINFORCE、PPO 等方法天然倾向于收敛到单一高奖励模式，生成的 alpha 缺乏多样性，容易过拟合到特定市场状态 |

作者的核心假设：（1）用图神经网络在 AST 上编码 alpha 可以捕捉到被序列模型丢失的组合语义；（2）GFlowNets 通过使采样概率与奖励成正比而非取 argmax，能够自然地维持高奖励样本的多样性；（3）将结构对齐和新颖性显式纳入奖励函数，可以引导生成器产生既有预测力又不冗余的 alpha 集合。

## 方法设计

### Pipeline

AlphaSAGE 的整体流程可分为四个阶段：

**阶段一：AST 解析与图编码**。输入为一个待生成的 alpha 公式或部分表达式。系统将 alpha 解析为抽象语法树（AST），节点代表运算符（如 Add、Mean、Corr）或原始特征（如 $close、$volume），边代表计算依赖关系。与线性 token 序列不同，AST 对语义等价但括号位置不同的表达式具有不变性。

**阶段二：RGCN 结构编码**。使用关系图卷积网络对 AST 进行编码。RGCN 为不同类型的边定义关系特定的权重矩阵，从而区分一元运算符与其操作数、可交换二元运算符两侧、不可交换二元运算符的左右操作数、滚动窗口运算符与其特征/时间操作数等六种关系类型。通过多层消息传递和最大池化（MaxPooling）获得图级嵌入 $e_{\text{alpha}}$。

**阶段三：GFlowNet 生成策略**。将 alpha 生成建模为有向无环图（DAG）中的轨迹 $\tau = (s_0 \rightarrow s_1 \rightarrow \cdots \rightarrow s_n = \text{alpha})$。每个状态 $s_t$ 是一个部分 AST，动作是向某个开放的叶节点添加一个有效 token（运算符或特征）。策略网络以当前部分 AST 的 RGCN 嵌入为条件，输出被掩码的动作 logits，确保仅采样语法有效的下一步。系统还引入了早期停止机制（early stopping），以概率 $p = \text{Len}(s_t)/\text{MaxLen}$ 在任意时刻终止生成，从而允许生成长度可变的 alpha。

**阶段四：多面奖励评估与组合**。生成的完整 alpha 进入评估模块，计算由三部分组成的奖励函数：终端预测性能（IC）、结构感知一致性（SA）、以及与已知 alpha 库的新颖性（NOV）。GFlowNet 通过 Trajectory Balance Loss 进行训练，使策略收敛到采样概率 $P(\text{alpha}) \propto R(\text{alpha})$ 的分布。最后，通过线性回归动态重选 alpha 子集进行组合，遵循 AlphaForge 的动态组合方法。

### 模块结构

**AlphaGenerator（生成器）**：RGCN 编码器（6 层关系特定图卷积、6 种关系类型）输出节点嵌入 $h_v^{(L)}$ 和图嵌入 $e_{\text{alpha}}$；策略头以 $e_{\text{alpha}}$ 为条件输出覆盖所有有效 token 的 logits，经掩码后通过 softmax 采样；停止决策头输出二元决策（继续扩展/停止生成）。

**AlphaEvaluator（评估器）**：回测引擎计算 alpha 在历史和交叉截面上的 IC；结构嵌入模块计算生成 alpha 与种子 alpha/已知 alpha 库的结构相似度；行为相似度模块计算 alpha 在时序上的 Z-score 特征向量，用于 KNN 行为距离；奖励聚合器按时间退火调度将 $R_{\text{IC}}$、$R_{\text{SA}}$、$R_{\text{NOV}}$ 合成为最终标量奖励。

### 公式与算法解释

**RGCN 编码公式**：

$$
h_v^{(l)} = \text{ReLU}\left(\sum_{r\in\mathcal{R}}\sum_{u\in\mathcal{N}_r(v)}\frac{1}{c_{v,r}}W_r^{(l)}h_u^{(l-1)} + W_0^{(l)}h_v^{(l-1)}\right)
$$

- $h_v^{(l)}$：节点 $v$ 在第 $l$ 层的嵌入向量；$\mathcal{R}$：关系类型集合，共 6 种；$\mathcal{N}_r(v)$：节点 $v$ 在关系 $r$ 下的邻居集合；$c_{v,r}$：归一化常数；$W_r^{(l)}$：关系 $r$ 在第 $l$ 层的可学习权重矩阵；$W_0^{(l)}$：自环连接的权重矩阵。

直觉：标准 GCN 对所有边使用同一权重，无法区分 `a+b` 中 `a` 和 `b` 的可交换关系与 `a/b` 中被除数和除数的不可交换关系。RGCN 通过为每种关系分配独立权重，使模型能够学习到不同语法位置上的不同语义角色。图级嵌入通过最大池化获得：$e_{\text{alpha}} = \text{MaxPooling}(\{h_v^{(L)}\}_{v\in V_{\text{alpha}}})$。

**GFlowNet Trajectory Balance Loss**：

$$
\mathcal{L}_{\text{TB}}(\tau) = \left(\log Z_\theta + \sum_{t=1}^n \log P_F(s_t|s_{t-1};\theta) - \log R(s_n) - \sum_{t=1}^n \log P_B(s_{t-1}|s_t;\theta)\right)^2
$$

- $Z_\theta$：学习分区函数，使流守恒条件成立；$P_F$：前向策略（生成器）；$P_B$：后向策略（通常固定为均匀分布）；$R(s_n)$：终端状态（完整 alpha）的奖励。

直觉：标准 RL 的轨迹损失只最大化终端奖励，导致所有高奖励轨迹的概率被无差别地提升。而 TB Loss 要求沿任意轨迹的总前向概率等于总后向概率乘以终端奖励，等价于要求从策略中采样 alpha 的概率正比于其奖励 $P(\text{alpha}) \propto R(\text{alpha})$，从而天然维护多个高奖励模式而非收敛到单一模式。

**多面奖励函数**：

$$
R(\text{alpha}, T) = R_{\text{IC}}(\text{alpha}) + \lambda(T) \cdot R_{\text{SA}}(\text{alpha}) + \eta(T) \cdot R_{\text{NOV}}(\text{alpha})
$$

其中 $T$ 为训练时间步，$\lambda(T)$ 和 $\eta(T)$ 为退火调度函数。

1. **终端性能奖励 $R_{\text{IC}}$**：$R_{\text{IC}}(\text{alpha}, y) = \left|\mathbb{E}_d\left[\text{Cov}(\text{alpha}(X_d), y_d) / \sqrt{\text{Var}(\text{alpha}(X_d)) \cdot \text{Var}(y_d)}\right]\right|$，即 alpha 预测值与真实收益的跨截面相关系数的绝对值；
2. **结构感知奖励 $R_{\text{SA}}$**：$R_{\text{SA}}(\text{alpha}_i) = \exp(-\sum_{j\in N_K(\text{alpha}_i)} w_{ij} \cdot d_{\text{behav}}(\text{alpha}_i, \text{alpha}_j))$，其中行为距离 $d_{\text{behav}}$ 基于 Z-score 特征向量的均方差，使用 KNN（$K=5$）加权相似度。直觉：如果结构相似的 alpha 在行为上也相似，则结构编码是可信的；如果结构相似但行为差异大，则可能是过拟合或伪相关；
3. **新颖性奖励 $R_{\text{NOV}}$**：$R_{\text{NOV}}(\text{alpha}) = 1 - \max_{\text{alpha}'\in F_{\text{known}}} |\text{IC}(\text{alpha}, \text{alpha}')|$，鼓励生成与现有 alpha 预测模式不同的新信号；
4. **熵正则化**：$\mathcal{L}_{\text{final}} = \mathbb{E}_{\tau}[\mathcal{L}_{\text{TB}}(\tau)] + \beta \cdot \mathcal{L}_{\text{ENT}}$，在 TB Loss 基础上增加策略熵奖励，进一步鼓励探索。

## 实现与结果

### 实验设置

**数据集**：中国 A 股 CSI300、CSI500、CSI1000 成分股（训练 2010–2020，验证 2021，测试 2022–2024）；美国 S&P500 成分股（训练 2010–2016，验证 2017，测试 2018–2024）。

**超参数**：RGCN 层数 6 层；关系类型数 6 种；早期停止概率 $p = \text{Len}(s_t)/\text{MaxLen}$；奖励权重退火从 0 逐渐增至目标值；KNN 参数 $K=5$。

### 主结果：单因子 IC/ICIR 对比（测试集）

| 数据集 | 最佳基线方法 | 最佳基线 IC / ICIR | AlphaSAGE IC / ICIR |
| --- | --- | --- | --- |
| CSI300 | AlphaGen | 0.058 / 0.414 | **0.079 / 0.496** |
| CSI500 | AlphaAgent | 0.053 / 0.396 | **0.054 / 0.379** |
| CSI1000 | AlphaAgent | 0.072 / 0.579 | **0.076 / 0.582** |
| S&P500 | AlphaAgent | 0.048 / 0.479 | **0.052 / 0.493** |

- 在 CSI300 上，AlphaSAGE 的 IC 比最佳基线 AlphaGen 提升 36.2%（0.058 → 0.079），ICIR 提升 19.8%（0.414 → 0.496）；
- 在 CSI1000 上，IC 提升 5.6%、ICIR 提升 0.5%；在美国 S&P500 上，IC 提升 8.3%、ICIR 提升 2.9%。

**组合层面（CSI300）**：年化收益率（AR）7.62%，最大回撤（MDD）-17.3%，夏普比率（SR）1.71。

### 消融实验（CSI300 测试集）

| 配置 | IC | ICIR | SR |
| --- | --- | --- | --- |
| Plain GFlowNet | 0.048 | 0.393 | 0.72 |
| +Early Stopping only | 0.046 | 0.313 | -0.11 |
| +GNN encoder | 0.070 | 0.495 | 1.25 |
| +SA reward | 0.071 | 0.453 | 1.14 |
| +Novelty reward | 0.075 | 0.494 | 1.53 |
| +Entropy (full AlphaSAGE) | **0.079** | **0.496** | **1.71** |

- **最大单组件提升**来自 GNN encoder：IC 从 0.048 提升至 0.070（+45.8%）；
- 熵正则化在完整配置中贡献了最后的增量：SR 从 1.53 提升至 1.71；
- 仅添加 Early Stopping 反而导致性能下降（SR 从 0.72 降至 -0.11），说明需要与其他组件协同。

## 总结

用图神经网络理解数学公式的树状结构，用流网络按质量比例采样，同时奖励性能高、结构合理且不重复的因子。

1. 将数学公式解析成树状结构（AST）；
2. 用关系图神经网络对树进行编码，区分不同语法位置的角色；
3. 用生成流网络在 DAG 中按概率采样公式，概率与奖励成正比；
4. 奖励由三部分组成：预测准不准、结构合不合理、与已知因子重不重复；
5. 用线性回归动态组合生成的因子，输出最终策略。
