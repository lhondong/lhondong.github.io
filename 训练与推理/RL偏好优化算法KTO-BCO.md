# RL/偏好优化算法详解：PPO、DPO、GRPO、KTO、BCO、Step-KTO、TR-KTO、IPO、ORPO、SimPO

> 说明：本文所有公式、实验数字、会议信息均逐一对照 arXiv 论文原文（含 HTML 全文）核实，不编造论文信息。核实日期 2026-06。部分名称（如 Step-BCO）在公开文献中未找到对应论文，文中会特别标注。引用具体数字时均注明数据集、base model 与对照基线，便于查证。

## 0. 前置概念：为什么需要这些算法？

大模型训练通常分三阶段：

1. **Pre-training**：在海量文本上预测下一个 token，学习语言知识和世界知识。
2. **SFT（Supervised Fine-Tuning）**：用高质量 `(prompt, answer)` 对做监督学习，学会"按人类喜欢的格式回答"。
3. **Alignment / Post-training**：让模型输出更符合人类偏好（有用、无害、诚实、会推理）。

对齐阶段的数据通常是**偏好数据**，常见三种形式：

| 数据形式 | 样子 | 说明 |
|---|---|---|
| **Pairwise preference** | `(x, y_w, y_l)` | 同一个问题，已知答案 A 比答案 B 好 |
| **Binary feedback** | `(x, y, label)` | 只知道单个答案好不好（赞/踩） |
| **Process / verifiable reward** | `(x, step_1, …, final)` 或可自动判定的对错 | 中间步骤或最终结果有客观信号 |

围绕这些数据，逐渐分化出两条主要路线：

- **RLHF / RLVR 路线**：先训练 Reward Model（或用规则奖励），再用 PPO/GRPO 等强化学习算法优化策略。
- **Direct Preference Optimization 路线**：不用显式在线 RL，直接把偏好数据当监督信号，用一个 loss 端到端训练。

### 0.1 谱系图（演化关系）

```
                          策略梯度 (Policy Gradient)
                                   │
                                  TRPO ──→ PPO ────────────────┐
                                   │        │                  │  (去掉 Critic，
                                   │        │  RLHF: RM+PPO     │   组内 baseline)
                                   │        ▼                  ▼
                                   │   传统 RLHF 三件套      GRPO (DeepSeekMath)
                                   │   (RM → PPO → KL)         │
                                   │                          ├─→ Dr.GRPO (去 std/长度归一化偏差)
                                   │                          ├─→ DAPO (解耦 clip + 动态采样)
                                   │                          └─→ DeepSeek-R1 (大规模 RLVR)
                                   ▼
   KL 约束 RL 的闭式最优解 π* ∝ π_ref · exp(r/β)
                                   │
                                  DPO ──── 把 RM 解析消掉，变成监督式分类 loss
                                   │
        ┌──────────────┬──────────┼───────────────┬───────────────┐
        ▼              ▼          ▼               ▼               ▼
   IPO (ΨPO/MSE)   ORPO(单阶段)  SimPO(无ref)   TR-DPO/IPO/KTO   ΨPO 一般框架
   防确定性过拟合   SFT+odds      长度归一化       动态参考模型
        
   二元反馈分支 (Binary feedback, 不需要成对数据)：
        KTO (前景理论/HALO) ──→ Step-KTO (步骤级)
              │
              └─→ BCO (二分类器 logit 当奖励，BCE 是 DPO 的上界)
```

### 0.2 前置基石：KL 约束 RL 的闭式最优解 π\* ∝ π_ref · exp(r/β)

谱系图正中间那一行不是某个算法，而是**整个直接偏好优化家族的理论起点**。它是"在 KL 约束下最大化奖励"这个优化问题的解析解，PPO 在逼近它、DPO 从它反解出损失，值得单独讲清楚。

#### 它解决的优化问题

RLHF 想让策略 $\pi$ 最大化奖励 $r$，但又不能离参考模型 $\pi_{\text{ref}}$（通常是 SFT 模型）太远，否则 reward hacking、胡言乱语。于是加一个 KL 惩罚：

$$
\max_{\pi}\ \mathbb{E}_{y\sim\pi(\cdot|x)}\big[r(x,y)\big]\;-\;\beta\, D_{\mathrm{KL}}\!\big(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
$$

- $\beta$ 是"缰绳松紧"：越大越贴近 $\pi_{\text{ref}}$，$\beta\to 0$ 退化成纯粹最大化奖励。
- 关键：这个问题**对每个固定的 $x$ 独立、对分布 $\pi$ 有闭式最优解**——不需要梯度下降就能写出答案。

#### 推导（拉格朗日乘子）

把 KL 展开、取负号变成最小化，并加上"$\pi$ 是合法概率分布"的约束 $\sum_y\pi(y|x)=1$：

$$
\min_{\pi}\ \sum_y \pi(y|x)\Big[-r(x,y)+\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\Big]\quad \text{s.t.}\ \sum_y \pi(y|x)=1
$$

用拉格朗日乘子 $\lambda$ 处理约束，对 $\pi(y|x)$ 求偏导令其为 0：

$$
-r(x,y)+\beta\Big(\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+1\Big)+\lambda=0
$$

解出 $\pi$，归一化常数由约束定死，即**配分函数** $Z(x)$：

$$
\boxed{\ \pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big),\qquad Z(x)=\sum_{y}\pi_{\text{ref}}(y|x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)\ }
$$

$\propto$ 号就是"省略掉只跟 $x$ 有关的归一化因子 $1/Z(x)$"的写法。

#### 直觉：对参考分布做指数重加权（exponential tilting）

- 拿 SFT 模型 $\pi_{\text{ref}}$ 当底座（先验）；
- 每个候选回答 $y$ 按奖励 $r(x,y)$ 乘上权重 $\exp(r/\beta)$：**奖励高的概率被放大，奖励低的被压低**；
- $\beta$ 控制锐利程度：$\beta$ 小 → 指数项大 → 分布变尖，几乎把概率全押到奖励最高的 $y$（接近 argmax，贪婪）；$\beta$ 大 → 权重接近 1 → $\pi^*$ 几乎不动，仍是 $\pi_{\text{ref}}$（保守）。
- 从能量模型看：$\pi^*$ 是一个玻尔兹曼分布，$r/\beta$ 是负能量，温度就是 $\beta$。

#### 为什么重要 / 卡在哪

致命障碍是 **$Z(x)$ 算不出来**——要对所有可能回答 $y$ 求和（语言模型的序列空间是天文数字级）。所以历史上只能绕道：不直接构造 $\pi^*$，而是用 PPO 在线采样、迭代逼近它。

DPO 的突破正是**利用这个式子却躲开 $Z(x)$**：把它反解成"用策略表示奖励"

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)
$$

代入 Bradley-Terry 时偏好只依赖**奖励之差**，而 $\beta\log Z(x)$ 只跟 $x$ 有关、对 $y_w/y_l$ 相同，**做差时直接抵消**——于是无需算 $Z(x)$、无需独立奖励模型，得到纯监督式的 DPO loss（详见 §2.2）。

### 0.3 数学符号总表（看不懂公式先查这里）

后面几乎每个算法都反复用到下面这些符号。先把它们记熟，再看公式会顺畅很多。

| 符号 | 读法/名字 | 通俗含义 |
|---|---|---|
| $x$ | prompt / 输入 | 用户的问题或指令 |
| $y$ | response / 回答 | 模型针对 $x$ 生成的一整段文字 |
| $y_w,\ y_l$ | winner / loser | 一对回答里人类**更喜欢的**（w=win）和**更差的**（l=lose） |
| $\pi_\theta(y\mid x)$ | policy 策略 | 待训练的模型；给定问题 $x$，它生成回答 $y$ 的**概率**。$\theta$ 是模型参数 |
| $\pi_{\text{ref}}(y\mid x)$ | reference 参考模型 | 训练开始前的模型（通常是 SFT 模型），当"锚点"，被冻结不动 |
| $\pi_{\theta_{\text{old}}}$ | 旧策略 | 上一轮采样时用的模型快照（PPO/GRPO 里用来算概率比） |
| $r(x,y)$ | reward 奖励 | 回答 $y$ 有多好的一个标量分数，越大越好 |
| $\beta$ | beta | "别偏离参考模型太多"的力度旋钮；也常控制奖励的缩放 |
| $\sigma(\cdot)$ | sigmoid | 把任意实数压到 $(0,1)$ 的 S 形函数，$\sigma(z)=\frac{1}{1+e^{-z}}$，常用来表示"概率" |
| $\log$ | 对数 | 这里都是自然对数；概率取 log 后，相乘变相加，数值更好算 |
| $\mathbb{E}[\cdot]$ | expectation 期望 | "平均"，代码里就是对一个 batch 的样本求平均 |
| $\mathcal{L}$ | loss 损失 | 训练要**最小化**的目标；loss 越小代表模型越符合我们想要的 |
| $\hat A_t$ | advantage 优势 | 某个动作比"平均水平"好多少；正=比平均好，负=比平均差 |
| $D_{\text{KL}}(\pi\Vert\pi_{\text{ref}})$ | KL 散度 | 衡量两个分布差多远；这里衡量"新模型和参考模型分歧多大" |

> 一个反复出现的核心量——**隐式奖励** $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$：它不是单独训练的奖励模型，而是直接用"**新模型比参考模型更/更不愿意生成这个回答**"来当奖励。比值 >1（新模型更爱它）→ log 为正 → 奖励为正；比值 <1 → 奖励为负。DPO、KTO、BCO、IPO 全都建立在这个量上，务必记住。

## 1. PPO（Proximal Policy Optimization）

### 1.1 一句话直觉
PPO 是经典 on-policy RL 算法，核心思想：**每次更新别把策略改得太猛**，用"剪切（clip）"机制把策略更新幅度软约束在一个信任区域内，从而避免一步更新就把策略推崩。它是 TRPO（用硬性 KL 约束 + 二阶优化）的"一阶简化版"——把 TRPO 的约束放进目标函数里，用普通 SGD 就能跑。

### 1.2 完整组件：Actor-Critic
PPO 是 actor-critic 架构，训练时同时维护：

- **Actor / Policy** $\pi_\theta$：要优化的策略（语言模型）。
- **Critic / Value** $V_\psi(s)$：估计状态价值，用来算优势函数，降低梯度方差。
- （RLHF 额外）**Reward Model** $r_\phi$ 和 **Reference Model** $\pi_{\text{ref}}$。

所以经典 RLHF-PPO 训练时显存里同时有 **4 个模型**（policy、value、reward、reference），这是它最被诟病的成本来源。

### 1.3 优势估计：GAE
PPO 通常配合 **GAE（Generalized Advantage Estimation）** 估计优势，平衡偏差和方差：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\gamma$：折扣因子；$\lambda$：GAE 的偏差-方差权衡系数。
- $\delta_t$ 是单步 TD 误差；GAE 把多步 TD 误差做指数加权平均。

### 1.4 在 RLHF 中的位置
传统 RLHF 流程：
1. 用偏好数据训练一个 **Reward Model** $r_\phi(x,y)$，给整段回答打一个标量分。
2. 用 PPO 优化语言模型 $\pi_\theta$，让它在 KL 约束下最大化奖励：

$$
\max_\theta \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)} \left[ r_\phi(x,y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

实际实现里，KL 惩罚通常**逐 token 拆进奖励**：序列最后一个 token 给 $r_\phi(x,y)$，每个 token 再减去 $\beta\big(\log\pi_\theta - \log\pi_{\text{ref}}\big)$，组成每步奖励 $r_t$，再喂给 GAE。$\beta$ 控制新策略不要离 SFT 模型 $\pi_{\text{ref}}$ 太远，避免 reward hacking / 胡言乱语。

### 1.5 PPO-Clip 目标函数
定义概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

对语言模型来说，$s_t$ 是前缀，$a_t$ 是下一个 token。剪切代理目标：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]
$$

完整 PPO 目标还包括**价值函数损失**和**熵正则**：

$$
L^{\text{PPO}} = \mathbb{E}_t\left[ L^{\text{CLIP}}_t - c_1\, (V_\psi(s_t)-V_t^{\text{target}})^2 + c_2\, \mathcal{H}[\pi_\theta(\cdot|s_t)] \right]
$$

- $\hat{A}_t$：优势函数，表示这个动作比平均水平好多少。
- $\epsilon$：剪切阈值，常见 0.1 或 0.2。
- `min` + `clip`：当优势为正时，比值被截在 $1+\epsilon$，防止过度增大；优势为负时截在 $1-\epsilon$，防止过度压制——本质是一个**悲观（取 min）**的下界目标。
- $c_1$ 价值损失系数、$c_2$ 熵系数（鼓励探索）。

### 1.5.1 拆解

**先看概率比 $r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$。**
- $s_t$（state）= 已经写出的前缀，$a_t$（action）= 接下来要写的这个 token。
- 分子是**现在的模型**生成这个 token 的概率，分母是**采样时那份旧模型**的概率。
- 比值 >1 表示"更新后模型比原来更想说这个词"，<1 表示"更不想说"。它衡量**这一步更新把概率改动了多少**。

**再看 $r_t(\theta)\cdot\hat A_t$。**
- $\hat A_t$（优势）是"这个 token 事后看好不好"：正数=这步走对了，负数=走错了。
- 两者相乘就是朴素策略梯度的目标：**好的动作（$\hat A>0$）就提高它的概率，坏的动作就降低**。

**关键是那个 `clip`（剪切）。** 只用 $r_t\hat A_t$ 有个危险：如果某一步优势很大，模型会疯狂拉高这个词的概率，一步就把自己"练崩"。clip 给比值套了个笼子 $[1-\epsilon,\ 1+\epsilon]$（比如 $\epsilon=0.2$ 就是 $[0.8,1.2]$）：
- 好动作最多把概率放大到 1.2 倍就"封顶"，不许再贪；
- 坏动作最多压到 0.8 倍就"触底"，不许一次压太狠。

外面再套一个 `min`（取两者较小值），保证这是个**保守的下界**——宁可少更新，也不冒险过度更新。

> **一句话生活类比**：PPO 像给学生纠错的老师，"这次答得好，鼓励你，但一次别得意忘形（涨幅封顶）；这次答错了，批评你，但一次也别否定太狠（跌幅设底）"。小步慢走，稳。

后两项 $-c_1(V-V^{\text{target}})^2$ 是训练那个估算优势的 Critic（让它预测得更准），$+c_2\mathcal H$ 是熵奖励（鼓励模型别太快变得死板、保留一点探索性）。

### 1.6 为什么需要 PPO？
朴素策略梯度里，一次更新步子太大，策略可能崩溃到只输出高奖励的"作弊"文本。TRPO 用硬 KL 约束解决但要算二阶；PPO 的 clip 机制相当于用一阶方法给更新加了一条"安全带"。

### 1.7 优缺点

| 优点 | 缺点 |
|---|---|
| 经典、稳定、通用性强 | 需要额外训练 Critic/Value 模型，4 模型同驻显存 |
| Clip 机制防止策略崩溃 | 超参数多（$\epsilon, \beta, \gamma, \lambda, c_1, c_2, \text{lr}, \dots$） |
| 支持在线采样、探索，能突破 SFT 数据上限 | 工程实现敏感，常需大量 trick（reward 归一化、advantage 白化等） |

**论文**：*Proximal Policy Optimization Algorithms* — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。RLHF 落地参考 InstructGPT（*Training language models to follow instructions with human feedback*, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)）。

## 2. DPO（Direct Preference Optimization）

### 2.1 一句话直觉
DPO 发现：**显式 Reward Model 是多余的**。在 Bradley-Terry 偏好模型 + KL 约束 RL 的设定下，最优策略与奖励之间存在解析关系，可以把奖励反解出来用策略表示，代回偏好似然，最终得到一个**只依赖策略模型和参考模型的监督式分类 loss**——你的语言模型本身就"暗含"了一个奖励模型（论文副标题：*Your Language Model is Secretly a Reward Model*）。

### 2.2 核心推导（完整版）
**第 1 步**：KL 约束 RL 目标的最优策略有闭式解：

$$
\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right),\qquad Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp\!\Big(\tfrac1\beta r(x,y)\Big)
$$

其中 $Z(x)$ 是配分函数（partition function），对所有可能回答求和，**无法直接计算**——这正是过去必须走 RL 的原因。

**第 2 步**：把奖励反解：

$$
r(x,y) = \beta \log\frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**第 3 步**：代入 Bradley-Terry。BT 模型里偏好概率只依赖**奖励之差**：

$$
P(y_w \succ y_l|x) = \sigma\big( r(x,y_w) - r(x,y_l) \big)
$$

做差时 $\beta\log Z(x)$ 这一项**完全抵消**（它只依赖 $x$，不依赖 $y$）。于是得到 **DPO loss**：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

这就是一个对偏好对的二元交叉熵，**不需要采样、不需要单独的 reward model**。

### 2.2.1 DPO loss

把最终那个 loss 拆开看，它其实只有三层，从里到外：

**第 1 层：隐式奖励。** $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ 就是 §0.3 讲的隐式奖励——"**训练中的模型相比参考模型，多大程度上更愿意生成这个回答**"。对好回答 $y_w$ 和坏回答 $y_l$ 各算一个。

**第 2 层：奖励之差。** 两个隐式奖励相减：
$$
\Delta = \underbrace{\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{好回答的隐式奖励}} - \underbrace{\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{坏回答的隐式奖励}}
$$
$\Delta$ 越大，说明模型越"拎得清"——把好的抬得比坏的高。我们希望 $\Delta$ 越大越好。

**第 3 层：$-\log\sigma(\Delta)$。** $\sigma$ 把 $\Delta$ 变成一个 $(0,1)$ 的"排序正确的概率"，取 $-\log$ 就是标准的分类损失：
- $\Delta$ 很大（排序很对）→ $\sigma(\Delta)\to1$ → $-\log\sigma(\Delta)\to0$，loss 几乎为 0，不用再学；
- $\Delta$ 是负的（把好坏排反了）→ loss 很大，狠狠惩罚，逼模型改。

> **一句话生活类比**：DPO 就是在做一道道"二选一"的判断题——每道题给模型一个好答案和一个坏答案，训练它"把好的概率提上去、坏的压下去"，仅此而已。所以它写起来跟普通监督分类几乎一样，稳定又简单。

**为什么要除以 $\pi_{\text{ref}}$（而不直接用 $\pi_\theta$）？** 因为 $\pi_{\text{ref}}$ 是锚：只有相对参考模型的"变化量"才算数。这样模型不能靠"把所有回答概率一起抬高"作弊，只能真正拉开好坏之间的**相对**差距。

### 2.3 梯度解释（为什么它有效）
DPO 的梯度是：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{权重：隐式奖励排错得越离谱越大}}\big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\big)\Big]
$$

其中 $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是**隐式奖励**。直观：
- 提高 $y_w$ 概率、压低 $y_l$ 概率；
- 前面的 sigmoid 权重在"模型当前把好坏排反了"时最大，已经排对的样本权重小——这是一个自带难度加权的机制，避免在已学会的样本上过度更新。

### 2.4 理论结论与一般化
- **Reward 等价类**：相差一个只依赖 $x$ 的函数 $f(x)$ 的两个奖励，诱导相同的偏好分布和相同的最优策略（Lemma 1/2）。
- **Theorem 1**：任何与 Plackett-Luce（含 BT 为 $K{=}2$ 特例）一致的奖励类，都可以用 $r(x,y)=\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 重参数化表示——所以"用策略表示奖励"不丢一般性。
- **Plackett-Luce 扩展**：若数据是 $K$ 个回答的排序而非单对，DPO 可推广到 PL 损失，配分函数同样抵消。

### 2.5 主要实验结果（原文）
- **IMDb 情感可控生成**：在 reward-KL 前沿上，DPO 在所有 KL 取值下都拿到最高期望奖励，甚至超过能拿到真值奖励的 PPO-GT。
- **TL;DR 摘要（GPT-J）**：DPO 对人类参考摘要胜率 **61%**（temp=0.0），PPO 最优温度下 **57%**；DPO 对采样温度更鲁棒。
- **Anthropic-HH 对话（Pythia-2.8B）**：DPO 是唯一"计算高效且能超过数据集中 chosen 回答"的方法，性能逼近 Best-of-128 这种采样密集的 oracle。

### 2.6 优缺点

| 优点 | 缺点 |
|---|---|
| 无需 Reward Model 与在线采样，Pipeline 极简 | 本质离线，无法在线探索（off-policy，受限于已有偏好分布） |
| 训练稳定，像监督学习；只需 policy + reference | 对偏好数据质量/标注噪声敏感 |
| 效果常与 PPO-based RLHF 相当甚至更好 | 容易出现**长度偏见**（偏好更长回答）；偏好确定性强时易过拟合（→ IPO 动机） |

**论文**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn, NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

## 3. GRPO（Group Relative Policy Optimization）

### 3.1 一句话直觉
GRPO 是 PPO 的"轻量版"：它**砍掉 Critic/Value 模型**，改为对同一个问题采样**一组**回答，用组内奖励的相对高低（均值/标准差归一化）来当优势，省掉了一整个价值网络。出自 DeepSeekMath，后来是 DeepSeek-R1 的核心算法。

### 3.2 为什么出现？
在数学推理等可验证任务中：
- 答案对错非常明确（规则奖励，通常 0/1）。
- 训练 Critic 很占显存，而且让 Critic 准确评估"长推理链每一步的价值"本身就很难（价值估计在 token 级别非常稀疏）。
- 所以 DeepSeekMath 干脆去掉 Critic，用"同一问题的一组样本"互相做 baseline。

### 3.3 算法流程
对问题 $q$：
1. 用旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个回答 $\{o_1,\dots,o_G\}$。
2. 对每个回答算奖励 $R_i$。
3. **结果监督（outcome supervision）**：整条回答所有 token 共享同一个归一化优势：

$$
\hat{A}_{i,t} = \tilde R_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

4. **过程监督（process supervision）**：若有步骤级奖励，某 token 的优势是它**之后所有步骤**归一化奖励之和：$\hat A_{i,t}=\sum_{\text{index}(j)\ge t}\tilde R_i^{\text{index}(j)}$。
5. 用 PPO-Clip 思想更新：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}} \left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \Big( \min\big( r_{i,t}\hat{A}_{i,t},\ \text{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t} \big) - \beta\, D_{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}] \Big) \right]
$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$。

### 3.3.1 拆解

**先理解符号。** 对一道题 $q$，让模型一次生成 $G$ 个不同答案 $o_1,\dots,o_G$（比如 $G=8$）。$o_i$ 是第 $i$ 个答案，$o_{i,t}$ 是它的第 $t$ 个 token，$|o_i|$ 是它的长度，$R_i$ 是它拿到的奖励（数学题就是对=1、错=0）。

**核心就一个公式——组内归一化优势：**
$$
\hat A_{i,t}=\frac{R_i-\text{mean}(R_1,\dots,R_G)}{\text{std}(R_1,\dots,R_G)}
$$
- 分子 $R_i-\text{mean}$：这个答案比"这一组的平均分"高还是低。高于平均→正优势（鼓励），低于平均→负优势（打压）。
- 分母 $\text{std}$（标准差）：按这组分数的波动幅度做缩放，让不同难度的题梯度尺度差不多。
- 同一个答案里**每个 token 共享这一个优势值**（结果监督）。

> **一句话生活类比**：PPO 要专门养一个"裁判"（Critic）去给每一步打分，很贵；GRPO 干脆让**同一道题的 8 份答卷互相比**——比全班平均好的答卷整体加分，差的整体扣分。"用同组同学当参照物"，省掉了裁判。

**外层那一大串** $\frac1G\sum_i\frac1{|o_i|}\sum_t\min(\dots)$ 只是在说：对 $G$ 个答案、每个答案的每个 token，都套用和 PPO 一样的 clip 目标（防止一步更新过猛），再平均。最后 $-\beta D_{KL}$ 是"别离参考模型太远"的缰绳。所以 GRPO ≈ **PPO 的 clip 机制 + 用组内相对分代替 Critic 优势**。

### 3.4 关键细节
- **KL 项用 k3 无偏估计器**（而非朴素 log 比值），保证非负且方差更小：

$$
D_{KL}[\pi_\theta\|\pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)} - \log\frac{\pi_{\text{ref}}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)} - 1
$$

  注意 KL 是**直接作为一项加在 loss 里**，而不像 PPO 那样揉进每步奖励。
- **Critic-free**：没有价值网络，显存与实现都简化。
- **Group-relative baseline**：用同组样本的均值/方差把绝对奖励变相对，等价于一个无 Critic 的优势估计。
- **迭代 GRPO**：用策略的新采样不断重训 reward model（带 10% 历史回放），再继续训练策略。
- **统一视角**：DeepSeekMath 提出一个统一范式 $\nabla_\theta\mathcal{J}=\mathbb{E}[\,GC_A(q,o,t)\,\nabla_\theta\log\pi_\theta]$，把 SFT / RFT / DPO / PPO / GRPO 都看成"数据源 × 奖励函数 × 梯度系数"三要素的不同组合。

### 3.5 后续变体（重要补充）
原版 GRPO 的两处归一化被后续工作指出有偏：

- **Dr. GRPO**（*Understanding R1-Zero-Like Training*, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)）指出两个偏差：
  1. **长度归一化偏差** $\frac{1}{|o_i|}$：正优势时短回答得到更大梯度（偏好简短正确答案），负优势时长回答被罚得更轻（偏好冗长的错误答案）→ 错误回答越练越长（"overthinking"）。
  2. **难度归一化偏差**（除以 std）：标准差很小的题（太简单或太难、奖励几乎全 1 或全 0）会被赋予过高权重。
  Dr. GRPO **直接去掉 $\frac{1}{|o_i|}$ 和 std 归一化**，回归无偏的 MC 优势，显著抑制了错误回答长度的膨胀。
- **DAPO**（[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)）：解耦上下 clip 阈值（clip-higher）、动态采样（过滤全对/全错的组）、token 级梯度、超长惩罚等工程改进。

> 实践提醒：说"GRPO 需要 reference model"只对**原版**成立；不少变体（如某些 R1-Zero 配方）直接**去掉 KL 项**以追求更强探索。

### 3.6 优缺点

| 优点 | 缺点 |
|---|---|
| 省内存（无 Critic），实现简洁 | 原版 std/长度归一化有偏（→ Dr.GRPO 修正） |
| 对数学/代码等**可验证**任务特别有效 | 奖励稀疏或主观时组内 baseline 不稳定 |
| 被 DeepSeek-R1 等大规模 RLVR 采用 | 需要好的规则奖励/可判定环境；开放域问答收益有限 |

**论文**：*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — Zhihong Shao, Peiyi Wang, Qihao Zhu, et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

## 4. 另一条路线：只给二元反馈（Binary Feedback）

DPO 需要成对偏好数据 `(x, y_w, y_l)`。但实际场景里很多反馈是二元的：用户只给单个回答点赞或踩，根本没有"配对的另一个回答"。下面这些方法致力于**只用单个回答的好坏信号**做对齐。这类数据更便宜、更贴近真实产品日志。

## 5. KTO（Kahneman-Tversky Optimization）

### 5.1 一句话直觉
KTO 把人当成"有损失厌恶的决策者"：赚到好处和遭受损失的**心理权重不对称**。它不需要成对偏好，只需要知道单个回答"好"还是"坏"。它直接最大化生成的"人类效用（utility）"，而不是最大化偏好似然。

### 5.2 背景：前景理论与 HALO 框架
Kahneman & Tversky 的前景理论指出：人对损失比对等量收益更敏感（损失厌恶），且评价结果时存在一个"参照点"。原始价值函数（论文 Eq.4）：

$$
v(z;\lambda,\alpha,z_0)=\begin{cases}(z-z_0)^\alpha & z\ge z_0\\ -\lambda(z_0-z)^\alpha & z< z_0\end{cases}
$$

经验上 $\alpha\approx 0.88$（增益区凹、风险厌恶）、$\lambda\approx 2.25$（损失放大 2 倍多）。

KTO 论文提出更一般的 **HALO（Human-Aware Loss）** 框架：一个损失若能写成

$$
f(\pi_\theta,\pi_{\text{ref}})=\mathbb{E}\big[a_{x,y}\, v\big(r_\theta(x,y)-\mathbb{E}_Q[r_\theta(x,y')]\big)\big]+C_\mathcal{D}
$$

（$r_\theta=l(y)\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，$a_{x,y}\in\{-1,+1\}$，$v$ 非减且在正区间凹），就是一个 HALO。论文证明 **DPO 和 PPO-Clip 都是 HALO**——它们之所以好用，部分是因为隐式地建模了人类决策偏差。

### 5.3 KTO 损失函数
隐式奖励：$r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。参照点 $z_0$ 取 policy 与 reference 之间的 KL：

$$
z_0 = \text{KL}\big(\pi_\theta(y'|x)\,\|\,\pi_{\text{ref}}(y'|x)\big)
$$

价值函数（论文用 logistic 替代幂律以避免数值不稳）：

$$
v(x,y)=\begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta(x,y)-z_0)\big) & y\ \text{desirable}\\[4pt]\lambda_U\,\sigma\big(\beta(z_0-r_\theta(x,y))\big) & y\ \text{undesirable}\end{cases}
$$

损失 $\mathcal{L}_{\text{KTO}}=\mathbb{E}[\lambda_y - v(x,y)]$。

### 5.3.1 拆解

KTO 和 DPO 的最大区别：**DPO 要成对的（好+坏一起给），KTO 只要单条**——这条回答要么被标"好（desirable）"，要么标"坏（undesirable）"。

**符号：**
- $r_\theta(x,y)=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$：还是那个隐式奖励（模型比参考模型多爱这条回答）。
- $z_0$：**参照点**，取当前模型和参考模型的平均 KL 距离，代表"一个普通回答大概值多少分"。好坏都是相对这个基准线来判断的。
- $\lambda_D,\lambda_U$：好样本（Desirable）和坏样本（Undesirable）的权重，用来应对"赞和踩数量不平衡"。
- $\sigma$：还是那个把分数压到 $(0,1)$ 的 S 形函数。

**价值函数在干嘛：**
$$
v=\begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta-z_0)\big) & \text{好回答：奖励越超过基准线 }z_0\text{，}v\text{ 越大（趋近 }\lambda_D\text{）}\\[4pt]\lambda_U\,\sigma\big(\beta(z_0-r_\theta)\big) & \text{坏回答：奖励越低于基准线，}v\text{ 越大（趋近 }\lambda_U\text{）}\end{cases}
$$
$v$ 是"这条数据当前被处理得多好"。loss $=\lambda_y-v$，要最小化 loss 就是要**最大化 $v$**：即让好回答的隐式奖励爬到基准线以上、坏回答的掉到基准线以下。

> **一句话生活类比**：KTO 借用了行为经济学的"损失厌恶"——人对亏钱的痛感强过对赚同样多钱的快感。所以它给"坏样本被压下去"的权重可以调得更重（$\lambda_U$），像人一样"厌恶损失"。好处是：产品里的点赞/点踩日志（没有配对）可以直接拿来训练。

### 5.4 工程关键点
- **$z_0$ 的 in-batch 估计**：从 $\pi_\theta$ 采样太贵，KTO 在 microbatch 内**错位配对**（把第 $i$ 个输入配第 $(i{+}1)\bmod m$ 个输出）估计，并 clamp 到 $\ge 0$：$\hat z_0=\max\big(0,\frac1m\sum\log\frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\big)$。**不反传** $z_0$，它只用来控制 loss 饱和。
- **不平衡数据**：用 $\lambda_D,\lambda_U$ 处理类别不平衡，推荐 $\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,\tfrac32]$。例：赞:踩 = 1:10 时设 $\lambda_U{=}1,\lambda_D\in[10,15]$。
- **学习率**：KTO 的最优 lr 通常是 DPO 的 **2–10 倍**（DPO 常用 5e-7，KTO 可用 5e-6）。

### 5.5 实验结果（原文）
- **规模对比（GPT-4 评 vs SFT target）**：Llama-{7B,13B,30B} 上"KTO alone"普遍优于"DPO alone"，7B 与 30B 差距显著（p<0.01）。
- **Zephyr-β-SFT + UltraFeedback（Table 2）**：换 DPO 为 KTO，**GSM8K 39.0→53.5**（+13.5），BBH 44.1→52.6，MMLU 58.2→58.6。
- **数据效率**：每个 prompt 只留一个回答（数据砍 72%）时，KTO（63.1%）仍超过用全部数据的 DPO（60.0%）；甚至可丢弃多达 90% 的 desirable 数据仍匹配 DPO。
- **鲁棒性**：跳过 SFT 直接对齐时，DPO 会"啰嗦"（回答暴涨变长），KTO 长度基本不变。

### 5.6 优缺点

| 优点 | 缺点 |
|---|---|
| 不需要成对偏好，数据更易得 | 需要调 $\lambda_D,\lambda_U,\beta,z_0$ |
| 能处理只有"好"或只有"坏"、严重不平衡的数据 | $z_0$ 的 in-batch 估计有偏 |
| 理论基础来自行为经济学（HALO） | 在干净的配对数据上有时不如 DPO |

**论文**：*KTO: Model Alignment as Prospect Theoretic Optimization* — Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela, 2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)

## 6. BCO（Binary Classifier Optimization）

### 6.1 一句话直觉
BCO 说：既然只有二元反馈，那我干脆**训练一个二分类器**，把分类器的 logit 当作隐式奖励来做对齐。它证明了：**二分类器的 BCE 损失是 DPO 损失的上界**，所以最小化 BCE 间接最小化 DPO。

### 6.2 核心思想与理论
数据形式 $(x,y,c)$，$c\in\{0,1\}$。隐式奖励 $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$（同 DPO/KTO），用 BCE：

$$
\mathcal{L}_{\text{BCE}} = -\mathbb{E}_{(x,y,c)} \big[ c \log \sigma(r_\theta(x,y)) + (1-c)\log\big(1-\sigma(r_\theta(x,y))\big) \big]
$$

**Theorem 1**：BCE 损失上界 DPO 损失。证明的关键是 **Lemma**：对任意实数 $x,y$，$\log\sigma(x+y) > \log\sigma(x)+\log\sigma(y)$。直觉上：DPO 对"赢-输差"做一次 sigmoid，BCE 对赢、输分别做 sigmoid 再相加，后者把前者夹在上面。

### 6.2.1 拆解

**符号：** 数据是 $(x,y,c)$，$c=1$ 表示这条回答被标"好"，$c=0$ 表示"坏"。$r_\theta$ 还是隐式奖励。

**BCE（二元交叉熵）就是最标准的"是/否"分类损失：**
$$
\mathcal L_{\text{BCE}}=-\mathbb E\big[\underbrace{c\log\sigma(r_\theta)}_{c=1\text{时生效}}+\underbrace{(1-c)\log(1-\sigma(r_\theta))}_{c=0\text{时生效}}\big]
$$
- 当 $c=1$（好）：只剩 $-\log\sigma(r_\theta)$，要 loss 小就得 $\sigma(r_\theta)\to1$，即**把隐式奖励 $r_\theta$ 推高**。
- 当 $c=0$（坏）：只剩 $-\log(1-\sigma(r_\theta))$，要 loss 小就得 $\sigma(r_\theta)\to0$，即**把 $r_\theta$ 压低**。

> **一句话生活类比**：KTO 借行为经济学、要调 $\lambda$、参照点等一堆东西；BCO 说"别想那么复杂——就训练一个判断'这条回答是好是坏'的二分类器，把它的打分（logit）当奖励用"。而且论文证明了：**最小化这个简单的 BCE，其实顺带就把 DPO 的目标也压下去了**（BCE 是 DPO 的上界，上界降了下面也跟着降）。后面 §6.3 的 reward shift、§6.4 的 UDM 只是两个修正小组件，让这个分类器的判分更准。

### 6.3 Reward Shift（缩小与 DPO 的差距）
BCE 的隐式奖励整体可能有个偏移，BCO 用一个平移 $\delta$ 校准，让"赞"奖励为正、"踩"为负的分界更准：

$$
\tilde r_\theta(x,y)=r_\theta(x,y)-\delta,\qquad \delta=\frac{\mathbb{E}_{\mathcal D^+}[r_\theta]+\mathbb{E}_{\mathcal D^-}[r_\theta]}{2}
$$

即取赞集 $\mathcal D^+$ 与踩集 $\mathcal D^-$ 的平均隐式奖励的中点，实现上用滑动平均（EMA）平滑。

### 6.4 Underlying Distribution Matching（UDM）
真实数据里"赞"和"踩"常来自**不同的 prompt 分布**（比如用户只在某类问题上点赞、另一类上点踩），reward shift 假设两者同分布会失效。UDM 用**重要性采样 + 密度比技巧（density ratio trick）**估计两个分布的比值，重新加权，使训练"仿佛"赞/踩来自同一底层分布。

> 论文报告：在对 gpt-3.5-turbo 的胜率上，单纯 BCE 为 **12.18%**，加上 reward shift + UDM 的完整 BCO 提升到 **16.16%**——两个组件协同才有效。

### 6.5 实验结果（原文，节选）
HelpSteer2 上把 Likert-5 评分转成二元信号训练：

| Model | Method | MT-Bench | AlpacaEval 2.0 LC | Arena-Hard |
|---|---|---|---|---|
| Llama-3.1-8B | DPO | 8.17 | 25.99 | 29.50 |
| Llama-3.1-8B | **BCO** | **8.32** | **28.61** | **31.37** |
| Qwen2.5-7B | DPO | 8.40 | 30.95 | 47.82 |
| Qwen2.5-7B | **BCO** | **8.59** | 30.54 | **50.60** |

注意：BCO 只用**二元**信号就达到甚至超过用成对偏好的 DPO。

### 6.6 优缺点

| 优点 | 缺点 |
|---|---|
| 只用二元反馈，数据成本低 | 概念上多了"分类器视角"，需理解 reward shift / UDM |
| 在真实 Likert / 点赞点踩数据上表现好 | 相比 KTO 多了组件，实现稍复杂 |
| 与 DPO 有清晰理论联系（BCE 上界 DPO） | 仍需参考模型 |

**论文**：*Binary Classifier Optimization for Large Language Model Alignment* — Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, Kyoung-Woon On, **ACL 2025 (main)**. [arXiv:2404.04656](https://arxiv.org/abs/2404.04656)

## 7. Step-KTO：把 KTO 做到推理步骤级别

### 7.1 一句话直觉
Step-KTO 把 KTO 从"整条回答的好坏"细化到**中间推理步骤的好坏**，同时仍利用最终答案对错作为结果监督，两级二元反馈联合优化，特别适合数学推理。

### 7.2 动机
- 只看最终答案对错（outcome-level）太粗：答案错了不知道错在哪一步。
- 给每一步人工标注正确性（process-level）成本太高。
- Step-KTO 用**自动生成的步骤级二元反馈** + 最终答案反馈共同训练。

### 7.3 训练框架（迭代式）
1. **采样**：对每题用当前模型 $M_t$ 采 8 个候选解（temperature $T=0.7$）。
2. **结果评价**：用正则匹配判定最终答案对错（outcome 二元标签）。
3. **过程评价**：用一个 **PRM（过程奖励模型）**——实际是 prompt 一个 **Llama-3.1-70B-Instruct** 按模板对每一步给 $\{0,1\}$ 二元判断。
4. **联合损失**：结果级 KTO + 步骤级 KTO 相加：

$$
\mathcal{L}_{\text{Step-KTO}}=\mathcal{L}_{\text{KTO}}^{\text{(outcome)}}+\mathcal{L}_{\text{step}},\qquad \mathcal{L}_{\text{step}}=\mathbb{E}_{x,y,s_h}\big[\lambda^{\text{step}}_y - v^{\text{step}}(x,s_h)\big]
$$

  其中 $v^{\text{step}}$ 与 KTO 同构：步骤判对用 $\lambda_D^{\text{step}}\sigma(\beta_{\text{step}}(r_\theta(x,s_h)-z_0^{\text{step}}))$，判错用 $\lambda_U^{\text{step}}\sigma(\beta_{\text{step}}(z_0^{\text{step}}-r_\theta(x,s_h)))$。
5. **迭代**：训练得到 $M_{t+1}$，重复 $T$ 轮（$M_0\to M_1\to\dots$）。

这样模型不仅学"答案要对"，还学"每一步都要合理"。

### 7.3.1 拆解

**符号：** $s_h$ 是回答里的第 $h$ 个推理**步骤**（比如解题的一行），$M_t$ 是第 $t$ 轮的模型。损失就是把两个 KTO 相加：
$$
\mathcal L_{\text{Step-KTO}}=\underbrace{\mathcal L_{\text{KTO}}^{\text{(outcome)}}}_{\text{看最终答案对不对}}+\underbrace{\mathcal L_{\text{step}}}_{\text{看每一步对不对}}
$$
$\mathcal L_{\text{step}}$ 的形式和 §5.3 的 KTO **一模一样**，只是把"整条回答好/坏"换成"这一步好/坏"（用一个大模型当批改老师 PRM 给每步打 0/1）。

> **一句话生活类比**：普通 KTO 只看学生的最终答案对不对；Step-KTO 像老师批改草稿——**不仅看答案，还逐行检查过程**，哪一步写错了单独扣分。所以特别适合数学题这种"答案对但过程可能蒙对、或过程错但答案侥幸对"的场景。

### 7.4 实验结果（原文，已核实）

| Base model | 数据集 | Seed $M_0$ | KTO $M_3$ | **Step-KTO** |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | MATH-500 Pass@1 | 53.4% | 60.6% | **63.2%** |
| Llama-3.1-8B-Instruct | AMC23 Pass@1 | 35.0% | 35.0% | **47.5%** |
| Llama-3.1-8B-Instruct | AIME24 Pass@1 | — | 16.7% | 16.7% |
| Llama-3.1-70B-Instruct | MATH-500 Pass@1 | 74.6% | — | **76.2%** ($M_1$) |
| Llama-3.3-70B-Instruct | MATH-500 Pass@1 | 75.8% | — | **79.6%** ($M_1$) |

要点：MATH-500 上 53.4%→63.2%（+9.8），且**比同样迭代的纯 KTO（60.6%）还高出 2.6 个点**，说明步骤级信号确有额外贡献；AMC23 上提升尤其明显（35.0→47.5）。原文未报告 GSM8K。

**论文**：*Step-KTO: Optimizing Mathematical Reasoning through Stepwise Binary Feedback* — Yen-Ting Lin, Di Jin, Tengyu Xu, Tianhao Wu, Sainbayar Sukhbaatar, Chen Zhu, Yun He, Yun-Nung Chen, Jason Weston, Yuandong Tian, Arash Rahnama, Sinong Wang, Hao Ma, Han Fang, 2025. [arXiv:2501.10799](https://arxiv.org/abs/2501.10799)

## 9. TR-KTO（Trust Region KTO，属 TR-* 系列）

### 9.1 一句话直觉
TR 系列认为：训练时一直把参考模型**冻结**在 SFT 模型上，会限制策略的优化空间。它提出在训练过程中**动态更新参考模型**，同时用信任区域思想约束策略不要一步跑太远。注意：**没有单独一篇"TR-KTO"论文**，TR-KTO 是论文 *Learn Your Reference Model for Real Good Alignment* 里 **TR-DPO / TR-IPO / TR-KTO** 三个变体之一。

### 9.2 核心机制
所有 TR-* 变体共用同一套"动态参考模型"更新，只是套在不同的基础损失（DPO / IPO / KTO）上：

- **软更新（soft update）**：$\pi_{\text{ref}}\leftarrow \alpha\,\pi_\theta+(1-\alpha)\,\pi_{\text{ref}}^{\text{prev}}$，$\alpha\in[0,1]$（对权重做滑动平均）。
- **硬更新（hard update）**：每隔 $\tau$ 步令 $\pi_{\text{ref}}\leftarrow\pi_\theta$。

这样 $\pi_{\text{ref}}$ 始终代表"当前认为不错的模型"，而不是远古的 SFT；配合信任区域，缓解过度优化（overoptimization）和对偏好数据表面模式的过拟合，让策略能离初始 SFT 更远、探索更优解。

### 9.2.1 拆解

**问题背景：** DPO/KTO 里那个参考模型 $\pi_{\text{ref}}$ 全程冻结在最初的 SFT 模型上。KL 约束逼着策略别离它太远——可 SFT 模型本身没那么强，把策略死死拴在它旁边，就限制了能进步的上限。

**TR 的解法——让参考模型也慢慢跟上来：**
- **软更新** $\pi_{\text{ref}}\leftarrow\alpha\,\pi_\theta+(1-\alpha)\,\pi_{\text{ref}}$：把当前策略的权重按比例 $\alpha$ 混一点进参考模型（$\alpha$ 小=慢慢挪）。
- **硬更新**：每隔 $\tau$ 步，直接把参考模型替换成当前策略。

> **一句话生活类比**：原来的锚点是"你入学时的水平"，你永远只能跟入学的自己比、不许差太远。TR 说"锚点应该是**你上个月的水平**"——参考线随着你进步而缓慢上移，于是你能不断超越过去的自己、走得更远，但每次又不会一步迈太大（信任区域）。TR-DPO / TR-IPO / TR-KTO 就是把这套"动态锚点"分别装到 DPO / IPO / KTO 上。

### 9.3 实验结果（原文，已核实）
GPT-4 评判、500 样本：

- **Reddit TL;DR（Pythia-2.8B，$\alpha=0.6$）**：TR-KTO 46.2% win / 11.6% tie / 42.2% loss。
- **Anthropic-HH（Pythia-2.8B，$\alpha=0.6$）**：TR-KTO 37.4% win / 28.2% tie / 34.4% loss。
- **摘要任务、6.9B 规模**：相对各自固定参考基线的胜率增益约为 **DPO +8.4%、IPO +14.3%、KTO +15%**——这正是原报告"TR-KTO 比标准 KTO 胜率提升约 15%"的出处（指 Reddit TL;DR / 6.9B 设定）。
- 论文还在 Llama3 上用 AlpacaEval 2 / Arena-Hard 验证。

> 修正：原报告把 TR-KTO 当成独立方法单列。准确说法是 **TR-\* 系列**（一篇论文，三变体）。

**论文**：*Learn Your Reference Model for Real Good Alignment* — Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, Daniil Gavrilov, 2024. [arXiv:2404.09656](https://arxiv.org/abs/2404.09656)

## 10. IPO（Identity Preference Optimization）

### 10.1 一句话直觉
IPO 指出 DPO 有个理论缺陷：当偏好**几乎确定**（经验上 $\hat p(y_w\succ y_l)=1$）时，DPO 会让 KL 正则项形同虚设，把策略推成确定性、过拟合到偏好数据。IPO 从更一般的 **ΨPO** 框架出发，把 Ψ 取成**恒等映射（Identity）**，把 KL 正则更紧地锁进目标，得到一个 MSE 形式的稳健损失。

### 10.2 ΨPO 一般框架
论文提出统一目标（Eq.6）：

$$
\max_\pi\ \mathbb{E}\big[\Psi\big(p^*(y\succ y'|x)\big)\big]-\tau\, D_{KL}(\pi\|\pi_{\text{ref}})
$$

$\Psi$ 是把"偏好概率"映射到收益的非减函数，$\tau$ 控制正则强度。**DPO 对应 $\Psi(q)=\log\frac{q}{1-q}$（logistic）**——这是个**无界**函数。

### 10.3 DPO 为什么会过拟合
当某对偏好在数据里被估成确定（$\hat p=1$），BT 要求奖励差 $\to\infty$；由于 $\Psi=\log\frac{q}{1-q}$ 无界，最优策略 $\pi^*\propto\pi_{\text{ref}}\exp(\tau^{-1}\cdot\infty)$ 会把 $y_l$ 概率压到 0，**无论 $\tau$ 多大都压不住**——KL 正则被无界的 Ψ 淹没。少量数据下尤其危险（真值哪怕是 0.8，小样本也容易估成 1）。

### 10.4 IPO 损失（Ψ = Identity）
取 $\Psi(q)=q$，论文导出（Eq.17）：

$$
\mathcal{L}_{\text{IPO}}=\mathbb{E}_{(x,y_w,y_l)}\left[\left(h_\pi(y_w,y_l)-\frac{1}{2\tau}\right)^2\right],\quad h_\pi(y,y')=\log\frac{\pi(y|x)\,\pi_{\text{ref}}(y'|x)}{\pi(y'|x)\,\pi_{\text{ref}}(y|x)}
$$

即让"赢-输的隐式奖励差" $h_\pi$ 稳定地**回归到一个有限目标值 $\frac{1}{2\tau}$**，而不是无限增大。$\tau$ 越大（正则越强），目标值越小，策略越贴近 $\pi_{\text{ref}}$。

### 10.4.1 拆解

**先看 $h_\pi$ 是什么。** 展开来看：
$$
h_\pi(y_w,y_l)=\underbrace{\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{好回答的隐式奖励(去掉}\beta\text{)}}-\underbrace{\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{坏回答的隐式奖励}}
$$
它跟 DPO 里那个"奖励之差 $\Delta$"是同一个东西——衡量模型把好坏拉开了多大差距。

**IPO 和 DPO 的唯一区别，在于拿这个差去做什么：**
- DPO：$-\log\sigma(\beta\cdot h_\pi)$，这是分类损失，**差距越大越好、永远鼓励拉得更开**，没有上限。
- IPO：$(h_\pi-\frac{1}{2\tau})^2$，这是平方误差（MSE），**目标是让差距正好等于一个固定值 $\frac1{2\tau}$**，到了就不再拉。

> **一句话生活类比**：DPO 像"分数越高越好，能拉多开拉多开"；IPO 像"及格线定在 $\frac1{2\tau}$ 分，到线即可，别使劲卷"。为什么这样更稳？因为如果数据里某对偏好被标成了 100% 确定（哪怕是标注噪声），DPO 会把坏回答概率一路压到 0、模型过拟合；IPO 因为有固定目标值，KL 约束（由 $\tau$ 控制）**始终生效**，不会被极端标签带偏。$\tau$ 就是"及格线松紧"：$\tau$ 越大、目标值 $\frac1{2\tau}$ 越小、越贴近参考模型、越保守。

### 10.5 直观理解与实验
- DPO 用 sigmoid+log（像分类交叉熵），目标值在确定性偏好下跑向无穷；IPO 用平方误差（MSE），把差距钉在 $\frac1{2\tau}$ 附近，从而**始终保留有效的 KL 约束**。
- **玩具实验**：3 个动作严格排序时，DPO 收敛到 $\pi(y_a)=1$（不管 $\tau$）；IPO 随 $\tau\to\infty$ 趋向均匀。某动作缺乏比较数据时，DPO 直接把它概率打到 0，IPO 只按 $\tau$ 比例温和下调，**不会把动作排除**。

### 10.6 优缺点

| 优点 | 缺点 |
|---|---|
| 对确定性/噪声小的硬标签偏好更鲁棒 | 实际大模型上相对 DPO 的增益不总是显著 |
| 理论上 KL 正则始终有效 | 多一层理论，超参 $\tau$ 需调 |
| 适合偏好标签干净、确定性高的数据 | 工程上不如 DPO 直接、生态成熟 |

**论文**：*A General Theoretical Paradigm to Understand Learning from Human Preferences* — Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos, AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)

## 11. ORPO（Odds Ratio Preference Optimization）

### 11.1 一句话直觉
ORPO 把"先 SFT，后 DPO"两步**合并成一步**：在监督微调的同时，用一个 odds ratio 惩罚项把不喜欢的回答压下去、喜欢的抬上来。**不需要参考模型，也不需要单独的偏好阶段**。

### 11.2 动机：SFT 自己的问题
论文的关键观察（Figure 3）：在 HH-RLHF 上只用 chosen 回答做 SFT 时，**chosen 和 rejected 的对数概率会同时上升**——因为交叉熵只奖励 chosen，没有任何机制压制 rejected。所以"想抬好的，结果连坏的一起抬了"。结论：应该在 SFT 阶段就顺手惩罚不良风格，而不是另起一个对齐阶段。

### 11.3 ORPO 目标
Odds（几率）定义：$\text{odds}_\theta(y|x)=\frac{P_\theta(y|x)}{1-P_\theta(y|x)}$。损失两部分：

$$
\mathcal{L}_{\text{ORPO}}=\mathbb{E}_{(x,y_w,y_l)}\big[\mathcal{L}_{\text{SFT}}+\lambda\,\mathcal{L}_{\text{OR}}\big]
$$

$$
\mathcal{L}_{\text{SFT}}=-\log\pi_\theta(y_w|x),\qquad \mathcal{L}_{\text{OR}}=-\log\sigma\!\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)
$$

- $\mathcal{L}_{\text{SFT}}$：学会生成正确文本。
- $\mathcal{L}_{\text{OR}}$：让 chosen 的 odds 显著高于 rejected 的 odds。
- $\lambda$：偏好优化强度。

### 11.3.1 拆解

**先懂 odds（几率）。** $\text{odds}=\frac{P}{1-P}$ 是赌博里的说法："发生的概率 ÷ 不发生的概率"。$P=0.5$ 时 odds=1，$P=0.9$ 时 odds=9（"9 比 1"）。它和概率一一对应，只是换了个刻度，对大概率更敏感。

**loss 两部分各管一件事：**
- $\mathcal L_{\text{SFT}}=-\log\pi_\theta(y_w\mid x)$：就是普通的监督微调——**照着好答案 $y_w$ 学**，提高它的生成概率。
- $\mathcal L_{\text{OR}}=-\log\sigma\big(\log\frac{\text{odds}(y_w)}{\text{odds}(y_l)}\big)$：让好答案的几率**除以**坏答案的几率尽量大，即"好的相对坏的要明显更可能"。
- $\lambda$ 调这两者的配比。

> **一句话生活类比**：别的方法是"先 SFT 学会说话，再单独开一个阶段 DPO 学会分好坏"，两步走还要多带一个参考模型。ORPO 说"一步到位"——**边学好答案，边顺手把坏答案的几率压下去**，既不需要参考模型 $\pi_{\text{ref}}$，也不需要单独的偏好训练阶段。用"几率之比"而不是"概率之比"，是因为前者惩罚更温和，不会在模型还没学好时就把坏 token 打得太死（详见 §11.4）。

### 11.4 为什么用 odds ratio 而非 probability ratio
论文论证：odds ratio 对 rejected 是"温和惩罚"，而 probability ratio 是"极端惩罚"（分布更尖，Figure 6）。在 SFT+对齐**同时进行**时，若用 probability ratio 会在模型还没适应目标域时就过度压制 rejected token 的 logit，破坏训练。odds ratio 的温和性更适配这种联合训练。

### 11.5 实验结果（原文）
- **AlpacaEval 2.0（LC win rate）**：Mistral-ORPO-α(7B) **11.33%**、Mistral-ORPO-β(7B) **12.20%**；Phi-2(2.7B)+ORPO 6.35%（超过 Llama-2-Chat 7B）；Llama-2(7B)+ORPO 9.44%（超过 Llama-2-Chat 13B）。
- **MT-Bench**：Mistral-ORPO-α **7.23**、β **7.32**，论文称可与体量大得多的模型相比。
- 对照 RM 胜率：相对 DPO 在 HH-RLHF 上最高约 70.9%，相对 SFT/PPO 普遍 78–84%。

### 11.6 优缺点

| 优点 | 缺点 |
|---|---|
| 单阶段训练，流程极简，省一个对齐阶段 | odds ratio 对概率值敏感 |
| **不需要 reference model** | 对数据质量和 $\lambda$ 敏感 |
| 在 Mistral-ORPO 等上效果不错 | 理论成熟度不如 DPO，且把 SFT 与对齐耦合，调参更微妙 |

**论文**：*ORPO: Monolithic Preference Optimization without Reference Model* — Jiwoo Hong, Noah Lee, James Thorne, EMNLP 2024. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)

## 12. SimPO（Simple Preference Optimization）

### 12.1 一句话直觉
SimPO 比 DPO 更简单：**连参考模型都不要了**。它用"每个 token 的平均对数概率"作为隐式奖励，并加入一个目标奖励间隔 $\gamma$，让好坏回答的差距更明显。动机是让"训练时的奖励"与"推理时实际指导生成的指标"对齐。

### 12.2 核心设计
**(1) 与生成对齐的无参考奖励。** DPO 的隐式奖励是 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，但模型实际生成时打分用的是**平均对数似然**。SimPO 直接拿后者当奖励（Eq.4）：

$$
r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x)=\frac{\beta}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})
$$

**(2) 长度归一化。** 除以 $|y|$ 避免"靠写更长刷分"。论文发现：不做长度归一化时，奖励与长度强正相关（Spearman $\rho=0.82$），归一化后降到 0.34。

**(3) 目标奖励间隔 $\gamma$。** 在 BT 目标里加一个固定 margin（Eq.6）：

$$
\mathcal{L}_{\text{SimPO}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)-\gamma\right)\right]
$$

强制好回答比坏回答至少高出 $\gamma$。

### 12.2.1 拆解

**符号：** $|y|$ 是回答的 token 数（长度）；$\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid\dots)$ 是把每个 token 的对数概率加起来，即"模型生成整段话的总把握"。

**SimPO 的奖励** $r=\frac{\beta}{|y|}\log\pi_\theta(y\mid x)$ 就是**每个 token 的平均对数概率**——"模型平均每个字有多大把握说出来"。注意这里**没有 $\pi_{\text{ref}}$**，比 DPO 更省。

**三个设计对应三个动机：**
1. **拿平均对数概率当奖励**：因为模型推理时给候选排序用的就是这个量，训练和推理指标对齐，不打架。
2. **除以 $|y|$（长度归一化）**：不除的话，回答越长、对数概率累加的负项越多/越少，模型会学会"靠写长刷分"。除以长度就把这个漏洞堵上。
3. **减 $\gamma$（目标间隔）**：要求好回答的平均把握比坏回答**至少高出 $\gamma$**，逼开差距，不满足才有 loss。

> **一句话生活类比**：DPO 要带一个参考模型当尺子；SimPO 说"不用尺子了，直接看模型自己对好/坏答案的平均信心差多少，好的必须比坏的信心高出一截（$\gamma$）"。更轻，但因为丢了参考模型这个锚，对超参 $\gamma,\beta$ 更敏感（见 §12.5）。

### 12.3 与 DPO 的对照分析
论文指出 DPO 的奖励排序与生成似然**不一致**：Figure 4(b) 显示，满足 DPO 奖励排序的偏好对里，竟有**近一半** $p_\theta(y_w|x)<p_\theta(y_l|x)$（即 DPO 认为赢的，模型生成概率反而更低）。SimPO 的奖励直接就是生成指标，reward accuracy 更高。去掉参考模型还省约 20% 运行时间、10% 显存。

### 12.4 实验结果（原文）

| 设置 | 基准 | DPO | **SimPO** |
|---|---|---|---|
| Llama-3-8B-Instruct | AlpacaEval 2 LC | 40.3% | **44.7%** |
| Llama-3-8B-Instruct | Arena-Hard | 37.9% | **40.5%** |
| Mistral-7B-Instruct | AlpacaEval 2 LC | 26.8% | **32.1%** |
| Mistral-7B-Instruct | Arena-Hard | 24.9% | **34.8%** |
| Llama-3-8B (Base) | Arena-Hard | 15.9% | **23.4%** |

消融（Table 5）：去掉长度归一化，Mistral-Instruct 的 AlpacaEval 2 LC 从 32.1% 暴跌到 19.1%——长度归一化是 SimPO 的命门。

### 12.5 优缺点

| 优点 | 缺点 |
|---|---|
| 无需参考模型，省内存/时间 | 没有 reference 锚点，缺少显式 KL 约束，离线下对 reward hacking / 分布漂移更敏感 |
| 长度归一化有效缓解长度偏见 | 对 $\gamma$、$\beta$ 敏感，且 $\gamma$ 与长度归一化强耦合 |
| AlpacaEval 2 / Arena-Hard 上超过 DPO，实现极简 | 安全性、开放域稳定性等场景仍需更多验证 |

**论文**：*SimPO: Simple Preference Optimization with a Reference-Free Reward* — Yu Meng, Mengzhou Xia, Danqi Chen, NeurIPS 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

## 13. 横向对比总表

| 方法 | 数据要求 | 需要 Reward Model | 需要 Reference Model | 在线/离线 | 核心思想 | 适合场景 |
|---|---|---|---|---|---|---|
| **PPO** | 在线采样 + 奖励 | ✅ 需要 | ✅ 需要 | 在线 | Clip + GAE Advantage + KL | 通用 RLHF，开放域 |
| **DPO** | 成对偏好 $(x,y_w,y_l)$ | ❌ | ✅ | 离线 | 解析消掉 RM，BT 分类 loss | 通用偏好对齐 |
| **GRPO** | 成组采样 + 可验证奖励 | ❌（组内 baseline） | ⚠️ 原版需要，变体常去掉 | 在线 | Critic-free + 组内相对优势 | 数学、代码等可验证任务 |
| **KTO** | 二元反馈 $(x,y,\text{label})$ | ❌ | ✅ | 离线 | 前景理论 / HALO + 损失厌恶 | 单条反馈、数据严重不平衡 |
| **BCO** | 二元反馈 $(x,y,\text{label})$ | ❌（分类器 logit 当奖励） | ✅ | 离线 | BCE 上界 DPO + reward shift + UDM | 真实点赞/Likert 数据 |
| **Step-KTO** | 步骤级 + 结果级二元反馈 | ⚠️ 用 LLM 当 PRM | ✅ | 迭代离线 | KTO 细化到推理步骤 | 数学/长链推理 |
| **TR-***（DPO/IPO/KTO） | 同基础方法 | ❌ | 🔄 动态更新 | 离线 | 信任区域 + 动态参考模型 | 想摆脱固定 SFT reference |
| **IPO** | 成对偏好 | ❌ | ✅ | 离线 | ΨPO + Identity + MSE | 偏好标签硬、确定性高 |
| **ORPO** | 成对偏好 $(x,y_w,y_l)$ | ❌ | ❌ | 离线（含 SFT） | SFT + odds ratio 单阶段 | 一次性完成 SFT+对齐 |
| **SimPO** | 成对偏好 | ❌ | ❌ | 离线 | 长度归一化平均似然 + margin | 极简实现、缓解长度偏见 |

## 14. 新手学习路线建议

### 第一步：理解 RLHF 大框架
先看 PPO + Reward Model 的传统流程，理解：为什么要 KL 约束？Advantage / GAE 是什么？为什么需要 Critic？

### 第二步：理解 DPO 的"奖励模型无用论"
DPO 是整个直接偏好优化家族的基石。掌握"KL 约束最优解 → 反解奖励 → 配分函数抵消"这条推导，再看 IPO、ORPO、SimPO 会很快。

### 第三步：按数据类型选方法
- **有干净成对偏好**：DPO → 试 IPO（防确定性过拟合）/ SimPO（无 ref、抗长度偏见）/ ORPO（合并 SFT）。
- **只有点赞/点踩**：KTO → BCO（真实日志）→ Step-KTO（推理步骤）。
- **数学/代码可验证任务**：GRPO → Dr.GRPO / DAPO（修正偏差、稳定长度）。
- **想摆脱固定 reference**：TR-DPO / TR-IPO / TR-KTO。

### 第四步：动手实验
- Hugging Face **TRL** 内置 `DPOTrainer`、`KTOTrainer`、`BCOTrainer`、`ORPOTrainer`、`CPOTrainer`（含 SimPO 选项）等。
- GRPO 可用 **TRL 的 `GRPOTrainer`**、**OpenRLHF** 或 **verl**，在 GSM8K/MATH 上试效果，并对照原版 vs Dr.GRPO 的长度曲线。

## 15. 参考来源（均已核实）

- PPO: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)；InstructGPT: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- DPO: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)（NeurIPS 2023）
- GRPO / DeepSeekMath: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
  - Dr.GRPO: [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)；DAPO: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- KTO: [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- BCO: [arXiv:2404.04656](https://arxiv.org/abs/2404.04656)（ACL 2025 main）
- Step-KTO: [arXiv:2501.10799](https://arxiv.org/abs/2501.10799)
- TR-DPO/TR-IPO/TR-KTO（Learn Your Reference Model）: [arXiv:2404.09656](https://arxiv.org/abs/2404.09656)
- IPO / ΨPO: [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)（AISTATS 2024）
- ORPO: [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)（EMNLP 2024）
- SimPO: [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)（NeurIPS 2024）
