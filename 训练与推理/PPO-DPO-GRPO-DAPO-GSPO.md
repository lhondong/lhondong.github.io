# PPO-DPO-GRPO-DAPO-GSPO

经典的三阶段 RLHF 流程

标准的 PPO-RLHF 流程是一个系统性的工程，旨在将抽象的人类偏好转化为具体的模型行为。它主要包含以下三个阶段：

1. **监督微调（Supervised Fine-Tuning, SFT）**: 首先，选择一个预训练好的 LLM 作为基础模型。然后，使用一个高质量、人工策划的指令-回答数据集对该模型进行微调。这个阶段的目标是让模型初步具备遵循指令和进行特定任务（如对话、摘要）的能力，为后续的 RL 阶段提供一个良好的策略起点。

2. **奖励模型（Reward Model, RM）训练**：此阶段的核心是将人类偏好量化。首先，针对一系列 prompts，使用 SFT 模型生成两个或多个不同的回答。然后，人类标注员对这些回答进行排序，指出哪个更好。这些成对的偏好数据（例如，对于提示 $x$，回答 $y_w$ 优于 $y_l$）被用来训练一个独立的奖励模型 $r_\phi(x,y)$。该模型输入一个提示和回答，输出一个标量分数，分数越高代表该回答越符合人类偏好。奖励模型通常基于 Bradley-Terry 模型进行优化，最大化预测偏好与人类标注一致的概率。

3. **强化学习微调（RL Fine-Tuning）**: 这是最后也是最核心的阶段。SFT 模型作为初始策略 $\pi_{\text{SFT}}$，在 RL 环境中进行优化。环境每次提供一个提示 $x$，策略模型 $\pi_\theta$ 生成一个回答 $y$。随后，训练好的奖励模型 $r_\phi$ 对这个回答 $(x, y)$ 进行评分，得到一个奖励值。PPO 算法的目标就是最大化这个奖励值，同时通过一个 KL 散度惩罚项来约束优化后的策略 $\pi_\theta$ 不过于偏离初始的 SFT 策略 $\pi_{\text{SFT}}$，以防止模型遗忘预训练知识或产生语法不通顺的文本。

PPO:

$$
\mathcal{J}_{\mathrm{PPO}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_{\theta_{\text{old}}}(\cdot|x)} \left[\frac{1}{\lvert y\rvert} \sum^{\lvert y\rvert}_{t=1} \min \left( w_t(\theta) \hat{A_t}, \operatorname{clip}(w_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A_t} \right) \right]
$$

其中 $w$ 是重要性权重，$\mathbf{w}_i=[w_{i,1}, w_{i,2}, \ldots, w_{i, L}]$，其粒度是实际生成的 token，衡量 “新策略生成某段内容” 与 “旧策略生成同一段内容” 的概率差异，而不是词表上的 token 概率分布的比较（KD 任务）。

A 是维护的 Critic 价值模型针对生成的 token 给分：

1. Critic 和 w 的尺度是对齐的，全是 token；
2. Critic 往往和训练的模型相似大小，需要维持两个网络，占用内存和计算负担，算法的有效性也过度依赖于其奖励有效性。

PPO 算法在 LLM 上的初始应用是作为模型输出内容的一种偏好调节，旨在使得模型输出更贴合人类偏好的回答内容，在这个过程中需要让人类去针对问题进行排序标注，使用排序标注好的模型来训练奖励（RM）模型以及价值（Value）模型，一般初始的奖励模型和价值模型是同一个模型（也有使用不同模型的情况），在 RLHF（人类偏好强化训练）过程中价值模型会进行参数更新，而奖励模型则仅进行推理输出奖励值。

## PPO

### 价值函数损失 (LVF)

PPO 通常与一个评论家（critic）网络（即价值函数 $V_\psi(s_t)$）一起训练，该网络的目标是准确估计在给定状态 $s_t$ 下未来奖励的总和。价值函数的主要作用是作为基线（baseline）来计算优势 $\hat{A}_t$，从而显著降低策略梯度的方差。价值函数的损失通常是一个简单的均方误差（MSE）：

$$
\mathcal{L}^{\text{VF}}(\psi) = \hat{\mathbb{E}}_t \left[ (V_\psi(s_t) - V_t^{\text{target}})^2 \right] \\
$$

其中，$V_t^{\text{target}}$ 是回报的蒙特卡洛估计，即从时间步 $t$ 开始的累积折扣奖励。

### 熵奖励 (S)

为了鼓励策略进行充分的探索，避免过早地收敛到次优的确定性策略，PPO 的目标函数中通常会加入一个熵奖励项。

熵 $S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \cdot \log \pi_\theta(a|s_t)$  衡量了策略在状态 $s_t$ 下输出动作的随机性。最大化熵等同于让策略的概率分布更平坦，从而探索更多可能性。策略对某个 token 的概率接近 1（确定性强），则该 token 的贡献项接近 0；若所有 token 概率均匀（随机性强），则熵值最大。

### 最终复合目标

PPO 的最终优化目标是上述三个部分的加权和。策略网络和价值网络可以共享部分参数，并通过一个统一的损失函数进行端到端的优化：

$$
\mathcal{L}^{\text{PPO}}(\theta, \psi) = \hat{\mathbb{E}}_t \left[ \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\psi) + c_2 S[\pi_\theta](s_t) \right]
$$

其中，$c_1$ 和 $c_2$ 是用于平衡各项损失重要性的超参数。在每个优化步骤中，算法通过最大化这个 $\mathcal{L}^{\text{PPO}}$ 来更新模型参数 $\theta$ 和 $\psi$。

### 优势函数

优势函数计算方式有很多种，其公式为：

$$
A_t=A_t(s, a) = Q_t(s, a) - V_t(s)
$$

优势函数中 $Q_t$ 是可以展开的，根据展开的方式不同也会存在非常多类型的优势函数，如果是一步展开即 TD（时序差分），如果全部展开到序列结束，则为 MC，介于两者中间的为 GAE。不同展开方式各有优劣，主要是方差与偏差。

TD 误差 Temporal Difference Error

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

广义优势函数 GAE（对 TD 误差加权求和）

$$
\begin{aligned}
\hat{A}_t &= \delta_t + (\gamma\lambda) \hat{A}_{t+1} \\ 
&= \delta_t + (\gamma\lambda) \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots + (\gamma\lambda)^{T-t-1} \delta_{T-1} \\ 
&= \sum_{l=0}^T (\gamma\lambda)^l \delta_{t+l}
\end{aligned}
$$

GAE 的核心思想是通过对多个时间步的 TD 误差进行加权求和来估计优势函数，当 $\lambda=1$ 时，GAE 退化为 MC 估计，当 $\lambda=0$ 时，GAE 退化为单步 TD 误差。

在 PPO 算法应用在 LLM 的 RLHF 训练过程中一般是存在 4 个模型：

1. Actor 模型也就是训练的主模型，其训练的目标函数也就是上面的目标函数，在对其训练中可以看到会计算优势函数 $A_t$ ，优势函数的简单理解就是在当前状态（LLM 上下文）情况下选择特定动作（选择哪个 token）相比较其他动作（vocab 中其他 token）带来的优势值，这里的 $(s,a)$ 代表当前状态下具体选择的动作（action）， $Q_t(s,a)$ 则代表当前状态选择具体价值后会带来多少潜在价值，而 $V_t(s)$ 则代表当前状态下所有动作潜在价值的期望，也代表了当前状态的潜在价值。二者差值则代表在当前状态下选择该动作相较其他所有动作均值的价值优势，因此也称为优势函数。
   - 如果使用 TD 方法来展开 $Q_t(s,a)$ ，则 $Q(s,a)=r_t+\gamma V(s_{t+1})$ ，这里的 $r_t$ 代表状态 $s$ 下选择动作 $a$ 带来的价值收益（单步价值），$\gamma$ 为奖励衰减因子，$V(s_{t+1})$ 代表执行动作后下一个状态的潜在价值。
   - 如果使用 MC 方法来展开 $Q_t(s,a)$ ，则 $Q(s,a)=\sum_{l=0}^{T-t} \gamma^l r_{t+l}$ ，$r_{t+l}$ 代表从当前时间步 $t$ 开始未来每个时间步的奖励值。
2. **奖励模型（Reward Model）**: 在 RL 微调阶段参数固定，为策略模型提供奖励信号，优势函数中奖励值 $r_t$。
3. **价值模型（Value Model）**: 通常称为 Critic，用于评估在给定状态下策略可能获得的期望回报，即状态价值 $V_t(s)$ 及 $V(s_{t+1})$，以降低策略梯度的方差。 
4. **参考模型（Reference Model）**: 通常是 SFT 模型的副本，参数固定，用于计算 KL 散度惩罚。

以 GRPO 论文中的示意图来理解更准确：

![image-20250921074651880](../../assets/image-20250921074651880.png)

可以看到 PPO 算法涉及到了 4 个模型，更主要的是奖励模型需要提前进行训练，而价值模型也是一个 LLM，这就造成两个问题。

1. 首先奖励模型的训练工作很复杂，虽然 DPO 算法可以通过绕过奖励模型的方案来训练，但 DPO 需要构造负样本，负样本与正样本的质量会影响训练的结。
2. 其次状态价值是由一个 LLM 给出的，这个值具备不确定性，也就是说价值模型提供的状态价值可能不准确，这样也就使得 PPO 训练过程变得极其脆弱。我们的目标是来最大化目标函数，价值模型预估偏差会使得 Actor 模型参数梯度往错误方向走一大步，很容易造成训练崩溃。
3. **Reward Hacking**：Actor 学会了利用奖励模型的漏洞来获得高分，而其生成的回答在人类看来质量并未提升，甚至更差。这种现象也被称为“模式崩溃（pattern collapse）”。模型会发现某些特定的文本模式或风格（如过度礼貌、冗长）能稳定地骗取奖励模型的高分，从而过度生成这些模式，导致输出质量下降。这暴露出 RLHF 的上限严重依赖于奖励模型的质量和鲁棒性。

另外还有一点考虑，我们之前是对 LLM 输出内容进行人类偏好的强化学习训练，人类偏好本身是没有固定规则的，因此才需要去训练奖励模型来进行 LLM 输出内容是否符合人类偏好的打分，但对于程序，数学等任务是具备规则特性的，对于一个程序任务或者数学任务，LLM 输出结果是可以使用规则来判断正确性的，这也是 GRPO 提出的一个前提。

## DPO：绕过奖励模型

标准的 RLHF 目标（即 KL 约束下的奖励最大化）存在一个封闭形式的最优解。这意味着，对于任意一个奖励函数 $r(x,y)$，其对应的最优策略 $\pi_r(y \mid x)$ 可以被解析地表示出来：

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x,y)\right)
$$

其中，$Z(x)$ 是确保概率归一化的配分函数，$\beta$ 是控制 KL 惩罚强度的超参数。

DPO 的关键一步是**逆向操作**：将这个方程重新整理，用最优策略 $\pi_r$ 和参考策略 $\pi_{\text{ref}}$ 来表示奖励函数 $r(x, y)$：

$$
r(x,y) = \beta \log\left(\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right) + \beta \log Z(x)
$$

奖励函数的重参数化是 DPO 理论的基石，建立了一个从策略空间到奖励空间的直接映射，意味着任何一个语言模型策略都隐含地定义了一个奖励函数。

DPO 的损失函数推导过程巧妙地将上述理论与人类偏好模型结合起来。

1. **从 Bradley-Terry 模型开始**：与奖励建模阶段类似，DPO 假设人类偏好遵循 Bradley-Terry 模型。该模型指出，人类偏好回答 $y_w$ 胜过 $y_l$ 的概率 $p^*$ 取决于它们在某个潜在的真实奖励函数 $r^*$ 下的得分差异：$p^*(y_w \succ y_l \mid x) = \sigma(r^*(x, y_w) - r^*(x, y_l))$ 其中 $\sigma(\cdot)$ 是 logistic sigmoid 函数。

2. **代入重参数化的奖励**：将重参数化奖励公式（应用于真实奖励 $r^*$ 和其对应的最优策略 $\pi^*$）代入 BT 模型中的 $r^*(x, y_w)$ 和 $r^*(x, y_l)$：
	$$
	r^*(x, y_w) - r^*(x, y_l) = \left(\beta \log\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta\log Z(x)\right) - \left(\beta \log\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta\log Z(x)\right)
	$$
	
	观察到未知的配分函数项 $\beta \log Z(x)$ 在相减过程中完全抵消了，那么奖励差异可以完全用策略来表示：
	
	$$
   r^*(x, y_w) - r^*(x, y_l) = \beta \log\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
	$$

3. **构建最大似然目标**：有了这个只依赖于策略的偏好概率表达式，DPO 就可以直接为策略模型 $\pi_\theta$ 构建一个最大似然目标。给定一个人类偏好数据集 $\mathcal{D} = \{ (x^{(i)}, y_w^{(i)}, y_l^{(i)}) \}_{i=1}^N$，DPO 的目标是最大化观测到这些偏好的对数似然。这等价于最小化负对数似然损失，即 DPO 的最终损失函数：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

​	这个损失函数本质上是一个简单的二元交叉熵损失，激励 $\pi_\theta$ 提高其对 $y_w$ 的相对概率（相对于 $\pi_{\text{ref}}$ ），同时降低其对 $y_l$ 的相对概率。

DPO 与 PPO-RLHF 相比，展现出显著的优势：

- **简洁性**：DPO 将复杂的 RLHF 流程简化为一步监督微调，不需要训练一个独立的奖励模型，也不需要复杂的强化学习采样和优化循环。
- **稳定性与效率**：由于避免了 RL 训练中的不稳定性（如奖励滥用和超参数敏感性），DPO 的训练过程通常更稳定，计算效率也更高。
- **性能**：实验表明，DPO 不仅实现起来更简单，而且在多种任务（如摘要和对话）上能够达到甚至超过 PPO 的性能。DPO 训练的模型在与人类评估者的比较中也表现出更高的胜率。

总而言之，DPO 通过一个深刻的理论洞见，成功地将偏好学习问题转化为了一个可以直接优化的分类问题，极大地简化了 LLM 的对齐过程。

### 无 Critic 基于组的策略优化

尽管 DPO 提供了一条优雅的替代路径，但在许多需要模型进行复杂、多步推理的任务（如数学、编程）中，基于强化学习的方法依然显示出独特的优势。这类任务的奖励通常是二元的（答案正确或错误），可以通过程序自动验证，这被称为可验证奖励的强化学习（RL with Verifiable Rewards, RLVR）。然而，PPO 的计算和内存开销依然是一个巨大的障碍，特别是其与策略模型大小相当的评论家模型。这催生了一系列旨在保留 RL 优势同时降低其复杂性的新算法。

这一系列算法的演进过程揭示了一个清晰的研究轨迹。首先，GRPO 通过移除 PPO 中资源消耗巨大的评论家模型，实现了在推理任务上的重大效率突破。它引入了一系列更微妙的、先前未被充分认识的问题。

## GRPO

GRPO 中一般存在两个模型，主模型 Actor 模型以及参考模型，这两个模型初始时是同一个模型，但训练过程中参考模型只作为 KL 散度约束项来防止主模型训练中权重偏离原始模型权重太多。PPO 中奖励模型更换成了特定的规则函数来进行奖励值打分，PPO 中的价值模型则直接取消，优势函数的计算更换成了 LLM 一组输出的奖励值标准化的形式。

GRPO 目标函数：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{\vert o_i \vert} \sum_{t=1}^{\vert o_i \mid} \min \left(\frac{\pi_{\theta}(o_{i,t} \mid \cdot)}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid \cdot)} \hat{A}_{i,t}, \text{clip}\left(\dots\right) \hat{A}_{i,t} \right) \right]   \\
\hat{A}_{i,t} = \hat{A}_{i} = \frac{r(x, y_i)-\operatorname{mean}(\{r(x, y_i\}^G_{i=1})}{\operatorname{std}(\{r(x, y_i\}^G_{i=1}))}
$$
对于一个输入 $q$，GRPO 会产生一组 $G$ 个输出，对于这些输出会使用规则函数分别计算奖励值，即 $\mathbf{r} = \{ r_1, r_2, \ldots, r_G \}$ ，使用这一组不同输出来计算优势函数，可以看到优势函数的计算只依赖于每个输出的结果奖励，对整个输出序列即多个 token 进行整体的奖励值计算，但注意此时动作的粒度仍是单个 token，而不是整个输出序列。

计算完每个输出的优势值，即 $\hat{A}_{i,t}$ 后，会将这个整体优势值均匀分配给该序列的每个 token，这么做还是因为 GRPO 的动作粒度是 token，后面进行重要性采样时是以 token 为粒度来进行计算的。也就是为什么优势值下标会带着时间步 $t$，实际上在计算时，同一个序列的每个 token 优势值是相同的。

GRPO 公式里也带上了 Actor 模型与参考模型的 KL 散度，KL 散度直观理解上十分简单，就是两个不同离散数据分布的相似程度，两个分布越相近则 KL 散度值越小，两个分布越大则 KL 散度值越大。我们强化训练目标是最大化目标函数，因此需要减去这个 KL 散度值，如果目标模型训练过程造成与原始权重偏差较大，这时 KL 散度值变大，目标函数计算时会减去较大的 KL 值，因此使得整体目标值变小，来达到约束模型训练不过分偏移原权重的目的，反之亦然。PPO 中目标函数也有这个 KL 约束，只不过省略掉了。

GRPO 使用 k3 unbias estimator 计算 KL，保证值为正：

$$
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{\mathrm{ref}}\right]=\frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}-\log \frac{\pi_{\mathrm{ref}}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}-1
$$

GRPO 通过移除了价值网络，而是依赖于 sentence 级别的是否正确来决定，奖励有效性由黑盒到白盒，完全依赖于自己提供的答案的准确性；此外，由于在一个组内生成答案，所以就用组间的对比来重新计算奖励，只有优于组间平均值的才能进一步得到强化。这样“内卷”起来，使得模型性能提升很大。

### GRPO 的问题

GRPO 的问题：每次训练时组的规模 $G$ 取多大合适？如果该组内输出全错或全对怎么办？如果使得组内 $G$ 个输出尽可能不一样？

这也是强化学习一直强调的“探索”，“探索”在强化学习训练中是十分重要的，如果模型想获得更有的策略决策能力，也就是在不同状态下需要去选择更优的动作就需要去“探索”新的动作带来的收益，因为强化学习训练本身就是一个通过不断试错来找到最优解的过程，如果在强化学习训练过程中让策略模型（也就是 Actor 模型）没有足够的“探索”能力（也就是模型输出不够多样）那模型就不会找的更好的路径，就会造成所谓的“熵坍塌”，就是模型只在少量的输出中训练，这样使得模型输出分布变得越来越尖锐，模型分布会变得集中，分布的熵会变小。因此 GRPO 训练中 $G$ 应该更大一些（当然也需要分任务来说，有的输出范围本身就很小，分组规模也没必要大）。 $G$ 中多样性也是通过不同的采样方法，不同温度超参数来进行采样控制。而如果同一组内模型输出都是错误的情况，那这个 step 训练过程来说也就变得没有意义。

其实 GRPO 还有个问题，就是极度依赖奖励函数，这也是对 PPO 精简后带来的一个显而易见的问题，去掉价值函数来对状态价值进行预估，仅使用奖励值来计算优势，那么就需要奖励值尽可能的公平公正，对于简单任务来说不是很困难，但对于大多数实际任务来说，奖励函数是需要多方面权衡的，很难找到一个标准化的奖励函数去做奖励判断，如智能体任务中如果只将智能体任务完成与否来作为奖励值，那就没办法权衡智能体中间状态是否合理有效，是否有冗余轨迹等。GRPO 训练甚至转变为去拟合奖励函数，换句话说会去钻奖励函数的空子，模型会为了获得高的奖励值生成与实际需要不符合的结果，因此奖励函数一定要综合权衡考虑。

GRPO 的奖励是序列级别，而动作是 token 级别，前面介绍了这是为了简化 PPO 算法带来的，但是这种动作与奖励粒度的不一致性其实会带来一些影响，后面 GSPO 会重点针对这个问题阐述方案。另外，GRPO 中的重要性采样是有问题的。

### 关于重要性采样修正项

重要性采样修正项

$$
r_{i,t} = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

分母是旧策略模型产生的每个 token 的概率，分子则是经过几轮训练后新策略 prefill 旧策略输出得到的该策略中对应 token 上的概率值。之所以使用重要性采样是因为强化训练轨迹是很珍贵的，一般情况下是需要多轮训练来提升利用率，这就使得当训练当前策略模型时会使用之前旧策略模型产生的轨迹，而这两个模型实际上输出分布是不同的，如何对当前策略使用旧策略的分布，这就是重要性采样修正项的作用。

注意 $w$ 的粒度是实际生成的 token，衡量的是"新策略生成某段内容"与"旧策略生成同一段内容"的概率差异，而不是词表上 token 概率分布的比较（区别于 KD 任务）。

假设存在两个数据分布， $p_0$ 和 $p_1$ ，可以使用下面等式将两个分布的期望连接起来，里面 $\frac{p_0(x)}{p_1(x)}$ 称为重要性权重修正项，做为两个分布之间的修正项。
$$
\mathbb{E}_{X \sim p_0}[X] = \sum_{x \in \mathcal{X}} p_0(x)x = \sum_{x \in \mathcal{X}} p_1(x)\frac{p_0(x)}{p_1(x)}x = \mathbb{E}_{X \sim p_1}[f(X)]
$$
上式中 $p_1$ 代表旧策略的分布，也就是 Actor 模型做 Rollout 的分布，而 $p_0$ 代表训练中新策略模型的分布，我们当前仅知道旧策略分布情况（即之前 Rollout 过程），但我们目标是使得当前训练策略模型目标函数更高，但又不知道新策略模型的分布情况（因为没有 Rollout），因此需要使用旧策略分布来表示新策略分布，即使用 $p_1$ 来表示 $p_0$ 。看公式的期望下标，经过重要性采样成功使用旧策略的分布来表示新策略。返回 PPO 和 GRPO 目标函数公式也可以看到，最外层的期望是旧策略模型的分布，不是新策略模型分布。

但是，重要性采样只是让两个分布在期望上达到了统一，但没有在方差上统一。

根据方差计算公式：
$$
\mathrm{Var}[X]=E \begin{bmatrix} X^2 \end{bmatrix}-(E[X])^2
$$
即使两个分布期望相同也只是上面公式的后面一项相同，两个分布的方差还是不同的。

这也是为什么从 PPO 或者更早 TRPO 开始需要增加 clip 项和 min 来控制这个重要性采样修正项，实际上是在控制两个分布的方差不要差距太大。

### 重要性采样视角：句子级 vs token 级的无偏性推导

上面的方差分析还可以从无偏性的角度进一步展开。GRPO 引入了 sentence 级别的奖励，但是重要性采样还是沿用的 token 级别的，这也会导致一些问题：

从重要性采样 IS 出发，奖励 $R(y)$ 是 sentence 级别上的，所以写成以下形式：

$$
\mathcal{J}(\theta)=\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[R(y)]
$$

那么无偏的 IS 应该是：

$$
\hat{\mathcal{J}}_{\mathrm{IS}}=R(y) w(y)
$$

当奖励 $R(y)$ 是**句子级**（即基于整个生成序列 $y$ 给出，而非单个 token）时，无偏的重要性权重需覆盖整个序列的策略比值，即新策略 $\pi_\theta$ 与旧策略 $\pi_{\theta_{\text{old}}}$ 在生成该完整序列 $y$ 上的概率比值：

$$
w(y)=\frac{\pi_\theta(y)}{\pi_{\mathrm{old}}(y)}=\prod_{t=1}^T w_t, \quad w_t = \frac{\pi_\theta\left(y_t \mid y_{<t}, x\right)}{\pi_{\mathrm{old}}\left(y_t \mid y_{<t}, x\right)}
$$

这样才能无偏估计新策略下的期望奖励，因为完整考虑了“旧策略采样出该序列、新策略生成该序列的概率差异”，避免因仅局部（如 token 级）计算权重导致的偏差。

而 GRPO 中，相应替换的是：

$$
a_{\mathrm{AM}}(y) = \frac{1}{T} \sum^T_{t=1} w_t
$$

用 token 级权重的算术平均替代了全序列的乘积权重，打破了这种无偏性（算术平均≠乘积，且不符合概率比值的链式逻辑），引入了偏差。

为什么不能直接用无偏的乘积？

在 PPO/GRPO 优化时会对重要性权重 $w$ 取对数（$\log w$），原本 token 级 $w$ 需逐 token 相乘，取 log 后变成逐 token 累加更易计算梯度。假设每个 token 的 $\log w$ 是**独立同分布**的（即每个 token 的策略偏差相互独立），根据概率统计的“方差性质”：

1. 多个独立随机变量“和的方差”=各变量方差的和，$\mathrm{Var}(X_1+X_2+…+X_T) = \mathrm{Var}(X_1)+Var(X_2)+…+Var(X_T)$ 
2. 若每个 $\log w_i$ 的方差为 $\sigma^2$，那么 T 个 token 累加后，总方差就是 $T\times \sigma^2$ 。随着序列 T 长度越大，总方差会“线性变大”。

- **T大（长序列）→ 梯度爆炸**：总方差 $T\times \sigma^2$ 会随 T 急剧增大，意味着 $\log w$ 的累加结果波动极大（可能出现远超正常范围的极端值）。梯度由 $\log w$ 的累加结果计算，极端值会让梯度也变得异常大，导致参数更新时“一步跨度过大”，模型参数失控（即梯度爆炸）。 

- **T小（短序列）→ 梯度消失**：总方差 $T\times \sigma^2$ 过小，$\log w$ 的累加结果波动极小（接近一个固定值）。此时梯度会变得非常小，参数更新时“几乎没变化”，模型难以学到有效信息（即梯度消失）。

另外，单采样本身就会带来很大的偏差：

$$
\mathbb{E}_{z\sim\pi_\mathrm{tar}}[f(z)] = \mathbb{E}_{z\sim\pi_\mathrm{beh}} \left[\frac{\pi_\mathrm{tar}(z)}{\pi_\mathrm{beh}(z)}f(z) \right]
$$

但是没有办法，如果多次采样去平均值计算成本肯定会暴涨。所以我们的目标是单样本采样+更低偏差+低方差（维持稳定）。这也是后文 GSPO 将重要性权重改为序列级几何平均 $s_i(\theta)$（即无偏 IS 权重取 $\frac{1}{T}$ 次方）来同时降低偏差与方差的设计动机。

### on policy / off policy

当生成采样数据的模型与使用采样数据进行训练的模型是有相同权重（也就是同一个模型）时，换句话说就是自己产生数据来训练自己，叫做 on policy。相反，如果使用其他模型或者非目标模型产生的数据来训练就属于 off policy。如果是 on policy 的话，上面的重要性修正项的值就为 1，因为相同模型会具有相同的数据分布，只有 off policy 情况下才会提现出重要性采样的价值。

一般在 LLM 的 RL 训练中不会使用 on policy。这一点主要从训练效率出发的，一般会使用 Actor 模型 Rollout 一大批的样本出来，将这一大批样本分为几个训练批次来进行训练，这个过程中只有第一个批次样本进行训练时是 on policy 的，这个时候重要性采样修正项的比值大概为 1（考虑随机性波动），也就不起作用，但后面几个批次的目标模型权重已经发生了变化，再使用之前采样的数据进行 RL 训练就属于 off policy 了。

从算法理论上来说，如果不计较训练损耗，一直使用 on policy 来训练，GRPO 结论可能也未必会好，因为 GRPO 的问题在于动作粒度与奖励粒度的不均衡，在对 GRPO 目标函数计算梯度时每个 token 具有相同的优势值

GRPO 梯度公式：
$$
\nabla_{\theta} \mathcal{J}_{\text{GRPO}} = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\vert o_i \vert} \sum_{t=1}^{ \vert o_i \vert} A_i r_{i,t} \nabla_{\theta} \log \pi_{\theta}(o_{i,t} \mid \cdot) \cdot \mathbf{1} \Bigl\{ ( A_i > 0 \land r_{i,t} \leq 1 + \varepsilon ) \lor ( A_i < 0 \land r_{i,t} \geq 1 - \varepsilon ) \Bigr\} \right]
$$
这时候主要的梯度贡献其实是重要性采样修正项的值，如果这个值为 1 的话，GRPO 每个 token 就只剩下了相同的优势值，其实这样就退化成了 SFT 训练的交叉熵梯度，也就是每个 token 是具有相同奖励的行为克隆，因为从强化学习策略梯度定理角度来看，RL 相对于 SFT 梯度公式最大的区别在于每个动作（token）是具备不同优势值的。

## DAPO

DAPO 引言部分就提到了其训练 GRPO 时出现了熵坍塌、训练不稳定、奖励噪声的问题，这也对应了上面我们分析的几个 GRPO 固有缺陷，这一点 DeepSeek 应该是有应对的 Trick，但论文中并没有提到。

DAPO 是字节的工作，开源代码使用的 verl，针对 GRPO 存在的问题提出来解耦裁剪和动态采样策略优化（Dynamic sAmpling Policy Optimization  DAPO），另外 DAPO 应对的场景是长 COT 场景， 也就是带思考模式的输出情况（上面也提到，当生成序列如果很长，重要性采样修正项会造成方差偏移的累积）。

下面是 DAPO 针对 GRPO 做的修正工作：

### 移除 KL 散度

上面的 PPO 及 GRPO 目标函数中都存在 Actor 模型与参考模型的 KL 散度，KL 散度的意义也说过了，就是不想让训练的模型与最初始模型分布差距太大，但 DAPO 的训练方案应对场景是有长思维链输出（带思考过程）情况，长输出也就代表着对于输出 token 分布调整更大，那么训练后的模型就必然会与原始模型存在很大差异，因为目标就是让他们有差异，因此 KL 散度的约束反而不是必需的了。

### 算法

DAPO 为每个问题 $q$ 及其配对答案 $a$ 采样一组输出 $\{o_i\}_{i=1}^{G}$，并通过以下目标函数优化策略
$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip} \left( r_{i,t}(\theta), 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}} \right) \hat{A}_{i,t} \right) \right] \\ \text{s.t.}  \quad 0 < \vert \{ o_i \mid \text{is-equivalent}(a, o_i) \} \vert < G
$$
从公式上看与 GRPO 整体相差不大，但有一些小细节的改动。

### 提高上限：Clip-Higher

首先改动是 clip 函数中的 $\varepsilon$ ，在 PPO 及 GRPO 中都使用一个固定的超参数（一般是 0.2），但 DAPO 中分化成了 $\varepsilon_{low}$ 和 $\varepsilon_{high}$ ，DAPO 论文中叫提高上限 Clip-Higher。

DAPO 使用这种非对称的裁剪方案，是为了解决 GRPO 中 “旧策略高低概率 token 训练后差距扩大、导致熵坍塌” 的问题。

模型根据指令会自回归的生成带思考过程的输出内容，每一个 token 被选择的概率是依赖于之前 token 内容的，而且根据当前主流的采样方式（核采样，top-k，贪心等）在都会去选择概率较高的 token。而 GRPO 是对整个序列来计算奖励值与优势值的，这就使得在进行目标函数计算时每个 token 的优势值都一样，假设某条序列的优势值是正值，则训练的目标就是对该序列中的每个 token 都要正向的训练，训练的结果就是要提升每个 token 在前置 token 中被采样的概率，使得模型输出多样性会进一步减小，造成整体输出分布熵的减小，也就是熵坍塌。

重要性采样比值是新策略模型生成某个 token 与旧策略模型生成某个 token 的比值，当旧策略模型生成某个 token 的概率本身比较高时，其被裁剪的概率就会变低；而如果旧策略模型生成某个 token 的概率本身比较低时，其被裁剪的概率就会变高，因此提高裁剪高值可以缓解这个问题。

否则本来旧策略模型采样概率高的 token 会随着训练变得采样概率越来越高，而低的 token 只会有很小的提升，那么结果就是模型输出的分布越来越尖锐，也就使得分布的熵变低，造成熵坍塌现象。

具体来说，当 $\varepsilon = 0.2$（大多数算法的默认值）时，考虑两个动作，其概率分别为 $\pi_{\text{data}}(o_i | q) = 0.01$ 和 $0.9$。更新后的最大可能概率分别为 $\pi(o_i | q) = 0.012$ 和 $1.08$。这意味着对于概率较高的 token（如 $0.9$），受到的约束较少。相反，对于低概率 token，要实现概率的显著增加要困难得多。经验上，我们还观察到裁剪 token 的最大概率约为 $\pi(o_i | q) < 0.2$。

增加 $\varepsilon_{high}$ 的值，为低概率 Token 的增加留出更多空间，这种调整有效地提高了策略的熵，促进了更多样化样本的生成。保持 $\varepsilon_{low}$ 相对较小，因为增加它会最终抑制这些 Token 的概率，导致采样空间的崩塌。

强化学习的训练过程一直以来的宗旨就是要平衡“探索”与“利用”，既要“利用”好已经有的经验来找到好的策略方案，又要去“探索”新的动作及状态，看是否存在更优的策略，上面的熵坍塌现象就会使得训练过程不能更好的去“探索”，无法找到更优策略

当下的强化训练算法都是使得模型输出的 pass@k 转为 pass@1，也就是没有更好的去探索新的可能性，而是让模型从原来可能的输出序列中去稳定生成最优的那一个，本质上没有脱离模型能力范围，这其实与强化学习的思想相矛盾，强化学习是让模型去学习能力范围以外的知识来找更优策略，从这一点来说，GRPO 及以此提升的算法其实都没有更好的让模型去“探索”。

### 动态采样

如果 GRPO 对某一组输出的结果全是错误，或全是正确的情况下，这样组内的每个样本序列计算后的优势都是 0，就会造成这一组的训练不会对模型梯度变化有任何贡献，代表本组训练没意义了。一个问题是训练后期就会面临一个组内很可能全正确的情况，这种情况出现的很自然，因为训练目标就是让模型输出序列的奖励值更高。这就使得后期的训练中有很多的组是没有意义的，白耗费训练资源。

除了耗费训练训练资源之外还会带来一个问题，假设我们有每个批次有 N 个指令来进行 GRPO 训练，这 N 个指令有的容易一些，有的难一些，模型在训练后期很可能在这 N 个指令中有 50% 的指令输出的组内序列全部正确，也就代表有一半的指令训练是没有意义，随着模型训练到后期，每个批次中全为 1 的指令占比会更高。这会使得强化学习训练方差变大，因为我们输入指令让模型产生组内输出的过程实际是生成旧策略模型输出分布的过程，只有指令足够多，旧策略模型输出的序列足够多才能更准确的表示旧策略的输出分布，当有效指令变少的时候，旧策略模型输出的分布也就存在一定的偏移，换句话说就是存在方差，也就是说 GRPO 越到训练后期训练的方差偏移越大。

关于解决这个问题 DAPO 也是简单粗暴，对于批次内生成的组内序列全部正确或错误的指令直接剔除掉，使用新的输出组中不全是错误或正确的指令来补充上，直到补全这个批次。这个方法粗看会影响训练效率，因为你需要让每条指令去生成一个组，再去使用奖励函数判断才能知道输出的组中序列是否全部正确或错误，但是作者实验发现这种方法可以更快的让模型收敛，也就是说可以平衡掉 GRPO 耗费的资源，甚至更优。

### 重新平衡动作：token 级别策略梯度损失

在 GRPO 训练中针对每个指令输出一组的序列样本，这一组训练样本是有长有短的（token 个数），但不管 token 长度多少，组内每个样本是最后求平均值，也就代表每个样本具备相同的权重系数，即每个序列样本对本次训练的重要性是相同的。

比较一下 GRPO 与 DAPO 的外层求和与均值算法。

GRPO： $\frac{1}{G} \sum_{i=1}^G \frac{1}{\vert o_i \vert} \sum_{t=1}^{\vert o_i \vert}$ 

DAPO： $\frac{1}{\sum_{i=1}^G \vert o_i \vert} \sum_{i=1}^G \sum_{t=1}^{ \vert o_i \vert}$ 

举个简单例子，假设 $G$ 大小为 2，一个序列 token 长度为 100，优势值是 1，另一个 token 序列长度为 10，优势值为 -1，为简单起见不考虑裁剪，重要性采样修正的操作。如果使用 GRPO 计算，则最后的结果为 0（此时只代表目标函数值标量为 0，但反向传播是看的梯度，梯度一般不为 0），而如果使用 DAPO 计算则结果为 $\frac{90}{110}$ ，在计算过程中可以感受到如果使用 GRPO 的方法是没有考虑 token 粒度的，也就是说在同一个样本中每个 token 所占的权重系数随着 token 长度的增加而减小，这样的话模型输出序列的长度越长，每个 token 在训练中对应的概率调整的幅度就越小。

从策略梯度定理的角度来看，分别对 GRPO 和 DAPO 的目标函数算梯度公式。

GRPO 梯度公式：

$$
\nabla_{\theta} \mathcal{J}_{\text{GRPO}} = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\vert o_i \vert} \sum_{t=1}^{\vert o_i \vert} A_i r_{i,t} \, \nabla_{\theta} \log \pi_{\theta}(o_{i,t} \mid \cdot) \cdot \mathbf{1} \Bigl\{ \bigl( A_i > 0 \land r_{i,t} \leq 1 + \varepsilon \bigr) \lor \bigl( A_i < 0 \land r_{i,t} \geq 1 - \varepsilon \bigr) \Bigr\} \right]
$$

DAPO 梯度公式：

$$
\nabla_{\theta} \mathcal{J}_{\text{DAPO}} = \mathbb{E} \left[ \frac{1}{\sum_{j} \vert o_j \vert} \sum_{i=1}^{G} \sum_{t=1}^{\vert o_i \vert} A_i r_{i,t} \, \nabla_{\theta} \log \pi_{\theta}(o_{i,t} \mid \cdot) \cdot \mathbf{1} \Bigl\{ \bigl( A_i > 0 \land r_{i,t} \leq 1 + \varepsilon_{\text{high}} \bigr) \lor \bigl( A_i < 0 \land r_{i,t} \geq 1 - \varepsilon_{\text{low}} \bigr) \Bigr\} \right]
$$

从梯度函数中可以更明显的看到 GRPO 中样本级别与 DAPO 中 token 级别计算的区别，从梯度的角度可以更好理解 GRPO 长 token 序列对于每个 token 权重的稀疏化对于梯度计算的实际影响。

另外还有一点是可以从梯度公式上看出来：对于一个每个 token 的梯度贡献来说，由于每个 token 的优势都是相同的，那么主要贡献值其实就来自于这个 token 的重要性采样修正项。而且从梯度公式后面看到（大括号内）一旦该 token 的重要性采样修正项被 clip 裁剪，那么这个 token 实际上对于梯度更新是没有任何作用的，也就是说被裁剪的 token 实际上对于 GRPO 训练没有任何帮助，从这一点上来重新思考 DAPO 去提升 Clip 裁剪上限也是有意义的。

GRPO 这个问题的影响是长 token 序列的训练中每个 token 的修正幅度慢，比如对于长思维链的输出，假如最终推导结果正确，无论思维的过程很优质还是思维过程很差，该组的训练都不会对中间过程的 token 有较大的修正，比如优质内容的 token 采样概率应该增大，很差的思维过程 token 采样概率应该减小。

而 DAPO 的修正方案是对组内全部 token 赋予相同的权重系数，这样就可以从 token 粒度来进行梯度的更新，解决上述问题。

### 长序列的奖惩

这一部分并不是针对 GRPO 公式的修正，而是对于带思考过程的长序列训练的一个 trick，其背景在于一般情况下在进行训练时考虑到模型输出长度限制会对于很长的内容进行截断，对于被截断的内容直接粗暴的给一个惩罚分值（负分），但忽略了可能这个长序列推导过程是正确的只是有点啰嗦，这种直接给负分会使得模型误认为这个推理逻辑是错误的，因此 DAPO 做了个动态的长度奖惩机制，让模型训练有个缓存，可以意识到是长度带来的惩罚而不是逻辑内容错误带来的惩罚，具体动态惩罚函数：

$$
R_{\text{length}}(y) =  \begin{cases}  0, & |y| \leq L_{\text{max}} - L_{\text{cache}} \\ \dfrac{(L_{\text{max}} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\text{max}} - L_{\text{cache}} < |y| \leq L_{\text{max}} \\ -1, & L_{\text{max}} < |y|  \end{cases}
$$

## GSPO

GSPO 全称是 Group Sequence Policy Optimization，组序列策略优化，是 Qwen3 团队的工作，该工作也是对 GRPO 的改进，不同于 DAPO 是应对带思考的长文本场景，GSPO 应对的场景是 MOE 模型的训练。

### 优化背景

GSPO 提到的一个 GRPO 关键性缺陷在于重要性采样修正项使用的粒度不对，GRPO 中是对序列中每个 token 进行的重要性采样，因为动作的粒度是 token，但是奖励却是对整个序列的奖励，这样会造成一种逻辑冲突的问题。其实这个问题其他论文中也讨论过，训练优化的目标是 token 的采样概率，这一点使用 token 作为动作粒度可以理解。但 GRPO 却是对一个序列整体奖惩，优化单元与奖惩单元粒度上不一致时的模型训练就容易出现偏差。

其实上面这个观点如果站在未简化的强化学习算法的角度上来讲（PPO 或者 A2C），优势应该是动作级别的，但是对于 LLM 的自回归输出场景来说动作级别的优势是不太容易计算的，上面介绍 RLHF 的时候也提到了，需要有奖励模型和价值模型。但 GRPO 的简化方案使得每个 token 共享该序列整体的优势值，那粒度上对不齐也是必然的，从理论上来说带来的结果就是无法从 token 粒度上更快速的让模型提升正确 token 的采样概率（如重要性采样思考中提到的如果使用 GRPO 进行 on policy 训练，当重要性采样修正项为 1 时，从目标函数梯度上看 GRPO 的 on policy 训练就成了 SFT 训练，对每个 token 一视同仁）。

除了上面问题外还有一个很重要问题是关于重要性采样修正项本身的，重要性采样的概念是围绕两个分布展开的，涉及到分布，在统计学上是使用大数定律来近似一个分布的，在传统强化学习中也有蒙特卡洛方法（可以理解成随机采样）来近似模型的策略分布。

但 GRPO 中的重要性采样修正项是计算新旧两个模型在某个 token 位置输出一个特定 token 的概率，使用这个特定 token 样本的概率来代替该 token 位置的整个动作空间分布的概率，这样就极容易造成方差偏移的问题（这是上面介绍重要性采样方差公式的具体场景），这个实际 token 的概率很困难随机性原因偏高或偏低（也就是方差），随着输出序列的增长，这种方差是会累积的（因为自回归的特性，每个 token 的采样概率依赖之前 token），前面有 GRPO 目标函数的梯度公式，对组内每个样本 token 会累加求均值，方差累积后求均值是不会消失的。

具体到 MoE 场景，每次 RL 梯度更新后，对于同一个 rollout 样本，新策略 $\pi_{\theta}$ 激活的专家中约有 10% 与旧策略 $\pi_{\theta_{\text{old}}}$ 不同，这也会导致重要性权重波动很大；如果像 GRPO 一样加上 token 级 clip 操作，把因为专家变化而波动的 token 裁掉，噪声反而会继续被放大，这也导致其需要依赖 Routing Replay，保证新旧策略必须激活相同的专家。而 GSPO 则是序列级的 clip，重要性权重波动要在整个句子上取平均，experts 只变化了 10% 左右，在整个句子来看变化的比例并不高，波动性被平均值中和掉了，因此不会裁剪掉那些因为 router 变化导致重要性权重放大的 token，clip 粒度把握得更好。

除了以上的内容，GSPO 算法的出发点其实源自于 MoE 模型与 Dense 模型的区别，在 Dense 模型上使用 GRPO 训练模型不会有很大训崩的概率，但 MoE 模型上训崩概率很大。区别在于新旧两个模型在进行序列中每个 token 概率计算时会被专家路由器路由到不同的专家模块，这样的话会造成重要性采样修正项变得极其不稳定，很容易被裁剪，被裁剪就会使得这个 token 在训练中没有任何的梯度贡献。还有一点使用旧策略模型激活的专家获得的分布来训练当前策略激活的专家梯度，这样训练到最后会使得专家功能混乱，基于以上原因可以得出结论，在 MoE 模型上使用 token 作为优化的动作粒度是不合理的。

### 算法

上面的背景中问题的本质就是奖励粒度与优化的动作粒度不一致问题，很自然的想法就是既然无法将奖励粒度细化到 token 的动作粒度，也就是重要性采样修正不再对应 token 级别，而是对应序列级别。

GSPO 目标函数公式：

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( s_i(\theta) \hat{A}_i, \text{clip} \left( s_i(\theta), 1 - \varepsilon, 1 + \varepsilon \right) \hat{A}_i \right) \right] \\
s_i(\theta) = \left( \frac{\pi_\theta(y_i \vert x)}{\pi_{\theta_{\text{old}}}(y_i \vert x)} \right)^{\frac{1}{|y_i|}} = \exp \left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \vert x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \vert x, y_{i,<t})} \right)
$$

从 GSPO 目标函数公式可以看到与 GRPO 最大的区别就是在重要性采样修正项的处理上，也就是这里的 $s_i(\theta)$，$s_i(\theta)$ 代表的是一个序列中每个 token 的重要性采样累乘后的值进行几何平均，也就是说括号中的 $\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$ （注意 $y$ 的下缀中不再有 $t$ ）是这个序列中每个 token 的重要性采样修正项的乘积结果，然后按照序列中 token 的个数来求几何平均。

GSPO 的关键创新点就是将每个 token 的重要性采样修正改成了序列级别，目的是为了避免单个 token 带来的随机性极值引发较大的方差问题，再进行几何平均后会一定程度上消除这个序列中少量的几个 token 带来的方差偏移，相对于 GRPO 那种将每个 token 的重要性采样修正的形式更优。

举个例子，正确的序列内容情况下：

- 假设每个 token 的重要性采样修正项为：[1.1, 1.1, 1.2, 1.05, 1.1]，因为新旧策略模型没有出现极值概率情况，内容正确，因此优势值为+1，此时 GRPO 在 token 粒度上对每个 token 都进行了鼓励，这没有问题，GSPO 的算数平均值为 1.109，也是对这个序列进行鼓励，在正确序列上二者都没有问题。

错误序列内容情况下：

- 假设每个 token 的重要性采样修正项为：[1.1, 1.1, 0.1, 0.1, 1.1]，第三个和第四个 token 可能是旧策略出现的极值影响造成错误，由于内容错误，优势值为 -1。此时 GRPO 会对第 1,2,5 个惩罚权重为 -1.1，但对错误的 token 惩罚权重只有 -0.1，这显然对于模型训练不利。而 GSPO 的粒度是整个序列，这是这个序列几何平均为 0.425，乘 -1 的优势值后是 -0.425，代表了整个序列是错误的，应该惩罚，但惩罚大小是根据训练中错误 token 及 token 错误情况计算的。

其实到这里 GSPO 的主要内容就已经结束了，后面做了一些补充证明，比如证明 GSPO 在 token 级别上的形式，梯度分析等。

GSPO 公式的梯度：
$$
\begin{aligned} \nabla_\theta \mathcal{J}_{\text{GSPO}}(\theta) &= \nabla_\theta \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)} \left[ \frac{1}{G} \sum_{i=1}^G s_i(\theta) \hat{A}_i \right] \\ &= \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)} \left[ \frac{1}{G} \sum_{i=1}^G s_i(\theta) \hat{A}_i \cdot \nabla_\theta \log s_i(\theta) \right] \\ &= \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} \right)^{\frac{1}{|y_i|}} \hat{A}_i \cdot \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}) \right] \end{aligned}
$$

对于 GSPO 来说，当使用序列为单位做重要性采样时会发现序列中每个 token 中梯度上的贡献是一样的，其实对于就弱化成了交叉熵的梯度，相对于对这个序列进行 SFT 训练（前提优势是正值，值越大这个序列梯度重要性越大）。

| 算法     | 核心思想                                 | 是否使用 Critic | 主要创新点                     | 关键优势                      | 目标问题                     |
| -------- | ---------------------------------------- | --------------- | ------------------------------ | ----------------------------- | ---------------------------- |
| PPO      | 将策略更新限制在信任区域内以稳定学习     | 是              | 截断代理目标函数               | 稳定、鲁棒                    | 通用 RL 对齐 (RLHF)          |
| DPO      | 通过分类损失直接在偏好对上优化策略       | 否              | 将奖励重参数化为最优策略的函数 | 简洁、稳定、无RM/RL 循环      | 通用 RL 对齐 (RLHF)          |
| GRPO     | 使用一组样本的奖励统计量来估计优势       | 否              | 基于组的优势估计               | 内存/计算效率高               | 资源密集型的推理任务         |
| DAPO     | 系统性地应用一套技术来解决大规模 RL 问题 | 否              | Clip-Higher、动态采样等的组合  | 解决特定的训练病理问题        | 规模化训练的可扩展性和稳定性 |
| Dr. GRPO | 从 GRPO 目标中移除已识别的长度和难度偏差 | 否              | 无偏的损失函数形式             | token 效率更高，偏差更小      | GRPO 中的长度/难度偏差       |
| GSPO     | 在序列级别执行重要性采样和截断           | 否              | 序列级重要性采样               | 极高的稳定性，尤其对 MoE 模型 | token 级更新的不稳定性       |
| GMPO     | 使用奖励的几何平均值以对异常值保持鲁棒   | 否              | 目标函数中使用几何平均         | 对异常奖励值的鲁棒性          | 奖励异常值导致的不稳定       |
| GFPO     | 在更新前根据行为指标过滤采样的轨迹       | 否              | 对轨迹进行拒绝采样             | 生成简洁、高效的回答          | 回答长度膨胀问题             |
| LitePPO  | 组合归一化和损失聚合的最佳实践。         | 否              | 对现有技术的有原则配置         | 以最小的复杂性实现高性能      | RL 流程中的过度工程化        |

## 附：KL 散度计算与近似方法

OpenRLHF 源码，KL 散度（Kullback-Leibler Divergence）计算有两种近似实现

```python
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
) -> torch.Tensor:
    log_ratio = log_probs.float() - log_probs_base.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
    return log_ratio
```

## KL 散度计算与近似方法

KL 散度标准计算公式为：

$$
KL(q \| p) = \sum_{x} q(x) \log \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

**性质**：KL 散度非负，且当且仅当 $q(x) = p(x)$ 对所有 $x$ 成立时，取值为 0：
$$
KL(q || p) \geq 0
$$

然而，在实际计算中，直接计算 KL 散度可能非常困难：

- 需要对所有 $x$ 进行求和或积分，计算成本高
- 计算过程中可能涉及大规模概率分布，导致内存消耗过大

因此，通常使用近似方法来计算 KL 散度。

首先定义一个比值 $r$，用于后续推导：

$$
r = \frac {p(x)} {q(x)}
$$

接下来介绍三种常见的 KL 近似计算方法。

### K1 估计器（无偏估计）

$$
k1 = \log \frac{q(x)}{p(x)} = - \log r
$$

K1 估计器是无偏估计：

$$
\mathbb{E}[k1] = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right] = KL(q \| p)
$$

但 K1 估计器的方差较大

- 若 $q(x) < p(x)$，$k1$ 为负值，导致估计值波动大。
- 若 $p(x)$ 和 $q(x)$ 差异大，$k1$ 可能出现较大的数值变化。

### K2 估计器（有偏估计，低方差）

$$
k2 = \frac{1}{2} \left( \log \frac{p(x)}{q(x)} \right)^2 = \frac{1}{2} (\log r)^2
$$

K2 估计器是 **有偏估计**，但其方差较低，且 **偏差在实证中较小**，因为：

- $k2$ 确保所有样本的值都是正数，使其在计算中更加稳定。


### K3 估计器（无偏估计，稳定）

$$
k3 = \frac{p(x)}{q(x)} - 1 - \log \frac{p(x)}{q(x)} = (r - 1) - \log r = (r - 1) + k1
$$

K3 估计器也是无偏估计。证明如下：

已知 $\mathbb{E}[k1] = KL(q || p)$，只需证明 $\mathbb{E}[r - 1] = 0$，即 $\mathbb{E}[r] = 1$。展开计算：

$$
\mathbb{E}[r] = \mathbb{E}_{x \sim q} \left[ \frac{p(x)}{q(x)} \right] = \int q(x) \cdot \frac{p(x)}{q(x)} dx = \int p(x) dx = 1
$$

此外，K3 估计器恒大于等于 0。证明如下：

设函数：

$$
f(x) = x - 1 - \log x\quad (x > 0)
$$

求导数：

$$
f'(x) = 1 - \frac{1}{x}
$$

当 $0 < x < 1$ 时，$f'(x) < 0$；当 $x > 1$ 时，$f'(x) > 0$。即，$f(x)$ 在 $x=1$ 处取最小值，且：

$$
f(1) = 1 - 1 - \log 1 = 0
$$

所以，对于任意 $x > 0$，均有：

$$
f(x) \geq 0
$$

即 **K3 估计器始终为非负数**。

## PPO 算法中的 KL 近似计算

在 PPO 中，Actor 模型的损失函数包含 KL 散度的隐式约束。其损失函数如下：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
$$

其中：

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$


优势函数 $A_t$ 采用 GAE（Generalized Advantage Estimation）递推公式计算：

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

该递推过程从时间步 $T$ 开始逆序计算：

$$
A_T = \delta_T
$$

时间步 $t$ 的 TD 误差（Temporal Difference Error） 计算公式为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

值函数 $V(s_t)$ 由 Critic 模型 估计。

奖励值 $r_t$ 计算如下：
$$
r(s_t, a_t) = \mathbf{I}(s_t = [\text{EOS}]) \times r(x, y) - \beta \text{KL}(t) \\
\text{KL}(t) = \log \frac{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}{\pi^{\text{SFT}}(a_t \mid s_t)}
$$
其中，指示函数 $\mathbf{I}(s_t = [\text{EOS}])$ 表示仅当 $s_t$ 为 句子结束标志（EOS）时取值为 1，否则为 0。因此，在 PPO 中，KL 散度隐含在奖励函数的计算中。

OpenRLHF 库中 PPO 算法可以采用 K1 估计器和 K3 估计器计算 KL 散度。

## GRPO 算法中的 KL 近似计算

在 GRPO（Generalized Reinforcement Policy Optimization） 算法中，KL 散度是显式融入到损失函数中，近似计算采用的是 K3 估计器。

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{\vert o_i \mid} \min \left(\frac{\pi_{\theta}(o_{i,t} \mid \cdot)}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid \cdot)} \hat{A}_{i,t}, \text{clip}\left(\dots\right) \hat{A}_{i,t} \right) \right]  \\
\mathbb{D}_{KL} [\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

## k1, k2, k3估计器的联系

给定两个分布 $q(x)$ 和 $p(x)$，定义比值

$$
r = \frac{p(x)}{q(x)}
$$

以及

$$
\delta = r - 1
$$

当 $p(x)$ 和 $q(x)$ 很接近时，$r$ 会接近 1，从而 $\delta$ 是一个小量。我们关心的是利用泰勒展开来近似下述三个估计器与 KL 散度之间的关系。

泰勒展开
$$
f(\delta)=\log(1+\delta)
$$

并在 $\delta = 0$ 处展开。泰勒展开公式为：

$$
f(\delta) = f(0) + f'(0)\delta + \frac{f''(0)}{2}\delta^2 + O(\delta^3) = 0 + 1\cdot\delta + \frac{-1}{2}\delta^2 + O(\delta^3) = \delta - \frac{1}{2}\delta^2 + O(\delta^3)
$$

### 对 K1 估计器

原定义为：

$$
k1 = -\log r
$$

注意 $r = 1+\delta$，代入上面的泰勒展开：

$$
\log r = \log(1+\delta) \approx \delta - \frac{1}{2}\delta^2
$$

因此，

$$
k1 = -\log r \approx -\left(\delta - \frac{1}{2}\delta^2\right) = -\delta + \frac{1}{2}\delta^2
$$

### 对 K2 估计器

原定义为：

$$
k2 = \frac{1}{2}(\log r)^2
$$

用同样的泰勒展开有：

$$
\log r \approx \delta - \frac{1}{2}\delta^2
$$

平方时：

$$
(\log r)^2 \approx \left(\delta - \frac{1}{2}\delta^2\right)^2
$$

展开平方：

$$
\left(\delta - \frac{1}{2}\delta^2\right)^2 = \delta^2 - \delta^3 + \frac{1}{4}\delta^4
$$

只保留到二阶项，所以忽略 $\delta^3$ 及更高次项，得到：

$$
(\log r)^2 \approx \delta^2
$$

因此，

$$
k2 \approx \frac{1}{2}\delta^2
$$

### 对 K3 估计器

原定义为：

$$
k3 = \delta - \log r
$$

同样代入泰勒展开：

$$
\log r \approx \delta - \frac{1}{2}\delta^2
$$

故：

$$
k3 \approx \delta - \left(\delta - \frac{1}{2}\delta^2\right) = \frac{1}{2}\delta^2
$$

## 总结

$k1$ 的期望即为 KL 散度：

$$
\mathbb{E}[k1] = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right] = KL(q \| p)
$$

$k2,k3$ 近似计算都可以由 $-k1 = \log r \approx \delta - \frac{1}{2}\delta^2$ 的泰勒展开进行近似：

$$
k1 = -\log r \approx  -\delta + \frac{1}{2}\delta^2 \\
k2 = \frac{1}{2}(\log r)^2 \approx \frac{1}{2}\delta^2 \\
k3 = \delta - \log r \approx  \frac{1}{2}\delta^2
$$

其中，$k_1$ 和 $k_3$ 只使用一次泰勒展开，$k_2$ 则先对 $\log r$ 近似，再平方，等于两次近似。

$k3$ 相当于在 $k1$ 的基础上加上了 $\delta$，而：

$$
\mathbb{E}_{x\sim q}[\delta] = \mathbb{E}_{x\sim q} \left[ \frac{p(x)}{q(x)} - 1 \right] = \int q(x) \left( \frac{p(x)}{q(x)} - 1 \right) dx = \int p(x) dx - \int q(x) dx = 1 - 1 = 0
$$

所以 $k3$ 相当于在 $k1$ 的基础上加上一个期望为0的变量，保证了 $k3 \geq 0$，降低方差。