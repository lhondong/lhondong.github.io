# RL 算法与 Infra 面试题

# 算法部分

## 1. 为什么要用 Actor-Critic 而不是纯 Critic？

**推导**

强化学习目标为最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
$$

Policy Gradient Theorem 给出：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{s_t,a_t\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)Q^{\pi}(s_t,a_t)\right]
$$

纯 Actor 用 Monte Carlo return 估计 $Q$，无偏但方差大：

$$
\hat Q_t = G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

引入 Critic 后，用价值函数 $V_\phi(s)$ 做 baseline：

$$
 A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t) 
$$

于是梯度变为：

$$
 \nabla_\theta J(\theta)= \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)A^\pi(s_t,a_t)\right] 
$$

baseline 不改变期望，因为：

$$
\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)b(s)\right] =b(s)\sum_a\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)=b(s)\sum_a\nabla_\theta\pi_\theta(a|s) =b(s)\nabla_\theta 1=0 
$$

纯 Critic 本质是 value-based，例如 DQN 学 $Q(s,a)$ 后取 $\arg\max_a Q(s,a)$。在 LLM 中动作空间是词表，状态是长上下文，纯 Critic 面临三个问题：第一，必须对巨大离散动作空间估计 $Q(s,a)$；第二，最终生成仍要一个策略分布；第三，对语言这种多峰输出，$\arg\max Q$ 容易丢掉多样性和可采样性。

**最终结论**

Actor-Critic 的 Actor 负责直接建模策略 $\pi_\theta(a|s)$，Critic 负责估计价值或 advantage 降低方差。相比纯 Critic，它更适合大动作空间、随机策略、连续或高维动作，以及 LLM token 生成。

**面试话术**

“我会用 Actor-Critic，是因为 LLM 的动作空间是词表，纯 Critic 要估计每个 token 的 $Q(s,a)$，工程和统计上都很难。Policy gradient 可以直接优化生成策略，但 Monte Carlo return 方差很大，所以引入 Critic 或 baseline 估计 $V(s)$，把 $Q$ 换成 $A=Q-V$。这个 baseline 不引入偏差，因为 $\mathbb{E}[\nabla\log\pi b(s)]=0$，但能显著降方差。PPO 是典型 Actor-Critic，GRPO 则进一步去掉显式 Critic，用组内 reward 归一化做 baseline。”

## 2. KL 散度和交叉熵、MLE 的关系？

**推导**

真实数据分布记为 $p(x)$，模型分布为 $q_\theta(x)$。交叉熵：

$$
 H(p,q_\theta)=-\mathbb{E}_{x\sim p}\log q_\theta(x) 
$$

熵：

$$
 H(p)=-\mathbb{E}_{x\sim p}\log p(x) 
$$

KL 散度：

$$
 D_{\mathrm{KL}}(p\|q_\theta)= \mathbb{E}_{x\sim p}\log\frac{p(x)}{q_\theta(x)} =\mathbb{E}_{p}\log p(x)-\mathbb{E}_{p}\log q_\theta(x) 
$$

所以：

$$
 D_{\mathrm{KL}}(p\|q_\theta)=H(p,q_\theta)-H(p) 
$$

训练数据固定时 $H(p)$ 与 $\theta$ 无关，因此最小化交叉熵等价于最小化 $D_{\mathrm{KL}}(p\|q_\theta)$。

MLE 最大化似然：

$$
 \max_\theta \sum_{i=1}^{N}\log q_\theta(x_i) 
$$

等价于最小化负对数似然：

$$
 \min_\theta -\frac{1}{N}\sum_i \log q_\theta(x_i) 
$$

当样本来自 $p_{\mathrm{data}}$ 时，这正是经验交叉熵。

**最终结论**

MLE = 最小化 NLL = 最小化经验交叉熵 = 最小化 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|q_\theta)$，差别只在是否显式写出常数项 $H(p_{\mathrm{data}})$。

**面试话术**

“交叉熵可以拆成真实分布熵加 KL：$H(p,q)=H(p)+D_{KL}(p\|q)$。训练集固定时 $H(p)$ 是常数，所以最小化交叉熵等价于最小化从数据分布到模型分布的 forward KL。MLE 最大化 $\sum\log q_\theta(x_i)$，也就是最小化 NLL，形式上就是经验交叉熵。因此语言模型预训练的 next-token CE 本质上就是 MLE。”

## 3. 不同 RL 场景应该如何设计 Reward？

**推理**

Reward 设计要先判断任务是否可验证、反馈是否稀疏、是否有安全约束、是否容易 reward hacking。

可验证任务使用 rule-based reward。例如数学答案可 exact match 或符号等价，代码可单元测试，格式任务可正则或 JSON schema。RLVR 的优点是稳定、可扩展、低主观性，但缺点是只奖励可验证部分，容易出现“答案对但过程错”或利用测试漏洞。

开放式任务用 reward model 或 judge。偏好、写作、安全、摘要通常无法客观验证，需要 pairwise RM、LLM-as-judge、多维 rubric。风险是 reward model misspecification，模型可能学会讨好 judge。

长链推理任务常用 outcome reward 加过程辅助。Outcome reward 只在最终答案给分：

$$
R(y)=\mathbb{1}[\mathrm{verify}(y)=\mathrm{true}]
$$

过程 reward 可分解为：

$$
 R(\tau)=\sum_{t=1}^{T}\gamma^{t-1}r_t 
$$

其中 $r_t$ 可来自步骤验证器、工具执行结果、编译错误、测试通过数等。

多目标任务要做 reward 加权：

$$
 R=\lambda_1 R_{\mathrm{task}}+\lambda_2 R_{\mathrm{format}}+\lambda_3 R_{\mathrm{safety}}-\lambda_4 C_{\mathrm{length}} 
$$

但权重需要归一化，否则某一项会主导训练。

**最终结论**

Reward 设计的优先级是：可验证优先、任务成功优先、过程信号辅助、长度和格式只做约束或小权重、必须做反作弊和 held-out verifier。

**面试话术**

“我会先判断任务是不是可验证。数学、代码、检索这类任务优先用 rule-based verifier，因为信号客观且可扩展。开放问答或偏好任务再用 reward model 或 LLM judge。对于长 CoT，我不会只看最终答案，还会考虑过程奖励、工具反馈和长度惩罚。但 reward 一定要防 hacking，比如单元测试要隐藏、格式奖励不能盖过任务奖励、judge 要做校准。最终目标是让 reward 对齐真实任务成功，而不是奖励模型表面特征。”

## 4. 如何理解 RL 中的 importance sampling / rejection sampling 等 Monte Carlo 方法？

**推导**

Monte Carlo 用采样估计期望：

$$
 \mathbb{E}_{x\sim p}[f(x)]\approx \frac{1}{N}\sum_{i=1}^{N}f(x_i),\quad x_i\sim p 
$$

Importance sampling 用 proposal $q(x)$ 采样，修正分布差异：

$$
 \mathbb{E}_{x\sim p}[f(x)] =\int f(x)p(x)dx =\int f(x)\frac{p(x)}{q(x)}q(x)dx =\mathbb{E}_{x\sim q}\left[w(x)f(x)\right] 
$$

其中：

$$
 w(x)=\frac{p(x)}{q(x)} 
$$

在 PPO/GRPO 中，rollout 来自 old policy，训练当前 policy，因此：

$$
 r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)} =\exp(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}) 
$$

这就是重要性权重。

Rejection sampling 从 proposal $q(x)$ 采样后按概率接受：

$$
 \alpha(x)=\frac{p(x)}{M q(x)},\quad Mq(x)\ge p(x) 
$$

LLM 中的 best-of-N、用 verifier 过滤正确答案、DeepSeek-R1 中用 rejection sampling 生成 SFT 数据，都属于这类思想。

**最终结论**

Importance sampling 是“样本来自旧分布，但想估计新分布期望”的修正；rejection sampling 是“先广泛生成，再筛选高质量样本”。两者都是 Monte Carlo 估计思想在 RL 和 LLM post-training 中的具体应用。

**面试话术**

“Monte Carlo 是用采样平均估计期望。importance sampling 解决 off-policy 问题，比如 rollout 是 old policy 采的，但更新 current policy，所以乘上 $\pi_\theta/\pi_{\theta old}$。PPO 的 ratio 就是这个权重。rejection sampling 则是先从 proposal 生成，再按 verifier 或 reward 筛选，DeepSeek-R1 用它构造高质量 SFT 数据。区别是 IS 保留样本但加权，rejection sampling 丢弃低质量样本。”

## 5. PPO / GRPO 的 advantage 是怎么算的，为什么要减去 baseline，这里一定要除以 std 吗？

**推导**

PPO 通常使用 GAE：

$ \delta_t=r_t+\gamma V(s_{t+1})-V(s_t) $ 

$$
 \hat A_t^{\mathrm{GAE}(\gamma,\lambda)} =\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l} 
$$

GRPO 对同一个 prompt 采样 $G$ 个回答，得到 reward $R_1,\dots,R_G$，组内均值和标准差为：

$$
 \mu=\frac{1}{G}\sum_{i=1}^{G}R_i,\quad \sigma=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(R_i-\mu)^2} 
$$

GRPO advantage：

$$
 \hat A_i=\frac{R_i-\mu}{\sigma+\epsilon} 
$$

减 baseline 的原因是降低方差且不改变期望。设 baseline $b(s)$ 与动作无关：

$$
 \mathbb{E}_{a\sim\pi}[\nabla\log\pi(a|s)b(s)] =b(s)\nabla\sum_a\pi(a|s)=0 
$$

是否必须除以 std？不必须。除以 std 是尺度归一化，能稳定学习率和梯度尺度，但会改变不同 prompt 之间的权重。若组内 std 很小，归一化会放大噪声；若全对或全错，std 为 0，此时 advantage 应置 0 或过滤。DAPO 的 dynamic sampling 就是过滤全对/全错组，保留有非零 advantage 的样本。

**最终结论**

PPO advantage 常来自 GAE；GRPO advantage 来自同 prompt 多回答 reward 的组内标准化。减均值是必要的方差控制思想，除 std 是常用稳定技巧但不是理论必需。

**面试话术**

“PPO 一般用 GAE，先算 TD error $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，再做指数加权。GRPO 没有 critic，所以对同一个 prompt 采样多个回答，用组内 reward 的均值做 baseline，$A_i=(R_i-\mu)/\sigma$。减均值不会引入偏差，因为 baseline 和动作无关，期望梯度为 0。除以 std 不是必须的，它是尺度归一化技巧；std 太小会放大噪声，所以实践里要加 epsilon、过滤全对全错组，或者用 batch-level normalization。”

## 6. RL training 和 test-time scaling 各自是如何 explore 的？

**推理**

RL training 的探索来自训练时采样。模型按 $\pi_\theta(y|x)$ 生成多个样本，reward/verifier 选择高 reward 区域，策略更新提高这些轨迹概率。探索强度由 temperature、top-p、group size、prompt 难度、entropy、KL、长度上限决定。

Test-time scaling 不更新参数，通过更多推理计算提升命中率。典型形式有 best-of-N、self-consistency、tree search、tool search、verifier rerank：

$$
 P(\mathrm{success\ in\ }N)=1-(1-p)^N 
$$

当单样本成功率为 $p$ 且样本独立时，采样 $N$ 次至少一次成功概率如上。但真实 LLM 样本相关，收益递减。

RL training 改变分布本身，目标是提高 pass@1 或 pass@K 曲线；test-time scaling 是在固定分布上多抽样，提高 pass@K。

**最终结论**

训练时探索是“采样-奖励-更新参数”；测试时探索是“固定参数-多采样/搜索/验证”。前者改变模型能力和分布，后者消耗推理算力换更高成功率。

**面试话术**

“RL training 的探索发生在 rollout 阶段，模型用一定温度对同一问题采多个答案，再通过 reward 把概率质量往好轨迹上推。test-time scaling 不更新模型，而是采更多次、做 self-consistency、best-of-N 或 tree search。公式上如果单次成功率是 $p$，独立采 $N$ 次成功率是 $1-(1-p)^N$。所以 training scaling 是改变策略分布，test-time scaling 是在固定分布上用更多计算找好样本。”

## 7. PPO 是如何 clip 的，为什么要取 min，不 clip 会怎么样，CISPO 是怎么做的？

**推导**

PPO clipped objective：

$$
 L^{\mathrm{CLIP}}(\theta)= \mathbb{E}_t\left[ \min\left( r_t(\theta)\hat A_t, \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t \right) \right] 
$$

其中：

$$
 r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)} 
$$

当 $\hat A_t>0$，希望提高该动作概率，但若 $r_t>1+\epsilon$，clip 后目标不再继续增大，防止过大更新。此时取 min 会选择 clipped 项。

当 $\hat A_t<0$，希望降低该动作概率，但若 $r_t<1-\epsilon$，继续降低会导致过大更新。因为乘以负 advantage 后大小关系反转，取 min 仍会选择保守项。

不 clip 时：

$$
 L^{\mathrm{PG}}=\mathbb{E}[r_t\hat A_t] 
$$

如果多 epoch 复用同一批 rollout，当前 policy 会偏离 old policy，$r_t$ 可能极端，导致梯度爆炸、KL 飙升、策略塌缩。

CISPO 不用 PPO 的 $\min(rA,\mathrm{clip}(r)A)$ 形式，而是直接 clip importance weight，并把其作为 stop-gradient 权重乘在 log-prob policy gradient 上。TorchRL 文档给出的核心形式为：

$$
 \mathrm{loss}=-\mathrm{clip}(w,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\cdot A 
$$

更完整可写为：

$$
 L_{\mathrm{CISPO}}(\theta)= -\mathbb{E}\left[ \mathrm{sg}(\mathrm{clip}(r_t,c_{\min},c_{\max}))\hat A_t\log\pi_\theta(a_t|s_t) \right] 
$$

这样极端 token 仍有 log-prob 梯度，只是权重被截断。

**最终结论**

PPO clip 是对 surrogate objective 做悲观截断，取 min 是同时处理正负 advantage 的保守下界。不 clip 会导致 off-policy ratio 失控。CISPO 改成直接截断 IS 权重并保留梯度通路，目标是减少 PPO 对关键 token 的梯度抑制。

**面试话术**

“PPO 的 ratio 是 $\pi_\theta/\pi_{old}$。目标是 $\min(rA,\mathrm{clip}(r)A)$。取 min 是为了拿一个 pessimistic bound：正 advantage 时防止把好动作概率提高太多，负 advantage 时防止把坏动作概率降低太多。不 clip 的话，多 epoch 复用 rollout 会让 ratio 失控，KL 爆掉。CISPO 的思路不一样，它直接 clip importance weight，并用 stop-gradient 的 clipped weight 乘 log-prob，所以极端但关键的 token 不会像 PPO 那样完全没梯度。”

## 8. GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了 KL 散度？

**推导**

GRPO 目标通常包含 reward surrogate 和 reference KL 惩罚：

$$
 J_{\mathrm{GRPO}}(\theta)= \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min(r_{i,t}\hat A_i,\mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i) -\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}) \right) \right] 
$$

token 级 reference KL 可写为：

$$
 D_{\mathrm{KL}}(\pi_\theta(\cdot|s_t)\|\pi_{\mathrm{ref}}(\cdot|s_t)) =\sum_v \pi_\theta(v|s_t) \log\frac{\pi_\theta(v|s_t)}{\pi_{\mathrm{ref}}(v|s_t)} 
$$

实际训练中全词表 KL 昂贵，常用采样 token 上的估计或近似 KL。常见 k1 近似为：

$$
 \log\pi_\theta(a_t|s_t)-\log\pi_{\mathrm{ref}}(a_t|s_t) 
$$

更稳定的 k3 估计为：

$$
 \exp(\Delta)-\Delta-1,\quad \Delta=\log\pi_{\mathrm{ref}}(a_t|s_t)-\log\pi_\theta(a_t|s_t) 
$$

GRPO 加 KL 是为了防止 policy 远离 SFT/reference，降低语言退化、格式漂移和 reward hacking。

DAPO/GSPO 去掉或弱化显式 KL 的原因是：在数学/代码 RLVR 中，强 KL 会限制探索和长 CoT 形成；PPO clip、动态采样、sequence-level ratio、本身已提供 trust-region 风格约束；reference KL 会消耗显存和算力，需要 reference model 或 logprob。

**最终结论**

GRPO 加 KL 是为了限制策略漂移；DAPO/GSPO 去掉 KL 是因为在可验证推理任务中，KL 可能抑制探索，且新的 clipping/采样机制已经承担一部分稳定作用。

**面试话术**

“GRPO 加 reference KL 是为了不让 RL 把模型推离 SFT 分布太远，避免语言质量下降和 reward hacking。KL 可以按全词表算，也可以对采样 token 用 logprob 近似。DAPO、GSPO 这类后续方法在 RLVR 场景里倾向去掉或弱化 KL，因为数学代码任务的 reward 客观，强 KL 会限制探索和长 CoT，而且它们通过 clip-higher、dynamic sampling、sequence-level ratio 来控制更新幅度。工程上少一个 reference model 也能省显存和通信。”

## 9. 在 LLM 训练时，如果不小心多 All Reduce 了几次 loss，会发生什么？

**推理**

设每个 rank 的 local loss 为 $L_i$，world size 为 $n$。

如果正确做平均：

$$
 L=\frac{1}{n}\sum_{i=1}^{n}L_i 
$$

若误做一次 SUM all-reduce 但没有除以 $n$：

$$
 L'=\sum_i L_i=nL 
$$

梯度也会放大：

$$
 \nabla L'=n\nabla L 
$$

相当于 learning rate 放大 $n$ 倍。若重复 $k$ 次 SUM all-reduce，可能放大 $n^k$ 倍。

如果每次 all-reduce 后都做 AVG，且所有 rank 已持有同一个平均 loss，则再次 AVG 理论上数值不变：

$$
 \frac{1}{n}\sum_i L=L 
$$

但仍有额外通信和浮点误差。如果是用于 logging 的 detached loss，不影响训练；如果参与 backward，则直接改变梯度尺度。

**最终结论**

多 all-reduce loss 的影响取决于 reduce op 和是否参与 backward。SUM 会把 loss/gradient 放大，AVG 多次通常数值不变但浪费通信。最危险的是把用于日志的全局 loss 又拿去 backward。

**面试话术**

“如果 loss 参与反传，多 AllReduce 会改变梯度尺度。比如 SUM 后没除 world size，loss 和梯度放大 $n$ 倍，重复 k 次就是 $n^k$，等价于学习率暴涨，容易 loss spike 或训练崩。如果 all-reduce 是 AVG，重复做数值上基本不变，但浪费通信。实际工程里我会区分 train loss tensor 和 detached logging loss，避免把日志统计结果用于 backward。”

## 10. DPO 的 reward 是什么，会不会 reward hacking，如何解决？

**推导**

DPO 从 KL-regularized RLHF 问题出发：

$$
 \max_{\pi}\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)] -\beta D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)) 
$$

最优策略满足：

$$
 \pi^*(y|x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) 
$$

反解 reward：

$$
 r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x) 
$$

同一个 prompt 下比较两个回答时 $Z(x)$ 抵消，所以 DPO 的 implicit reward 可写为：

$$
 r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)} 
$$

DPO loss：

$$
 \mathcal{L}_{\mathrm{DPO}} =-\mathbb{E}_{(x,y_w,y_l)} \log\sigma\left( \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} -\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) 
$$

DPO 会不会 reward hacking？会，但形式不同。它没有显式 RM 可被在线攻击，但会过拟合偏好数据偏差，例如长度偏好、格式偏好、安全模板化、chosen/rejected 噪声。

**解决**

数据层面：清洗偏好对、控制长度、hard negative、多轮审计。
目标层面：加 reference regularization、IPO/SimPO/KTO 等变体、label smoothing。
评估层面：held-out preference set、真实用户评估、reward model ensemble、OOD benchmark。
训练层面：早停、低学习率、混合 SFT loss、防止过度偏离 reference。

**面试话术**

“DPO 的 reward 是隐式的，$r_\theta=\beta\log(\pi_\theta/\pi_{ref})$，同一个 prompt 下归一化常数会抵消，所以可以直接用 preference pair 做 logistic loss。它不是完全没有 reward hacking，只是没有单独 reward model 被在线优化。它仍然会学偏好数据里的 spurious pattern，比如更长、更像模板、更安全但没用的回答。解决上要做偏好数据去偏、长度控制、reference regularization、早停和独立评估。”

## 11. 有哪些解决 MoE 训推不一致问题的算法，各自是什么原理？

**推理**

MoE 训推不一致主要来自 routing、batch composition、expert capacity、并行策略、精度和确定性差异。训练时 large batch 和 expert parallel 通信让 token 分布更平滑；推理时 batch 小且动态 batching，路由负载和 expert 命中不同，可能造成输出差异或吞吐不稳定。

典型算法与机制：

Auxiliary load-balancing loss：在训练时惩罚 expert 使用不均衡，常见形式为：

$$
 L_{\mathrm{aux}}=\alpha N\sum_{i=1}^{N} f_i p_i 
$$

其中 $f_i$ 是分到 expert $i$ 的 token 比例，$p_i$ 是 router 给 expert $i$ 的平均概率。优点是简单，缺点是可能损伤主任务性能。

Auxiliary-loss-free balancing：DeepSeek-V3 使用动态 bias 调整 expert 负载，避免把负载均衡作为额外 loss 强行反传进主目标。核心思想是负载高的 expert 降低 routing bias，负载低的 expert 提高 bias。

Router z-loss / entropy regularization：限制 router logits 过大，防止路由过度尖锐，提高泛化和稳定性。

Node-limited routing / expert locality：限制 token 只路由到少数节点内 expert，降低跨节点通信，提高训练和推理拓扑一致性。

Expert capacity 与 token dropping 策略：训练中 capacity factor、drop/pad 策略要和推理一致。推理通常不希望 drop token，因此训练若大量 drop，会造成不一致。

Sequence-level RL objective：GSPO 把 token-level ratio 改成 sequence-level ratio，Qwen 团队报告其能稳定 MoE RL 训练，因为 token-level ratio 在 MoE 中方差更大。

确定性与 batch invariance：推理时 dynamic batching 会改变 kernel reduction 顺序和 routing batch 统计。batch-invariant kernels 或固定调度能减少同一请求在不同 batch 下输出不同。

**最终结论**

解决 MoE 训推不一致不是单一算法，而是 routing loss、aux-free balancing、容量策略、拓扑约束、sequence-level RL、确定性推理和 serving 调度共同处理。

**面试话术**

“MoE 训推不一致主要是 router 在训练大 batch 和推理动态 batch 下看到的负载、容量和通信环境不同。常见方案包括 auxiliary load balance loss，但它可能影响主任务；DeepSeek-V3 的 aux-loss-free balancing 用动态 bias 调负载，避免把均衡目标写进 loss；还有 router z-loss、node-limited routing、capacity 策略一致化。到 RL 阶段，GSPO 用 sequence-level ratio 替代 token-level ratio，也能缓解 MoE token ratio 高方差问题。”

## 12. RL 训练时，group size / learning rate / ppo epoch / generation length 如何设置？

**推理**

Group size 决定组内 advantage 估计质量。GRPO 中：

$ \hat A_i=\frac{R_i-\mu_G}{\sigma_G} $ 

$G$ 越大，$\mu_G,\sigma_G$ 估计越稳，但 rollout 成本线性增加。常见从 4、8、16 起步。数学/代码任务若 reward 稀疏，建议 8 到 16；如果全对/全错多，则配合 dynamic sampling。

Learning rate 取决于模型大小和 KL/clip。大模型通常 $10^{-6}$ 量级，小模型可到 $5\times10^{-6}$ 或更高。有效更新尺度近似为：

$$
 \Delta\theta\propto \eta \cdot E \cdot \|\nabla L\| 
$$

其中 $\eta$ 是 learning rate，$E$ 是 PPO epoch。因此 epoch 增加时 LR 要更保守。

PPO epoch 控制同一批 rollout 重用次数。多 epoch 提高样本效率，但使 policy 更 off-policy，ratio 偏离更大。LLM RL 常用 1 到 4，GRPO/RLVR 中很多实践倾向较少 epoch 以避免过拟合旧样本。

Generation length 必须覆盖正确推理长度，但过长会提高长尾、显存和 reward noise。设置应基于答案长度分布的 P95/P99，并配合 overlong penalty：

$$
R'=R-\lambda\max(0, |y|-L_{\mathrm{safe}}) 
$$ 

**最终结论**

初始配置建议：group size 8 或 16，LR 从 $1e-6$ 到 $5e-6$ 网格搜索，PPO epoch 1 到 2 起步，generation length 设到训练任务 P95 以上并做 overlong penalty。

**面试话术**

“我会把这几个超参看成一个整体。group size 影响 advantage 方差，reward 稀疏就用 8 或 16，并用 dynamic sampling 去掉全对全错组。LR 通常是 $10^{-6}$ 量级，epoch 越多 LR 越要小，因为同一批 rollout 会变 off-policy。PPO epoch 我一般从 1 到 2 开始。generation length 要覆盖任务 P95 推理长度，但不能无限放大，否则长尾和显存会爆，所以要配 length penalty 或 overlong filtering。”

## 13. 相比 GRPO，Dr.GRPO / DAPO / GSPO / CISPO / SAPO / DPPO / MaxRL / SimKO 是如何改进的，各自又有什么缺？


**逐项结论**

Dr.GRPO：核心是重新审视 GRPO 的实现细节和偏差，常讨论 length normalization、advantage normalization、KL、response-level vs token-level aggregation 等问题。优点是指出 GRPO 训练中很多“效果”可能来自实现细节。缺点是更像诊断和修正集合，不是单一统一算法。

DAPO：在 GRPO 上加四个关键技巧：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping。优点是长 CoT RL 更稳定、样本更有效。缺点是超参更多，dynamic sampling 会改变数据分布，overlong shaping 设计不好会惩罚必要长推理。

GSPO：把 token-level importance ratio 换成 sequence-level likelihood ratio：

$$
 s_i(\theta)= \exp\left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})} {\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})} \right) 
$$

然后做 sequence-level clipping。优点是 reward 是 sequence-level，优化粒度一致，MoE RL 更稳定。缺点是对序列内关键 token 的细粒度 credit assignment 弱。

CISPO：直接 clip importance weight，并保留 log-prob 梯度。优点是减少 PPO-style clipping 对关键低概率 token 的梯度屏蔽。缺点是稳定性依赖权重截断和 stop-gradient 设计，理论 trust region 不如 PPO min objective 直观。

SAPO：Soft Adaptive Policy Optimization 用平滑温控 gate 替代硬 clipping：

$$
 g(r,A;\tau)\in(0,1),\quad L\approx -g(r,A;\tau) r A 
$$

优点是避免硬截断导致梯度突然为 0，兼顾 token-adaptive 和 sequence-coherent。缺点是温度/gate 设计引入新超参，公开实践仍较新。

DPPO：如果指 Distributed PPO，是把 PPO 的采样和更新并行化，提升吞吐；如果指 Decoupled PPO，是把行为策略、近端策略、目标策略分离，缓解 stale/off-policy。缺点是实现复杂，staleness 控制难。

MaxRL：从最大化 pass@1 的 expected reward 转向近似最大似然地提高正确 rollout 分布，目标更贴近 pass@K/test-time scaling。优点是改善 pass@K 和采样效率。缺点是需要更多采样计算，适合可验证二元反馈任务。

SimKO：Simple Pass@K Policy Optimization，直接面向 pass@K，缓解 RLVR 把概率质量过度集中到单一路径的问题。优点是探索更好，pass@K 提升。缺点是可能牺牲 pass@1 或需要在探索和利用之间调权重。

**面试话术**

“这些方法大多是在修 GRPO 的三个问题：ratio 粒度、clip 梯度抑制、探索塌缩。DAPO 是工程化增强，动态采样、clip-higher、token-level loss 和过长惩罚；GSPO 把 ratio 提到 sequence 级，解决 reward 和优化粒度不一致，尤其稳定 MoE；CISPO 直接 clip IS weight，避免 PPO min 让关键 token 没梯度；SAPO 用 soft gate 替代 hard clip；MaxRL 和 SimKO 则从目标上转向 pass@K 或最大似然，解决只优化 pass@1 导致多样性下降的问题。缺点是这些方法都引入新超参或新假设，没有免费午餐。”

## 14. TRPO / DPPO / AReaL 是如何用 trust region 约束 RL objective 的？

**推导**

TRPO 原始约束优化：

$$
 \max_\theta \mathbb{E}_{s,a\sim\pi_{\theta_{\mathrm{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\mathrm{old}}}(a|s)} A_{\theta_{\mathrm{old}}}(s,a) \right] 
$$

约束：

$$
 \mathbb{E}_{s\sim\pi_{\theta_{\mathrm{old}}}} \left[ D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}(\cdot|s)\|\pi_\theta(\cdot|s)) \right]\le \delta 
$$

TRPO 用 KL 二阶近似和共轭梯度求自然梯度方向：

$$
 \max_x g^Tx,\quad \frac{1}{2}x^THx\le\delta 
$$

解的方向为：

$$
 x^*=\sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g 
$$

PPO/DPPO 用 clip 或 KL penalty 近似 trust region：

$$
 r_t\in[1-\epsilon,1+\epsilon] 
$$

Decoupled PPO 的关键思想是把 off-policy correction 和 trust-region constraint 解耦，避免同一个 ratio 同时承担校正和约束导致不稳定。

AReaL 是异步 RL 系统，rollout policy 可能滞后当前训练 policy。它用 staleness-aware PPO 处理 stale samples。可抽象为对样本引入版本差：

$$
 d=\mathrm{version}(\theta_{\mathrm{train}})-\mathrm{version}(\theta_{\mathrm{rollout}}) 
$$

并用 importance ratio/KL/clip 或 staleness weighting 限制过旧样本影响。

**最终结论**

TRPO 是显式 KL constrained optimization；PPO/DPPO 是用 clipping 或解耦 ratio 近似 trust region；AReaL 在异步场景中额外把 policy staleness 纳入 trust-region 约束。

**面试话术**

“TRPO 是最标准的 trust region：最大化 surrogate，同时约束 old policy 和 new policy 的平均 KL 小于 $\delta$，然后用自然梯度近似求解。PPO 把这个硬 KL 约束简化成 ratio clipping。DPPO 或 decoupled PPO 进一步把重要性采样校正和近端约束拆开，减少 ratio 一物两用的问题。AReaL 的特殊点是异步 rollout 有 staleness，所以 trust region 不只看 KL，还要控制样本版本差和过旧样本权重。”

## 15. RL 能否拓展 LLM 的能力边界？

**推理**

需要区分 pass@1 和 pass@K。若 RL 只是把 base model 已经能采到的正确轨迹概率提高，则 pass@1 上升，但大 $K$ 下 base model 也能采到，能力边界未必扩展。

能力边界可用 pass@K 曲线判断：

$$
 \mathrm{pass@K}=P(\exists i\le K,\ y_i\ \mathrm{correct}) 
$$

如果 RL 后 pass@1 上升但大 K 收敛到 base，同样本空间没有扩大，只是重排概率质量。若 RL 后在大 K 仍高于 base，说明可达正确解集合扩大或采样分布更有效。

ProRL 论文主张 prolonged RL 能在数学、代码、STEM、逻辑等任务上持续提升，并观察到 novel reasoning patterns，支持 RL 在足够稳定和足够长训练下可能拓展边界。也有论文持谨慎观点，认为很多 RLVR 主要激发 base latent capability。

**最终结论**

RL 可以拓展 LLM 能力边界，但不是自动发生。短程 RL 经常只是提高 pass@1；长程、稳定、有足够探索和高质量 verifier 的 RL 更可能扩大 pass@K 边界。

**面试话术**

“我不会简单说 RL 一定拓展能力边界。要看 pass@K 曲线。如果 RL 只提高 pass@1，但大 K 下和 base 持平，那更多是把已有正确轨迹提权。若大 K 也明显超过 base，才说明边界可能扩大。ProRL 这类工作给了正面证据，说明 prolonged、稳定、多任务 RL 可以产生新的推理模式。但前提是 verifier 可靠、探索充分、训练没有过早熵塌缩。”

## 16. 结合 ProRL 等工作，谈谈如何 scale RL 训练边界？

**推理**

Scale RL 训练边界不是只加 GPU，而是同时 scale 五件事。

数据规模和难度：训练题要覆盖 base 当前“临界区”。太简单全对无梯度，太难全错也无梯度。应维护难度 curriculum，使组内 reward 方差非零：

$$
 0<\mathrm{Var}(R_1,\dots,R_G)<\infty 
$$

采样计算：增大 group size、rollout 数、temperature、多样化 prompt，提升探索。

训练时长：ProRL 的核心是 prolonged RL，即稳定训练数千步后仍持续提升。关键是避免 KL 过强、熵塌缩、长度爆炸和 reward overfit。

算法稳定性：DAPO dynamic sampling、GSPO sequence-level ratio、SAPO soft gate、MaxRL/SimKO pass@K objective 都是在扩展训练稳定区。

系统吞吐：rollout 是瓶颈，需要 vLLM/SGLang、异步 AReaL/slime、KV cache 优化、部分 rollout、动态负载均衡。

**最终结论**

Scale RL 边界 = scale 可学习数据、scale 探索、scale 稳定训练时长、scale 可验证 reward、scale rollout 系统吞吐。

**面试话术**

“ProRL 给我的启发是，RL 不是训几百步就结束，而是要把稳定训练时间拉长。要做到这一点，数据难度要在模型临界区，group 内不能全对全错；采样要足够多，保证探索；算法上要防 KL 过强、熵塌缩和过长输出；系统上 rollout 吞吐要跟上。也就是说 scale RL 不是只加卡，而是数据、采样、算法稳定性和系统吞吐一起 scale。”

## 17. OPD 相比于传统 RL / SFT 的改进，有哪些 OPD 的应用？

**推导**

SFT 在离线 teacher trajectory 上训练：

$$
 L_{\mathrm{SFT}}=-\mathbb{E}_{(x,y^*)\sim D}\sum_t\log\pi_\theta(y_t^*|x,y_{<t}^*) 
$$

问题是 off-policy exposure bias：训练状态来自 teacher，推理状态来自 student。

RL 在 student trajectory 上训练：

$$
 L_{\mathrm{RL}}=-\mathbb{E}_{y\sim\pi_\theta}[(R(y)-b)\sum_t\log\pi_\theta(y_t|x,y_{<t})] 
$$

问题是 reward 稀疏、方差大。

OPD 让 student 先 on-policy 生成 $y\sim\pi_\theta$，再让 teacher 在这些 student states 上提供 token-level 分布 $q_T(\cdot|s_t)$：

$$
 L_{\mathrm{OPD}}= \mathbb{E}_{y\sim\pi_\theta} \sum_t D_{\mathrm{KL}}(q_T(\cdot|s_t)\|\pi_\theta(\cdot|s_t)) 
$$

这样同时解决 SFT 的 off-policy 状态不匹配和 RL 的稀疏 reward 问题。

应用：推理模型蒸馏、agent/tool-use 轨迹修正、speculative draft model 训练、长上下文 reasoning、模型压缩、多任务增强、黑盒 teacher 的 logit-free OPD、隐私 OPD。

**最终结论**

OPD 是“on-policy 采样 + dense teacher supervision”。它比 SFT 更贴近学生推理分布，比 RL 信号更密集，但受 teacher 质量、teacher 成本和 token-level imitation 上限约束。

**面试话术**

“SFT 的问题是学生训练时看 teacher 轨迹，推理时却走自己的轨迹；RL 的问题是 reward 太稀疏。OPD 折中：让学生自己生成轨迹，在学生真实状态上请 teacher 给 token-level 或 chunk-level 指导，然后最小化 KL。它适合 reasoning distillation、agent 工具调用、speculative draft model 和模型压缩。缺点是 teacher 成本高，而且如果只做 token imitation，可能学不到真正的全局规划。”

## 18. LLM 推理能力是在哪一个训练阶段产生的？

**推理**

推理能力不是单一阶段突然产生，而是分层形成。

预训练阶段学习语言、知识、程序模式、数学模式和隐式推理电路，是能力的主体来源。没有足够强的 base，后续 RL 很难凭空创造复杂能力。

SFT/指令微调阶段把能力格式化为可用行为，例如按步骤解题、遵循指令、输出结构化答案。

RL/RLVR 阶段改变策略分布，提高高 reward 推理轨迹概率，鼓励长 CoT、自我检查、反思、工具使用。DeepSeek-R1/R1-Zero 显示纯 RL 能激励长推理行为，但其 base model 已有强预训练能力。

蒸馏阶段把强模型的推理轨迹压缩到小模型中，提高小模型表现。

**最终结论**

预训练产生潜在推理能力，SFT 赋予可读格式，RL 激发和重分配高质量推理策略，蒸馏迁移推理行为。

**面试话术**

“我认为推理能力的底座主要来自预训练，因为模型必须先有知识、代码和数学模式。SFT 让它学会用人类可读的步骤表达。RL 尤其是 RLVR 的作用是把概率质量推向能拿高 reward 的长推理轨迹，比如自检、反思和更长 CoT。DeepSeek-R1 说明 RL 可以显著激励推理行为，但不是从零创造所有能力，而是在强 base 上释放和重组能力。”

## 19. DeepSeek R1 到 V3.2 / V4，RL 部分有哪些改进，MoE RL 有什么不同？

**已检索结论**

DeepSeek-R1/R1-Zero：公开资料显示使用 GRPO 作为 RL 框架，R1-Zero 从 base 直接做 RL 激励推理；R1 使用 cold-start data、reasoning-oriented RL、rejection sampling、SFT 和多阶段 RL。

DeepSeek-V3：技术报告重点是 MoE 架构、MLA、auxiliary-loss-free load balancing、MTP，并说明预训练后有 SFT 和 RL 阶段，但公开报告没有像 R1 一样展开所有 RL 算法细节。

DeepSeek-V3.2/V3.2-Exp：公开资料重点是 DeepSeek Sparse Attention，用于长上下文训练和推理效率。V3.2 技术报告强调 DSA、效率和 agent/reasoning 表现。可说 V3.2 的主要公开创新是架构/效率，不应编造其 RL 算法细节。

DeepSeek-V4：截至 2026-06-09 的公开检索中，没有找到可靠的官方 DeepSeek-V4 技术报告。不能编造 V4 RL 改进。

MoE RL 不同点：MoE 的 token-level logprob ratio 方差更高，因为每个 token 的 expert routing 可能变化；expert 负载不均会影响训练稳定；训练和 rollout engine 的并行策略不同会导致 logprob/路由一致性问题。GSPO 这类 sequence-level objective 被 Qwen 报告能稳定 MoE RL，原因是 sequence-level ratio 对 token 局部波动不那么敏感。

**面试话术**

“R1 公开资料里最明确的是 GRPO、多阶段 RL、cold start 和 rejection sampling。V3 的公开重点是 MoE 架构、aux-loss-free load balancing、MLA 和 MTP，报告只说 SFT 后有 RL，没有把 RL 细节完全展开。V3.2 的公开创新主要是 DeepSeek Sparse Attention，提升长上下文训练和推理效率。至于 V4，截至 2026-06-09 我没有看到可靠官方技术报告，所以不能编。MoE RL 的难点是 router、expert 负载和 token-level ratio 方差，sequence-level 方法如 GSPO 更适合稳定 MoE RL。”

# Infra 部分

## 20. 不考虑 CPU offload，GRPO 训练时显存里有几个模型，考虑了能省多少显存？

**先区分“逻辑模型”和“GPU 上同时驻留的模型副本”**：GRPO 没有 Critic，但是否需要 Reference / Reward Model 取决于训练目标；rollout engine 也可能与训练 Actor 共用权重，也可能另存一份推理权重。因此没有固定的“GRPO 必须有 3 个模型”答案。

| 组件 | 是否必需 | 显存特点 |
| --- | --- | --- |
| Actor / Policy | 必需 | 训练时需权重、梯度、优化器状态和激活；rollout 引擎可能另有一份推理副本 |
| Reference | 可选 | 使用 KL 正则或参考 log-prob 时需要；冻结模型通常只需权重及推理工作区 |
| Reward model | 取决于任务 | 偏好 RLHF 常见；RLVR 可以由规则、测试器或环境 verifier 代替 |
| Critic | GRPO 不需要 | PPO actor-critic 通常有 value/critic，GRPO 用组内相对 reward 构造 advantage |

粗略估算 BF16 Actor：参数约 2 bytes/param、梯度约 2 bytes/param、Adam 一阶/二阶状态通常约 8 bytes/param；是否额外保留 FP32 master weights 取决于实现。因此完整训练态常按约 **12–16 bytes/param** 粗估，另加激活、通信 buffer、临时 workspace。冻结的 BF16 Reference / Reward 参数约 **2 bytes/param**，另加推理工作区；不能把它们按训练态计算。

例如 7B 模型的 BF16 参数约 14 GB（十进制），Actor 训练态粗略约 84–112 GB；一个冻结的 7B BF16 模型约 14 GB。实际可用显存还受并行切分、激活检查点、序列长度、KV cache、内存碎片和框架实现影响，不能简单用“模型个数 × 参数显存”代表峰值。

**Offload 能省多少**取决于 offload 的对象和其是否原本驻留 GPU：optimizer state offload 可释放大约 8 bytes/param 量级；Reference / Reward offload 分别最多释放其驻留的权重及相关 buffer；若 rollout 与训练权重本来共享，则没有第二份权重可省。代价是 CPU↔GPU 传输和更复杂的调度，常需分层搬运及通信-计算 overlap。

**面试回答**：GRPO 的确定性优势是没有 Critic，不是固定只有几个模型。Actor 必需，Reference、Reward 取决于 KL 和任务形式，rollout 权重可能与 Actor 共享或复制。估算显存应分训练态 Actor（约 12–16 bytes/param）和冻结推理模型（约 2 bytes/param）计算；offload 节省量按实际搬离 GPU 的状态/权重核算，并把激活和 KV cache 单独考虑。

## 21. 分布式推理：KV Cache 传输优化、多卡通信优化策略

KV cache 的显存/传输量可粗略写作：

$$
M_{\mathrm{KV}}\approx 2 L B S H_{\mathrm{kv}} D \times \mathrm{bytes}
$$

其中 2 表示 K/V，$L$ 为层数，$B$ 为并发序列数，$S$ 为缓存长度，$H_{\mathrm{kv}}$ 为 KV head 数，$D$ 为 head dimension。GQA/MQA 会通过较少的 KV heads 降低占用。

| 优化 | 作用与适用场景 |
| --- | --- |
| Paged KV 管理 | 按 block 分配，降低碎片并支持按需增长；不等于减少模型本身的 KV 数量 |
| Prefix caching | 复用相同前缀的 KV，适合共享 system prompt 或同 prompt 多次采样；受缓存容量和淘汰策略限制 |
| KV 量化 / 分层存储 | 降低显存或传输带宽；需评估精度、访问延迟和硬件支持 |
| Prefill / decode 分离 | 将两种负载放到不同资源池，需传输 prompt KV，适合可接受传输开销的规模化服务 |
| Chunked prefill | 分块处理长 prompt，改善 decode 请求的排队与资源公平性 |

多卡推理常见并行方式：TP 将单层计算切分，带来层内 collective；PP 切层，传递层间 activation；EP 为 MoE dispatch/combine 引入 all-to-all；DP 复制模型处理独立请求。把高频通信尽量放在高速互联域内，结合拓扑感知放置、通信与计算 overlap。Ring / Context Parallelism 主要用于长序列的分布式 attention，不应笼统说成将所有通信复杂度降为某个固定比例。

**面试回答**：先判断瓶颈是 KV 容量、搬运带宽还是层间通信，再选 paged 管理、前缀复用、KV 量化/分层或 prefill-decode 分离。TP、PP、EP、DP 分别解决不同维度的扩展问题；部署时需要结合并行通信量、节点拓扑和请求形态测量，而不是只比较卡数。

## 22. INT8 与 FP8 优劣对比，训推分别用什么精度

| 维度 | INT8 | FP8 |
| --- | --- | --- |
| 表示 | 整数，通常配 scale；对称/非对称、per-tensor/per-channel 等方案均有 | 浮点格式，常见 E4M3（精度较好）和 E5M2（动态范围较大） |
| 优势 | 推理量化生态成熟，适合权重量化及部分 W8A8 路径 | 动态范围和硬件矩阵计算支持适合低精度训练/推理 |
| 注意点 | outlier、校准、scale 粒度及 kernel 支持影响效果 | 依赖硬件和软件栈；需选择 scaling recipe，精度并非自动优于 INT8 |

训练常见 BF16/FP16 混合精度；部分新硬件和训练栈支持 FP8 混合精度，通常需 scaling、保留部分敏感运算的较高精度，并按框架管理累积与优化器状态。完整训练是否使用 FP8 不能仅由格式决定。

推理可按质量、吞吐、硬件和 kernel 选择 BF16/FP16、FP8、INT8 或更低比特权重量化；KV cache 精度也是独立选项。INT8 与 FP8 的实际速度/质量收益依赖具体模型、量化方法、batch 和硬件，避免把某种格式的固定倍数收益当成普遍结论。

**面试回答**：INT8 是配 scale 的整数表示，工具链成熟但要处理 outlier 与校准；FP8 有不同指数/尾数分配，支持低精度训练和推理，但更依赖硬件及 scaling recipe。训练通常从 BF16/FP16 出发，满足硬件和 recipe 条件时采用 FP8；推理则通过目标硬件上的质量与吞吐评测决定 FP8、INT8 或其他精度。

## 23. RL rollout 中的长尾问题是什么，有哪些解决方案？

同一批 rollout 的回答长度、生成速度、工具调用或环境等待时间差异很大时，同步流程往往受最慢样本/组制约。静态 padding 还会让短序列浪费训练算力；长回答也会增加 KV 占用和样本延迟。

| 方向 | 方法 |
| --- | --- |
| 调度 | Continuous batching、长度分桶、长短请求分流，降低请求互相等待 |
| 批处理 | Sequence packing / variable-length attention 减少 padding（需训练 kernel 支持） |
| 算法约束 | 合理的 max length、终止条件和 overlong 样本处理，避免奖励机制诱导无效变长 |
| 异步化 | 分离 rollout 与训练、按完整样本或可训练片段异步供数；要管理 policy staleness |
| 运行时 | Prefix cache、合适的 KV 管理、工具超时/失败策略，改善有效吞吐 |

截断或丢弃超时样本可能改变训练分布，需记录 finish reason、截断比例和样本有效性，不能只以吞吐变快作为成功标准。

**面试回答**：长尾是少数慢且长的请求拖住同步 batch，既降低 rollout 吞吐，也造成 padding 和 KV 浪费。先从调度和 packing 减少等待/填充，再按任务设计长度边界和异常处理；若采用异步，还要监控过旧样本及有效样本率。

## 24. Continuous batching 在 RL 训练时会有什么问题，vLLM 和 SGLang 的区别？

Continuous batching 让已完成请求释放出的算力可被新请求利用，但 RL 还必须维护训练语义。每条轨迹应关联 prompt/group、rollout policy 版本、old log-prob、reward、mask 和结束原因；GRPO 计算组内 advantage 前需有足够的同 prompt 样本。动态调度本身不必然破坏 on-policy，混用不同权重版本且未标记/校正才是问题。

KV cache 通常用于生成；后续 reward 计算或训练重算 log-prob 并不必然要求保留 rollout KV。Prefix cache 可能受淘汰影响，但这通常是性能问题，不应误说成 RL 算法必须保留 KV。

| 维度 | vLLM | SGLang |
| --- | --- | --- |
| 主要特征 | Paged KV 管理、continuous batching，生态和模型支持广 | RadixAttention / prefix reuse、结构化生成及请求编排能力突出 |
| 选择关注点 | 目标模型与 kernel 支持、调度、权重更新接口、部署成熟度 | 同前，并重点评估前缀复用、复杂 agent workflow 和版本兼容 |

两者功能持续演进；吞吐差异和权重更新能力需按具体版本、模型及工作负载基准测试，不宜给出脱离配置的固定性能百分比。

**面试回答**：RL 需要 continuous batching 在提升吞吐的同时保留每条样本的版本、分组和 log-prob 元数据；组内优势需要组样本凑齐，异步更新则要显式处理 staleness。vLLM 和 SGLang 的主要取舍可从 KV 管理、prefix reuse、结构化生成、权重更新接口及目标 workload 的实测来判断。

## 25. vLLM / SGLang 怎么看利用率，KV Cache 在训练里的利用率怎么看？

利用率需分层看，且 metric 名称会随版本变化：

| 层次 | 指标示例 | 能回答的问题 |
| --- | --- | --- |
| 硬件 | SM/HBM 利用率、显存、NVLink/PCIe、功耗（如 DCGM/NVML） | 算力、带宽、显存或互联是否饱和 |
| 推理引擎 | 运行/排队请求、prefill/decode token/s、延迟分位数、KV pool 占用、抢占/驱逐、prefix cache 命中 | 队列、KV 容量、缓存或调度是否成瓶颈 |
| RL 流水线 | rollout 与训练耗时、队列深度、有效样本/有效 token 比例、policy 版本差、失败/截断率 | GPU 时间是否转化为可用训练数据 |

KV cache 的“使用率”应区别预留池容量与当前活跃 KV；训练阶段通常另有参数、梯度和激活内存，不应简单将 KV 指标套用到整个训练显存。可结合 cache 占用、队列/抢占、吞吐和延迟判断：占用高本身不是目标，满载并伴随排队、抢占或吞吐下降才提示容量压力；占用低也可能是并发不足，需与请求队列及硬件指标一起诊断。

**面试回答**：硬件层看算力、显存和互联，engine 层看吞吐、排队、KV 活跃占用和抢占，RL 层看有效 token、样本时效及 rollout/train 平衡。metric 名称按所用版本查文档；不能用单一的 GPU 或 KV 百分比代表整个系统利用率。

## 26. 多机多卡 RL 训练时如何实现反向传播？

Rollout 通常在推理引擎中采样，不通过离散 token 的采样过程反传。训练阶段对轨迹重新计算当前策略的 log-prob，结合 rollout 时保存的 old log-prob、reward/advantage 计算 PPO/GRPO 等目标，再对当前 Actor 做反向传播。old log-prob、reward 和 advantage 通常作为训练目标中的固定量，不对它们反传；是否训练 Critic 取决于算法，GRPO 不需要 Critic。

| 并行方式 | 主要工作 |
| --- | --- |
| DP / FSDP / ZeRO | 数据并行；FSDP/ZeRO 还分片参数、梯度和优化器状态，按需 all-gather / reduce-scatter |
| TP | 将层内矩阵计算切分，前后向需要相应 collective |
| PP | 按层切分模型，流水传递 activation 和 gradient |
| EP | 分发 MoE token 到 experts，前后向均涉及相应通信 |

具体通信和 optimizer 更新方式受并行配置及框架实现影响。更新后需将新权重同步或加载到 rollout engine，并保证 rollout 版本与样本元数据一致。

**面试回答**：梯度来自训练阶段当前策略对 log-prob 的计算图，不是对采样动作本身反传。先重算 current log-prob、计算策略损失，再由 FSDP/ZeRO、TP、PP、EP 等并行策略完成分布式 backward；更新后同步权重，并记录 rollout policy 版本。

## 27. RL 训练有哪些异步框架，解决了同步训练的什么问题？

同步流程通常分阶段执行 rollout、reward、训练和权重更新；长尾等待及阶段间资源闲置会形成流水线气泡。异步设计让不同阶段并行或由队列解耦，提高资源利用率，但会引入数据版本差异、队列管理、背压和复现复杂度。

| 系统/方向 | 讨论重点 |
| --- | --- |
| AReaL | 面向大规模异步 RL，解耦生成与训练，并需处理异步样本的策略滞后 |
| slime | 面向 RL scaling 的系统组合，关注 rollout、训练引擎及数据/权重流转的协同 |
| OpenRLHF、verl | 提供不同程度的并行/异步流程和后端组合；能力与接口随版本发展 |

“异步”并不等于相同架构：有的只并行 reward/ref 或流水化阶段，有的允许训练与 rollout 持续解耦。比较时应明确同步边界、资源是否共置、更新/权重同步机制以及如何处理 stale data，而不要用未经同配置验证的 GPU 利用率或 speedup 数字概括。

**面试回答**：异步主要缓解阶段互等、长尾和资源闲置；代价是 rollout 样本可能来自旧策略。系统设计需要给样本附 policy version，限制队列积压，并监控策略差异与有效样本率。选择框架时要按具体版本确认异步边界和训练栈，而不是只看“async”标签。

## 28. AReaL 或其他 partially rollout 框架，在 rollout 时会不会保存之前 policy 的 KV cache？

自回归生成过程中，只要同一段生成仍由相同权重版本执行，就会像普通推理一样复用该请求的 KV cache。KV 是模型权重和输入前缀的函数；如果模型更新，旧版本算出的 KV 通常不能直接作为新版本的 KV 使用。

因此要区分：

| 场景 | 处理 |
| --- | --- |
| 同一 policy version 下继续生成 | 可继续复用该请求的 KV |
| 中断后仍由同一权重版本恢复 | 若框架保存了兼容的请求状态/KV，可恢复；实现上取决于框架 |
| 切换到更新后的 policy 继续生成 | 通常需用新权重重算前缀 KV，或从新版本重新 rollout |
| 将旧版本轨迹用于训练 | 可保留轨迹和 old log-prob，但应按异步/off-policy 机制处理；保存轨迹不等于复用 KV |

不能仅凭框架名称断言其一定清空或持久化 KV；需查具体版本实现。

**面试回答**：同一版本内正常复用 KV；跨权重版本则通常不能直接复用，因为注意力投影和隐藏状态可能已改变。部分 rollout 是否持久化 cache 是实现细节；即使保留旧轨迹，也要用版本信息和相应校正处理，而不是把旧 KV 当成新策略状态。

## 29. MoE 的 EP 对 throughput 的影响

Expert Parallelism 将不同 experts 放在不同设备。一个 MoE 层通常要执行 router、token dispatch、expert 计算和结果 combine；dispatch/combine 会带来 all-to-all 通信。

| 可能收益 | 主要代价 |
| --- | --- |
| 分摊 expert 参数显存，支持更大专家池 | token dispatch/combine 的通信与同步 |
| 专家计算可并行，适合足够大的 token batch | 路由负载不均、capacity/填充导致 straggler 或浪费 |
| 配合 DP/TP 扩展吞吐 | 小 batch、尤其 decode 每步 token 少时，通信/启动开销占比可能高 |

总体吞吐取决于 token 数、top-k、expert 负载均衡、通信拓扑、all-to-all 实现和并行组合。节点内高速互联、拓扑感知放置、均衡路由及合适 batch 有助于降低成本；EP size 增大并不保证吞吐单调增加。Shared expert 等结构的收益也依赖模型实现，不能直接推导成固定通信削减比例。

**面试回答**：EP 以 all-to-all 换取 expert 参数和计算的分布式扩展。大 batch prefill 较容易摊薄通信，小 batch decode 更可能受通信延迟影响；负载不均会让慢 expert 拖住整层。因此要结合实测通信时间、token 分布、拓扑和吞吐选择 EP 配置。

## 30. Long context 场景下的 compute-communication overlap，Megatron 和 FSDP 各自的 parallelism

长上下文会增加 attention 计算、激活/缓存需求，也可能改变通信张量大小。并行方式的作用不同：

| 方式 | 作用与通信 |
| --- | --- |
| Megatron TP | 切分层内计算；需层内 collective，序列长度会影响部分通信张量大小 |
| Megatron PP | 按层切分，以 activation/gradient 传递换取模型容量；受 pipeline bubble 影响 |
| SP / CP | SP 切分部分序列维 activation；CP 将长序列分布到多卡，attention 需跨 rank 交换/访问所需信息 |
| Megatron EP | 分布 MoE experts，带来 token all-to-all |
| FSDP | 参数、梯度、优化器状态分片；forward/backward 按需 all-gather 参数、reduce-scatter 梯度 |

Overlap 的关键是尽量在当前层计算时预取下一层所需参数，并让可异步的梯度归约与反向计算重叠；长序列 attention 中也可将远端数据交换与本地 block 计算流水化。FSDP 的 all-gather 通信量主要受参数分片及模型配置影响，并非简单地随序列长度线性增长；但 activation 和计算仍会随序列长度变化。Megatron 与 FSDP 可组合，具体比较取决于并行组和实现，不能笼统断言哪一种通信量总更小。

**面试回答**：Megatron 提供 TP/PP/SP/CP/EP 等可组合的模型并行；FSDP 重点是训练状态分片及参数 all-gather、梯度 reduce-scatter。Long context 重点评估 attention/activation 和通信是否可 overlap：FSDP 可用 prefetch 与异步归约，CP/ring 类 attention 可流水处理跨卡 KV/块。最终看目标长度、模型和拓扑上的 profile。

## 31. 确定性模式怎么开，什么是 batch invariance，是什么导致的，有没有 atomic add，atomic add 能解决吗？

PyTorch 可从以下设置开始，但它们不保证整个程序在所有设备、版本和算子上都跨运行/跨平台完全一致：

```python
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

部分 CUDA 运算还要求在启动 CUDA 前设置 `CUBLAS_WORKSPACE_CONFIG`（例如 `:4096:8` 或 `:16:8`，具体按 PyTorch/CUDA 文档）。同时固定 Python/NumPy/PyTorch 随机种子，并检查所用算子是否有确定性实现；确定性模式可能报错或降低性能。

**Batch invariance** 通常指同一样本在 batch 组成、大小或调度变化时，其输出/计算结果仍满足所需的一致性。浮点加法不满足结合律，不同 reduction 次序、不同 kernel/shape、padding 和原子操作竞争都可能造成数值差异；微小误差在生成边界附近也可能改变后续 token。

`atomicAdd` 保证并发更新不会互相覆盖，但浮点累加的执行顺序仍可能不固定，所以它本身不能保证确定性或 batch invariance。需要确定性归约/固定执行路径，或框架提供的 batch-invariant kernel；需按具体 workload 验证。

**面试回答**：开 deterministic algorithms 并固定随机性，同时检查 CUDA 算子和环境限制；batch invariance 是样本不受同批其他请求或 batch 调度影响。不同 reduction 顺序和 atomic 浮点累加都可能引入差异，atomicAdd 只解决竞争写入，不保证顺序确定。确定性和 batch invariance 相关但不是同一个保证。

## 32. AReaL 和 slime 对 RL rollout bottleneck 的理解有什么不同？

二者都关注 rollout 与训练之间的系统效率，但不宜简化成“一个只加速生成、另一个只做 GPU 共置”：

| 角度 | AReaL | slime |
| --- | --- | --- |
| 公开定位 | 大规模异步 RL 系统，重点关注生成与训练解耦及异步训练 | 面向 LLM RL scaling 的系统/工程栈，重点关注 rollout、训练及数据流协同 |
| 讨论瓶颈 | 同步阶段等待、rollout 长尾、生成与训练资源利用不均，以及异步数据的 staleness | rollout 引擎、训练引擎、数据/权重同步和并行资源协同 |
| 关注的代价 | 允许异步后需评估样本陈旧度及训练偏差 | 需评估所选 rollout/training backend、调度和并行配置带来的系统开销 |

具体使用何种 GPU 分配、并行或缓存策略会随版本、配置和部署变化，面试中应结合所讨论的实现/论文，而不是把固定硬件拓扑归为框架的定义。

**面试回答**：AReaL 的代表性思路是异步解耦 generation 与 training，并显式处理 staleness；slime 更强调将 rollout、Megatron 训练和数据/权重流转作为可扩展系统协同。两者都面对长尾和资源利用问题，比较时应落到同步边界、并行配置、队列和样本时效，而不是简单说谁“只加速 rollout”。

## 33. Full async staleness 怎么看，训练时大概是多少？

可用 rollout policy 版本与训练时 policy 版本之差表示：

$$
s = v_{\mathrm{train}} - v_{\mathrm{rollout}}
$$

但版本差只是代理指标；相同的版本差在不同学习率、batch、更新频率和策略变化幅度下并不等价。还应监控 rollout 与当前策略的 log-prob ratio 分布、近似 KL、PPO clip fraction、重要性权重有效样本量（ESS）及队列年龄。

**没有通用的“训练时大概几步”**：staleness 分布由 rollout/train 吞吐、同步频率、队列容量及模型更新幅度共同决定。设置硬性最大版本差或队列年龄、丢弃过旧样本、提高同步频率或采用适当 off-policy correction，都是可能的控制手段；校正不能恢复已丢失的信息，也不能保证过旧样本无害。

**面试回答**：我会报告 staleness 的分布（如中位数、P95 和最大值），同时看版本差、ratio/KL、clip fraction 和 ESS，而不报一个脱离配置的固定数字。阈值需要在目标任务上结合吞吐与训练质量验证。

## 34. slime 里 data 怎么流，Megatron 怎么结合，loss 怎么算？

以下是常见 RL pipeline 的概念流程；slime 的具体组件、接口和并行配置应以所用版本为准：

```text
prompt → rollout 生成（记录 token、old log-prob、版本等）
       → reward / verifier
       → 按算法计算 advantage、组装 mask 与训练 batch
       → Megatron 当前策略 forward、计算 loss、backward / optimizer step
       → 更新 rollout worker 的策略权重
```

Megatron 可通过 TP/PP/DP/EP 等并行方式承载训练；rollout 与训练的权重加载/同步和资源是否共置取决于配置。样本需保留 prompt、response、old log-prob、reward/advantage、有效 token mask、结束状态和 policy version 等元数据。

以 PPO-style clipped GRPO 为例，记 $r_{i,t}=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$，$A_i$ 为组内 advantage，则有效 token 上的目标可写作：

$$
L=-\frac{1}{\sum_{i,t}m_{i,t}}\sum_{i,t}m_{i,t}
\min\left(r_{i,t}A_i,\ \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i\right)
$$

这里 $m_{i,t}$ 是有效 token mask；不同 GRPO/DAPO 实现可能采用不同归一化、裁剪、长度处理或 KL 项，必须与训练代码保持一致。loss 在对应并行配置中计算并完成分布式反向传播，不应假定所有实现都在完全相同的 rank/阶段执行。

**面试回答**：整体是 rollout 生成并记录 old log-prob 和版本，reward/verifier 产生奖励，按算法计算 advantage，再交给 Megatron 对当前策略重算 log-prob、算 masked policy loss 并分布式更新。最后将新权重同步给 rollout。要重点讲清元数据、有效 token mask、loss 归一化和版本一致性。

## 35. VeRL / TRL / Unsloth / AReaL / slime 你会选哪个？

| 框架 | 更适合优先评估的场景 | 选型时关注 |
| --- | --- | --- |
| TRL | Hugging Face 生态内快速实验和常见后训练算法 | 模型/算法接口、单机到目标规模的性能和分布式能力 |
| Unsloth | 希望简化、加速特定微调/后训练工作流 | 支持的模型、算法、硬件和版本范围；不要默认等同于通用分布式 RL 栈 |
| verl | 需要模块化 RLHF/RL 训练流程和多种 backend 的团队 | backend 组合、并行策略、可扩展性和维护成本 |
| AReaL | 重点评估异步 rollout-training 解耦的团队 | staleness 控制、调度、队列与目标任务上的端到端效果 |
| slime | 已有或计划采用相应大规模训练/推理栈的团队 | Megatron、rollout backend、权重同步及工程集成成本 |

实际选择应使用目标模型、上下文长度、算法、GPU 拓扑和团队已有栈做小规模端到端基准；同时比较有效训练 token/s、峰值显存、策略质量/稳定性、调试和升级成本。不要仅凭通用规模分档、单一吞吐排名或某次 benchmark 下结论。

**面试回答**：小规模快速验证可先看 TRL/Unsloth；需要模块化和多后端时评估 verl；核心问题是异步解耦时评估 AReaL；已有 Megatron 等大规模系统能力时评估 slime。最终按目标 workload 的训练效果、有效吞吐、维护成本和现有基础设施选，而不是宣称某个框架普遍最好。


## 参考来源

1. PPO: Schulman et al., “Proximal Policy Optimization Algorithms”, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. TRPO: Schulman et al., “Trust Region Policy Optimization”, [https://arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)
3. DPO: Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”, [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
4. DeepSeekMath/GRPO: “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models”, [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
5. DeepSeek-R1 Nature article, [https://www.nature.com/articles/s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z)
6. DeepSeek-V3 Technical Report, [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
7. DeepSeek-V3.2-Exp official repository, [https://github.com/deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
8. DeepSeek-V3.2 arXiv, [https://arxiv.org/abs/2512.02556](https://arxiv.org/abs/2512.02556)
9. DAPO project, [DAPO](https://dapo-sia.github.io/)
10. DAPO paper, [https://openreview.net/pdf?id=2a36EMSSTp](https://openreview.net/pdf?id=2a36EMSSTp)
11. GSPO: “Group Sequence Policy Optimization”, [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
12. SAPO: “Soft Adaptive Policy Optimization”, [https://arxiv.org/abs/2511.20347](https://arxiv.org/abs/2511.20347)
13. CISPO TorchRL documentation, [https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.llm.CISPOLoss.html](https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.llm.CISPOLoss.html)
14. CISPO Swift documentation, [https://swift.readthedocs.io/en/v4.3/Instruction/GRPO/AdvancedResearch/CISPO.html](https://swift.readthedocs.io/en/v4.3/Instruction/GRPO/AdvancedResearch/CISPO.html)
15. MaxRL: “Maximum Likelihood Reinforcement Learning”, [https://arxiv.org/abs/2602.02710](https://arxiv.org/abs/2602.02710)
16. SimKO: “Simple Pass@K Policy Optimization”, [https://arxiv.org/abs/2510.14807](https://arxiv.org/abs/2510.14807)
17. ProRL: “Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models”, [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864)
18. Capability boundary counterpoint: “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?”, [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)
19. OPD geometry: “On the Geometry of On-Policy Distillation”, [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082)
20. AOPD: “Asymmetric On-Policy Distillation”, [https://arxiv.org/abs/2605.06387](https://arxiv.org/abs/2605.06387)
21. AReaL: “A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning”, [https://arxiv.org/abs/2505.24298](https://arxiv.org/abs/2505.24298)
22. verl/HybridFlow paper, [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)
23. verl GitHub, [https://github.com/verl-project/verl](https://github.com/verl-project/verl)
24. slime GitHub, [https://github.com/THUDM/slime](https://github.com/THUDM/slime)
25. OpenRLHF async docs, [https://openrlhf.readthedocs.io/en/latest/async_training.html](https://openrlhf.readthedocs.io/en/latest/async_training.html)
26. vLLM/PagedAttention paper, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
27. vLLM metrics docs, [https://docs.vllm.ai/en/v0.4.1/serving/metrics.html](https://docs.vllm.ai/en/v0.4.1/serving/metrics.html)
28. vLLM reproducibility docs, [https://docs.vllm.ai/en/latest/usage/reproducibility/](https://docs.vllm.ai/en/latest/usage/reproducibility/)
29. SGLang/RadixAttention blog and paper link, [https://lmsys.org/blog/2024-01-17-sglang/](https://lmsys.org/blog/2024-01-17-sglang/)
30. PyTorch reproducibility docs, [https://docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html)
31. PyTorch deterministic algorithms API, [https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html)
32. NVIDIA Transformer Engine FP8 primer, [https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
33. NVIDIA TensorRT-LLM quantization docs, [https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
34. NVIDIA TensorRT-LLM attention and KV cache quantization docs, [https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)

