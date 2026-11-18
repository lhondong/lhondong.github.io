# Score Centering

Score centering 主要为了缓解  RL 中的 off-policy 问题，消除策略 drift。

## 1. Off-policy 和 Importance sampling

在 LLM RL 中，训练-推理不一致（training-inference mismatch, TIM）和策略陈旧性（policy staleness）都可能导致采样策略 $\mu$ 与训练策略 $\pi$ 不一致，使得直接使用采样数据计算的梯度偏离目标策略梯度，进而影响训练稳定性。

考虑 on-policy  的 policy gradient：

$$
\mathbb{E}_{\pi}\left[A(s,a) \nabla \log \pi(a|s) \right]
$$

然而实际 $(s,a)$ 由 $\mu$ 采出，即我们实际能拿到的梯度是服从 $\mu$ 分布的：

$$
\mathbb{E}_{\mu}\left[A(s,a) \nabla \log \pi(a|s) \right]
$$

通常使用重要性采样（importance sampling，IS）使得梯度估计无偏，即引入重要性权重 $\frac{\pi(a|s)}{\mu(a|s)}$ 得到：

$$
\mathbb{E}_{\mu}\left[\frac{\pi(a|s)}{\mu(a|s)} A(s,a) \nabla \log \pi(a|s) \right]=\mathbb{E}_{\pi}\left[A(s,a) \nabla \log \pi(a|s) \right]
$$

然而，ratio $\frac{\pi(a|s)}{\mu(a|s)}$ 的数值波动很大，所以 IS 的梯度估计虽然是无偏的，但是 variance 非常大，导致训练不稳定。

常用做法使用 Clip（代表算法为 PPO[2]、GRPO[3]、DAPO[4]、RIPO[5] 等等）限制只更新 $\frac{\pi(a|s)}{\mu(a|s)}$ 的数值在某个区间内（称为 proximal region）的梯度。然而这个做法带来了一定的 bias，并且区间超参数的不同设定对训练有很大影响（导致 clip 失效，entropy 暴涨，训练崩溃等等）。

> 求 $\log \pi(a|s)$ 时需要 IS 的矫正，本质是因为 $A(s,a)$ 是根据 $\mu$ 得到的，对于 $\pi$ 来说并不准确。否则只要对于 $\pi$ 准确，那么对于 $\pi$ 任意 state-action pair（不限制是否被采样出来）都可以得到有效更新

## 2. Score centering

Score centering 不希望通过 importance sampling 以及 clip ratio 的方式得到无偏的梯度估计，而是用了一种类比的方式。

### Motivation

首先，回顾我们现在拿到手的梯度是：

$$
\mathbb{E}_{\mu}\left[A(s,a) \nabla \log \pi(a|s) \right]
$$

其中 $\nabla \log \pi(a|s)$ 称为 **score**。

将其分解为：

$$
\mathbb{E}_{\mu}\left[A(s,a) \nabla \log \pi(a|s) \right]=\mathbb{E}_{\mu}\left[A(s,a)\right]\mathbb{E}_{\mu}\left[ \nabla \log \pi(a|s) \right]+\text{Cov}_{\mu}\left[A(s,a),\nabla \log \pi(a|s)\right]
$$

当 on-policy 的时候，score 的期望恒等于 0，即 $\mathbb{E}_{\pi}\left[ \nabla \log \pi(a|s) \right]\equiv0$，而在 off-policy 的情况下，显然 score 的期望$\mathbb{E}_{\mu}\left[ \nabla \log \pi(a|s) \right]\neq0$

所以作者认为，$\mathbb{E}_{\mu}\left[ \nabla \log \pi(a|s) \right]$ 这一项就是 off-policy 分布所带来的漂移（drift），导致梯度的方向是有偏的，会使得 $\pi$ 逐渐偏向于 $\mu$ 的方向（这一项本质是 $\mu,\pi$ 两者 reverse KL 的梯度）。

漂移项并不包含哪些 rollout 成功了的信息，它对优势（奖励）的依赖仅体现在优势（奖励）的均值 $\mathbb{E}_{\mu}[A(s,a)]$ 上。只有协方差项才能反映具体哪个 token 对应了怎样的优势（奖励），即学习信号。所以策略梯度更新可以看作：

$$
\underbrace{\mathbb{E}_{\mu}\left[A(s,a) \nabla \log \pi(a|s) \right]}_{\text{policy gradient update}}=\underbrace{\mathbb{E}_{\mu}\left[A(s,a)\right]\mathbb{E}_{\mu}\left[ \nabla \log \pi(a|s) \right]}_{\text{drift}}+\underbrace{\text{Cov}_{\mu}\left[A(s,a),\nabla \log \pi(a|s)\right]}_{\text{signal}}
$$

为什么协方差是 signal term？协方差项可写为：

$$
\operatorname{Cov}_{\mu}\left(A,\nabla\log\pi\right) =\mathbb{E}_{\mu}\left[ \left(A-\mathbb{E}_{\mu}[A]\right) \nabla\log\pi \right]
$$

它刻画了优势相对于其均值的变化与 score 之间的关联，也就是不同动作的优势如何影响梯度。而 drift 项只依赖优势的均值，不区分具体哪个动作获得了怎样的优势。因此，作者将前者解释为 signal、后者解释为 drift，是有直觉依据的。

所以我们希望消除 drift 项，从而降低 off-policy 分布差异导致的梯度偏差。

### Method

记 score：$z(s,a)=\nabla \log \pi(a|s)$，score 的期望：$\bar{z}=\mathbb{E}_{\mu}[\nabla \log \pi(a|s)]$。

将原本的 score 项替换为 centered score，即：$\tilde{z}(s,a)=z(s,a)-\bar{z}$。

此时使用 score centering 的 policy gradient update 就变成：

$$
 \underbrace{\mathbb{E}_{\mu}\left[A(s,a) \tilde{z}(s,a)\right]}_{\text{policy gradient update}}=\underbrace{\mathbb{E}_{\mu}\left[A(s,a)\right]\mathbb{E}_{\mu}\left[\tilde{z}(s,a) \right]}_{\text{drift}}+\underbrace{\text{Cov}_{\mu}\left[A(s,a),\tilde{z}(s,a)\right]}_{\text{signal}} 
$$

由于

$$
\mathbb{E}_{\mu}\left[\tilde{z}(s,a) \right]=\mathbb{E}_{\mu}\left[z(s,a)-\bar{z} \right]=\mathbb{E}_{\mu}\left[z(s,a)\right]-\mathbb{E}_{\mu}\left[z(s,a)\right]=0
$$

所以第一项 drift 项数值为0，从而 score centering 不产生漂移，只剩下协方差的 learning signal。

而对于协方差项：${\text{Cov}_{\mu}\left[A(s,a),\tilde{z}(s,a)\right]}={\text{Cov}_{\mu}\left[A(s,a),{z}(s,a)\right]}$，learning signal 相比于 off-policy gradient 不变。

### Implementation

由于存储 $\mu$ 的词表分布开销过大，所以考虑 Top-K（K=32或128）估计，并且使用 $\pi$ 的分布来近似 $\mu$ 其余尾部 token 的分布，同时用尾部概率质量比 $\rho$ 对其进行重新缩放，使其尾部概率质量与采样器的尾部概率质量一致。

可写为：

$$
 \mathcal{L} = -R\left( \log \pi_\theta(a_t\mid s_t) -\sum_{v\in\mathcal{H}} \operatorname{sg}\left[ \mu(v\mid s_t)-\rho\,\pi_\theta(v\mid s_t) \right] \log \pi_\theta(v\mid s_t) \right) 
$$

具体代码为：

```python
import jax.numpy as jnp
from jax.lax import stop_gradient

def score_centering_loss(train_logp, samp_logp. topk_ids, sampled_token, advantage):
  	train_head_logp = train_logp[topk_ids]
    tail_mass_ratio = (1 - jnp.exp(samp_log).sum()) / (1 - jnp.exp(train_head_logp).sum())
    head_prob_residual = jnp.exp(samp_logp) - tail_mass_ratio * jnp.exp(train_head_logp)
    logp_correction = (stop_gradient(head_prob_residual) * train_head_logp).sum()
    return -advantage * (train_logp[sampled_token] - logp_correction)
```

## 3. Score centering 的 bias 和 variance 讨论

显然，相比于重要性采样的无偏性，score centering 的梯度估计是有偏的。

考虑 score centering 的梯度：

$$
\mathbb{E}_{\mu}\left[A(s,a)(\nabla \log \pi(a|s)-\mathbb{E}_{\mu}[\nabla \log \pi(a|s)])\right]
$$

全期望展开为：

$$
\sum_a \mu(a|s) A(s,a) \left( \nabla \log \pi(a|s) -\sum_{a'}\mu(a'|s)\nabla \log \pi(a'|s) \right)
$$

将真实梯度 $\mathbb{E}_{\pi}\left[A(s,a) \nabla \log \pi(a|s) \right]$ 全期望展开：

$$
\sum_a \pi(a\mid s)\, A(s,a)\,\nabla \log \pi(a\mid s)
$$

显然两者不相等，所以 score centering 虽然消除了梯度向 $\mu$ 的drift，但梯度的估计相对于 $\pi$ 是有偏的。这是因为协方差项显然也是有偏估计。我们来具体讨论一下：

将 $\mathbb{E}_{\pi}\left[A(s,a) \nabla \log \pi(a|s) \right]$ 写成协方差的形式，即：

$$
\mathbb{E}_{\pi}\left[A(s,a) \nabla \log \pi(a|s) \right]=\mathbb{E}_{\pi}\left[A(s,a)\right]\mathbb{E}_{\pi}\left[\nabla \log \pi(a|s)\right]+\text{Cov}_{\pi}\left[A(s,a),\nabla \log \pi(a|s) \right]
$$

由于 $\mathbb{E}_{\pi}\left[\nabla \log \pi(a|s)\right]=0$，所以 $\mathbb{E}_{\pi}\left[A^\pi(s,a) z(s,a) \right]=\text{Cov}_{\pi}\left[A^\pi(s,a),z(s,a) \right]$（感觉这里的分析又印证了上文中 score centering的 motivation；另外这里对 A 的来源更严格的定义了）

对于 score centering，亦有：

$$
\mathbb{E}_{\mu}\left[A^\mu(s,a) \tilde{z}(s,a) \right]=\text{Cov}_{\mu}\left[A^\mu(s,a),{z}(s,a) \right]
$$

两者做差：

$$
\begin{aligned} \mathbb{E}_{\pi}\left[A^\pi(s,a) z(s,a) \right]- \mathbb{E}_{\mu}\left[A^\mu(s,a) \tilde{z}(s,a) \right]&=\text{Cov}_{\pi}\left[A^\pi(s,a),z(s,a) \right]-\text{Cov}_{\mu}\left[A^\mu(s,a),z(s,a) \right]\\ &=\operatorname{Cov}_{\pi}(A^\pi,z)-\operatorname{Cov}_{\mu}(A^\mu,z)+\operatorname{Cov}_{\mu}(A^\pi,z)-\operatorname{Cov}_{\mu}(A^\pi,z)\\ &=\underbrace{ \operatorname{Cov}_{\pi}(A^\pi,z) -\operatorname{Cov}_{\mu}(A^\pi,z) }_{\text{distribution mismatch}}+\underbrace{ \operatorname{Cov}_{\mu}(A^\pi-A^\mu,z) }_{\text{advantage mismatch}} \end{aligned}
$$

所以 bias 来源于两个地方：一个是概率分布，一个是优势 A 的估计。

概率分布的 mismatch 可以通过重要性采样解决，但这个又会引入 variance。优势估计的 mismatch 往往都被忽略了。

所以这里值得做的事情还很多。

进一步的，相比于原来的off-policy gradient，score centering 去掉drift项，也不一定会降低bias。考虑：

$$
\text{bias}_{\text{off}}=|\mathbb{E}_{\pi}\left[A^\pi(s,a) z(s,a) \right]- \mathbb{E}_{\mu}\left[A^\mu(s,a) {z}(s,a) \right]|=|\text{Cov}_{\pi}\left(A^\pi,z \right)-\text{Cov}_{\mu}\left(A^\mu,z\right)+ \mathbb{E}_{\mu}\left[A\right]\mathbb{E}_{\mu}\left[z \right]|
$$

$$
\text{bias}_{\text{sc}}=\left|\mathbb{E}_{\pi}\left[A^\pi(s,a) z(s,a) \right]- \mathbb{E}_{\mu}\left[A^\mu(s,a) \tilde{z}(s,a) \right]\right|=|\text{Cov}_{\pi}\left(A^\pi,z \right)-\text{Cov}_{\mu}\left(A^\mu,z\right)|
$$

两者做差：

$$
\Delta=\text{bias}_{\text{sc}}-\text{bias}_{\text{off}}=|\text{Cov}_{\pi}\left(A^\pi,z \right)-\text{Cov}_{\mu}\left(A^\mu,z\right)|-|\text{Cov}_{\pi}\left(A^\pi,z \right)-\text{Cov}_{\mu}\left(A^\mu,z\right)+ \mathbb{E}_{\mu}\left[A\right]\mathbb{E}_{\mu}\left[z \right]|
$$

由于绝对值内的项正负性未知，所以不能得出 $\Delta$ 是否大于 0 还是小于 0，所以不能得出 score centering 和 off-policy gradient 相比于真实梯度的偏差大小。

另外，从 variance 角度来看，于重要性采样 ratio $\frac{\pi(a|s)}{\mu(a|s)}$ 的数值波动很大导致 variance 非常大，训练不稳定。而 score centering 通过减 baseline 的方法，实际是降低了梯度估计的 variance，所以训练更加稳定。而且依然是 token-level 的，并且不丢弃 tokens，所以 efficiency 会更好。（注：当然 score centering 可以和 IS/clip 结合使用）

所以总体来说，score centering相比于重要性采样，可以看作一种梯度估计 bias和variance的trade-off。

## 4. 谈谈 SFT 和 RL

论文中实际的 motivation 来源于 SFT 的讨论。即为什么不管 SFT 的 off-policy 程度多么剧烈，依然不会产生训练崩溃的问题。文中给出的解释是：对于 SFT 来说，data distribution 是预先固定的，而对于 RL，data distribution 是从 $\mu$中 采样得到的，而随着训练，$\mu$ 又不断向 $\pi$ 移动，所以 RL 与 SFT 的重要区别在于：教师模型会随着学生模型一起移动。因此，作者认为，不稳定性必然源于这一动态变化的过程。

笔者认为 SFT 和 RL 的另一个差别是正负样本的影响（论文其实也讨论到了），即 RL 会同时考虑正负样本，对负样本的打压会是的概率分散给整个 vocabulary，导致 entropy 的增加。而 SFT 只考虑对样本的正向学习。其实这里可以做一个简单的实验验证一下，比如蒸馏的时候会得到大部分正样本和小部分负样本，我们通常只对正样本学习，而不考虑负样本，从而向 teacher model 逼近。如果同时考虑正负样本，即同时最小化 student 对 teacher 负样本的最大似然，能不能学习到比 teacher 更好的模型呢？同样会不会有稳定性的问题呢？

另外还可以考虑 RFT，同样 RFT 也是 off-policy，也只训练正样本，但是 RFT 稳定性很高，不会出现训练崩溃的现象，而且能较好的保持 RL 之后模型的性能。

最后一点想讨论的是数据质量的影响，在 RL 训练过程中，其实对采样得到的数据质量没有把控，只对最终答案的正确性进行判别。但显然 SFT 和 RFT 的训练数据都经过一系列的筛选。

[1] [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)

[2] [Proximal Policy Optimization Algorithms](https://www.google.com.hk/goto?url=CAESWAHrOzAVrNgYsimj9Dj5-6U-DZhJ4oSCNjFjaGEoRmRhOrGUiy-zRwgXXzLUCmx1MNJsXGsJJTUIpBwV1nCqg9gPlHqRaVJIH-IOQ8WeahBauXNaYI9nhHY)

[3] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

[4] [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

[5] [Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization](https://arxiv.org/abs/2607.10169)

## 训推分离

首先导致训推不一致的本因来自训练引擎（Megatron or FSDP）和推理引擎（VLLM or SGLang）的区别，具体区别是：生成的 token 序列并不完全一致。生成 token 序列不一致在真实使用中影响不大，但在 RL 模型更新的时候却有着很大的影响。具体而言，我们需要对每个 token 计算其 advantage（无论 GRPO or PPO），这个 advantage 会最终作用到更新模型上（无论 on policy or off policy）。那么对于原始模型生成的很低概率的 token（这里说 trajectory 其实更合理一些。更新有小概率存在的 traj 里的大概率 token 也会有些问题。）进行模型更新会引发一系列的问题：train engine 模型的生成和 rollout engine 的生成产生了巨大的变化，模型输出崩溃。

但是这里有几个点很值得深究：

1. 并不是所有的训推不一致都会导致模型 RL 训练以后会崩溃。
2. 不同尺寸的模型，甚至同尺寸不同公司的模型，训推不一致的耐受度是不一样的。有些模型训个 200 步就崩了，但有的训 1000 步也没有问题。
3. 训推不一致是个静态的评估，也就是训练引擎（Megatron or FSDP）和推理引擎（VLLM or SGLang）生成的 token 概率的 diff，但为什么在 RL 过程中这个 diff 会越来越大。当然一种解释是，RL 会让模型 output token 变多。那么生成长度越长，越容易出现训推不一致。但其实即使生成长度不变长，模型在 RL 后期依然有可能遇到训推不一致问题。

考虑到以上因素，训推不一致并不是纯粹的RL问题，这里有非常大一部分耦合了 base model 本身能力。

个人感觉是 base model 越好，训推不一致越低，换言之：

- 模型越大
- 预训练和 mid train 越充分
- 模型长文训练越充分

越不需要考虑训推不一致问题。

**这里又想说一句，越好的 base model，越朴素的 RL 算法，不需要考虑 off policy，不需要考虑 clip，maybe Reinforce 算法就够了，因为 RL 探索的范围都被 Pretraining 充分的泛化覆盖到了。**

**最后分享一两句实战经验和看法（不一定对，欢迎讨论）：**

- **这个问题本质解决方案是基座能力提升，主要提升方向是长文和一些知识的学习。**
- **无论 TIS 还是 IcePop 解决都不本质，尤其对不同模型需要调整参数，一直得看。**
- **绝对的 on policy 追求（包括 R3 这种，只有 ds 在做），算是一种放弃RL算法研究和大部分算法提速的方向，ds 这么做是因为耦合了他们的 pretrain DSA 设计，极致长文，推理 infra 能力强等多重因素。**
- **RL 大部分的 trick 都太 tricky，算是特定 pretrain，RL 不能 scaling 下的无奈之举，可能就连 thinking budget 方向这种在万卡 RL 下，都是很简单的，但是对于国内各家研究了很久都找不出门道。**
