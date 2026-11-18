# RLHF / PPO 代码

在游戏场景，每个 step 给环境一个 action 之后，agent 拿到的 state 都是会变化的，通常也会设计奖励函数使得每个 step 都会有 reward；但是在用强化学习微调语言模型这里，prompt 是 state，只输入一次，然后输出一串 action（回答的单词），得到一个 reward，对于 answer 的第二个词，可以把 prompt+answer 的一个词当作新的 state，而不只是把 prompt 当作 state，状态转移蕴含在 transformer 内部。

## 奖励（reward）模型训练

首先要声明的是，在强化学习阶段，**用到的 reward model 和 critic model 都使用同一个模型初始化，因此在训练 reward 模型的过程中，也是在训练 critic model**。其次对符号进行说明，大模型中间隐藏层的参数维度为 (B,L,D)，B 为 batch size 大小，L 为句子长度，D 为 embedding 维度。在接下来的代码讲解中，我也会标明代码中各个变量的维度，以更好的理解其意义。

在进行 RLHF 时，需要一个奖励模型来评估语言大模型（actor model）回答的是好是坏，这个奖励模型通常比被评估的语言大模型小一些（deepspeed 的示例中，语言大模型 66B，奖励模型只有 350M）。奖励模型的输入是 prompt+answer 的形式，让模型学会对 prompt+answer 进行打分。奖励模型最后一层隐藏层的输出维度为 (B,L,D)，通过一个 D✖️1 的全连接层将维度变为 (B, L)，在 L 这个维度上，第 i 个位置的数据表示：从第 i 个位置到最后一个位置输出所能获得的奖励分值的累加和（和 DQN 里边的 Q 值一个意义），这种形式的输出满足了 critic model 的输出要求。对应代码如下：

```python
#huggingface 模型返回值是个 list，第 0 位是模型最后输出的 hideen state
hidden_states = transformer_outputs[0]
# v_head 为 Dx1 的全连接网络对最后一维压缩
rewards = self.v_head(hidden_states).squeeze(-1)
```

对于一个奖励模型来说，目标是给一个句子进行打分，按理说每个句子对应一个分值就行了，但是目前对于长度为 L 的句子，奖励模型输出了 L 个值。我们用 L 维度上的最后一个位置的值当作为本句话的奖励得分。奖励模型训练优化采用 pair wiss loss，即同时输入模型关于同一个问题的两个回答，让模型学会这两个句子哪个分高哪个分低。之所以如此训练是因为，在给奖励模型进行数据标注的过程中，给同一个问题的不同回答量化的打具体分值比较难，但是对他们进行排序相对简单，代码如下：

```python
# 同一个 batch 里边的句子需要等长，短句后边会被 padding
# [divergence_ind:end_ind] 索引了 padding 前一个位置的输出分值
# chosen_reward 是同一个句子 pair 里分数高的句子，r_truncated_reward 是句子 pair 里分数低的句子
c_truncated_reward = chosen_reward[divergence_ind:end_ind]
r_truncated_reward = rejected_reward[divergence_ind:end_ind]
```

pair wise loss 代码如下，如果给 pair 里边好的句子打分高（c_truncated_reward），坏的句子（r_truncated_reward）打分低，loss 就会小：

```python
loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
```

在训练强化学习的过程中，会用到 reward model（critic model，再次提醒，critic model 和 reward model 是同一个模型的两个副本）的推理过程，通过调用 forward_value 实现，具体代码如下，返回的值中有两种值，values 表示每个位置 i，从第 i 个位置到最后一个位置的奖励累加值，供强化学习过程中 critic model 使用；“chosen_end_scores”指的是对每个 prompt+answer 的打分，供 reward model 使用。

```python
def forward_value(...):
    ...
    if return_value_only:
        #(B,L)
        return values
    else:
        ...
        return {
            "values": values,
            # （B,）
            "chosen_end_scores": torch.stack(chosen_end_scores),
        }
```

## 强化学习微调

强化学习微调阶段，会用到 4 个模型，actor model， ref_model，reward model 和 critic model。其中 actor model 和 ref_model 是 RLHF 第一个阶段**有监督微调模型**的两个副本。

首先说明 actor model 的**训练模式**和**推理模式**的区别。训练模式是用 teacher force 的方式，将整句话输入到模型中，并通过 mask 机制在保证不泄漏未来的单词情况下预测下一个单词。推理模式是真正的自回归，预测出下一个单词之后，当作下一步输入再预测下下个单词。

首先用 actor model 在**推理模式**下根据 prompt 生成一个 answer（prompt 对应强化学习里边的 state，answer 对应一些列的 action），代码如下：

```python
# 保证不触发反向传播
with torch.no_grad():
    seq = self.actor_model.module.generate(prompts, max_length=max_min_length, min_length=max_min_length)
```

然后利用 reward model 和 ciric model 对输出的 prompt+answer 进行打分（PPO 训练时使用的奖励值并不单单是 reward model 的输出还要考虑 kl 散度，后文介绍）：

```python
# 奖励模型返回的是个字典，key 为 chosen_end_scores 位置存储数据维度为 (B,)，表示对于 prompt+answer 的打分
reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
# critic model 返回的数据维度为 (B,L)，L 维度上第 i 个位置代表从 i 位置到最后的累积奖励
# 舍去最后一个位置是因为句子“终止符”无意义 
values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
```

actor model 是我们想通过强化学习微调的大模型，但是强化学习过程很容易把模型训练“坏”，因此需要另外一个不会参数更新的 ref_model 来当作标的，别让 actor mode 跑偏太远。在**训练模式**下，将 prompt+answer 分别输入到 actor mode 和 ref model，用 KL 散度来衡量 ref model 和 actor mode 输出的差别。同时将 KL 散度（衡量数据分布差距大小）纳入损失函数（KL 散度本质是纳入到奖励值里边的，奖励值被纳入到了损失函数），进而来约束 ref_model 和 actor mode 的输出分布别差距太大。具体代码如下：

```python
# 得到两个模型的输出
output = self.actor_model(seq, attention_mask=attention_mask)
output_ref = self.ref_model(seq, attention_mask=attention_mask)
logits = output.logits
logits_ref = output_ref.logits
...
return {
...
# 分别得到两个模型在真实单词上的预测概率
'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
...
}
...
# 计算 kl 散度，log_probs 里边存的数字经过 log 变化了，因此减法就对应除法
kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
```

PPO 训练时候的奖励值综合考虑 KL 散度和 reward 模型的输出，只考虑 answer 部分的 KL 散度，将 reward model 的输出加到 KL 散度 L 维度的最后一个位置上，得到最终的奖励值，代码如下：

```python
rewards = kl_divergence_estimate
# 只考虑 answer 部分的奖励，不考虑 prompt
start = prompts.shape[1] - 1
# 不考虑 padding 部分
ends = start + action_mask[:, start:].sum(1)
reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                         self.clip_reward_value)
batch_size = log_probs.shape[0]
# 在 L 维度上，每个位置都有 KL 散度，但是只在最后一个位置加上奖励值
for j in range(batch_size):
    rewards[j, start:ends[j]][-1] += reward_clip[j]
```

接下来的内容就是 PPO 的训练过程的比较核心的内容了，目标是计算 PPO 更新公示里边的 advantage，具体公式如下

```python
def get_advantages_and_returns(self, values, rewards, start):
    # values（B，L） critic model 输出
    # rewards（B，L）reward 包含 kl 散度
    # start answer 开始的位置
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    # 计算每个时刻（序列位置）的 critic model 预测误差
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        # critic model 预测的是 t 到到最后一个时刻的奖励和，所以变化量 delta 可以用如下公式表示
        delta = (rewards[:, t] + self.gamma * nextvalues) - values[:, t]
        # self.gamma=1，self.lam=0.95 是衰减因子，表示之前计算的 delta 对现在影响越来越小
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # 后续用来更新 critic model 用
    returns = advantages + values[:, start:]
    return advantages.detach(), returns
```

以上过程，我们已经拿到了 PPO 训练所需要的 advantage 以及 actor  model 的输出，我先现在可以对 actor model 进行训练啦。具体代码如下。logprobs 和 old_logprobs 这两个参数分别是“老 actor（n 个 epoch 才会更新一次）”和新 actor（每个 batch 都会更新它）”在正确单词上出处的概率，这块 PPO import sampling：

```python
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    # policy gradient loss
    # logprobs, old_logprobs 都是经过 log 变化的单词概率，这里带着 log 做减法就相当于在做概率除法
    log_ratio = (logprobs - old_logprobs) * mask
    # 指数操作去掉 log
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                            1.0 + self.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss
```

同样的，也要对 critic model 进行训练，更新，loss 就是 mse loss。

```python
def critic_loss_fn(self, values, old_values, returns, mask):
    ## value loss
    # 用“老 critic model”的输出约束“新 critic model”不要步子太大，裁剪一下
    values_clipped = torch.clamp(
        values,
        old_values - self.cliprange_value,
        old_values + self.cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss
```

至此，RLHF 训练流程就结束了，共涉及 actor model， ref_model，reward model 和 critic model 这四个模型，其实更新参数的模型只有 actor model 和 critic model。
