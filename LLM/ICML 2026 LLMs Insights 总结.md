# ICML 2026 LLMs 相关 Insights 总结

## 一、Post-training / RL / Process Supervision

**Insight 1**：之前关于 RLVR 的两种相反观点（RL 只是 sharpening V.S. RL 可以扩展边界）已经有定论了：**两个都对**，冲突的原因是真实大模型训练太不可控。

如果任务已经被 pre-training 覆盖，RL 主要提升 pass@1，不提升 pass@128，本质是 sharpening；如果任务太难，base model 几乎没有成功轨迹，RL 又会因为 reward sparsity 学不动；只有当 RL 数据位于模型的 edge of competence，也就是 pass@1 失败但 pass@k 偶尔成功的区域时，RL 才会带来真正的 extrapolative capability gain，提升 OOD 任务的 pass@128。

不同 training 的 Intuition：**pre-training 负责播种 reasoning primitives（基本操作），mid-training 负责把这些 primitives 组织成 RL-ready priors。模型已经有正确行为的雏形后，post-training RL 后面可以采样到、识别到、强化它，然后在边界任务上探索和放大**。我感觉可以结合另一篇 ICML 2026 GRPO 学习的本质 "Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification "从数学视角理解这种学习过程的差异。在 binary verifiable reward 下，GRPO 的 group mean+variance normalization 可以被解释成 adaptive weighted contrastive loss。（On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models）

**Insight 2**：pass@128=0 的题不能直接用 GRPO 学，因为没信号。一种解决方是让模型先学会自己给自己出题。**模型不会解最终难题，不代表它不能生成有帮助的中间问题**。pretraining 里可能已经有很多局部概念和基础题型，用 meta-RL 把这些 latent stepping stones 挖出来，把不可学习的 hard set 变成可学习的 curriculum。

具体做法是，训一个 teacher 生成 synthetic question-answer pairs，student 在这些题上 RL 训练几步，然后用 student 在真实 hard problems 上的提升作为 teacher reward。这样 teacher 不是被 learnability、difficulty 这类 intrinsic proxy 奖励，而是被真实 student progress 奖励。我感觉用真实学习进展来 ground 数据生成这个思路很有趣。

（Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability，方法叫 SOAR：Self-Optimization via Asymmetric RL）

**Insight 3**：大家现在把 GRPO 直接用于 multi-reward RL，其实会把不同 reward 组合压成相同 advantage，导致训练信号分辨率下降；例如，GRPO 训练 math reasoning 时模型很快把 format/length reward 刷满，但 correctness 会下降，400 步后甚至出现训练不稳定，因为 ormat reward 很容易拿，correctness reward 难。(correct=1, format=0) 和 (correct=0, format=1) 如果都被压成总分 1，算法就不知道这两个 response 在不同维度上的差异。修正很简单：**不要先把多个 reward 加起来再 normalize，而是每个 reward 单独做 group normalization，再把 normalized advantages 加起来**。

（GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization）

**Insight 4**：Process Reward Model 是用来判断一个 solution 每一步推理质量的模型。PRM 的价值在于：training 时可以给更细粒度反馈，减少 reward hacking；reasoning test-time scaling 时作为 verifier 可以决定哪些 paths 值得被选择或继续搜索。传统 PRM 大多是 discriminative PRM，接 classification head，用 binary cross-entropy 训练 step-level scores，输入题目 + 解法 prefix + 当前 step，输出这个 step 正确的概率。直接用 LLM-as-a-Judge 也行，但是未经专门训练的 judge 不稳定（e.g., Invalid output, overthinking, looping）

更好的方法是：用强 reasoning model 生成 verification CoT，用原有 process labels 过滤，再把思考过程蒸馏给小模型，可以比直接训练 step classifier 更 data-efficient。1% 的数据量就能蒸一个超过之前 discriminative PRM 的 ThinkingPRM。
现在来看，这已经属于常规的工程操作了（strong model 生成 CoT，gold labels 做 filter，小模型做 SFT）。
题外话，最近火的 OPD 和 RPM 都属于一个大趋势，post-training 正在从 final answer supervision 走向 process/state supervision。

（ThinkPRM: Process Reward Models That Think）

**Insight 5**：解决 RL 训练监督信号稀疏的另一种方法。代码环境、tool use 和 LLM judge 其实经常返回 rich textual feedback 告诉模型为什么错。同一个模型在没看到 feedback 时是 student，看到 feedback 后就可以作为 self-teacher。它不需要外部 strong teacher，也不需要训练 reward model，而是把环境反馈放进 context，让模型重新评估自己原来的回答，然后把 feedback-conditioned next-token distribution 蒸馏回原 policy。这样模型就能把事后反馈转化成 dense token/logit-level credit assignment。和 OPD、ThinkPRM 都属于同一个大趋势。

（Reinforcement Learning via Self-Distillation）

**Insight 6**：只有 expert demonstrations（专家示范）没有 reward 时，能不能也做 on-policy learning，学新技能且不灾难性遗忘？思路大同小异：同一个模型，没看 demonstration 时是 student；看了 demonstration 后是 teacher。训练时让 student 先自己 on-policy 生成，然后让"看过 demo 的自己"给这些 student-generated tokens 打分，再把 teacher distribution 蒸馏回 student。

减少遗忘的 intuition：SFT 为了拟合新数据，会把模型强行拉向 offline expert dataset，可能偏离 base model 太远；而这里的 teacher 虽然看了 demo、能完成新任务，但它仍然是同一个 base model conditioned on context，因此 distribution 更接近原模型。

Callback：这篇和 Insight 5 很像，都是 self-distillation，但一个 condition on feedback，一个 condition on expert demo。ICML 2026 另一篇论文 "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models" 和这篇论文的思路也差不多，那篇是让学生看正确解答过程然后自己当老师的，就不额外总结了。

（Self-Distillation Enables Continual Learning）

**Insight 7**：另一种在没有正确答案时搞到 RL reward 的方法。多采样不只是为了 test-time 提升，而是可以被回收成训练信号。模型的多个 attempts (aka. rollouts) 会暴露不同局部信息和错误模式，可以把这些探索痕迹综合成一个临时 teacher rollout，作为 math 问题的正确答案用，也可以在 non-verifiable 任务中生成 self-proposed rubrics 再用 judge 判断并给 RL reward。**以前我们多采样是为了 vote 或 Best-of-N，我现在感觉用一个模型做聚合的方法或许更好**。
本质上和 OPSD/SDFT/SDPO/RLEF 属于同一条线，都说的是**post-training 正在从依赖人工 reference/verifier，走向从模型自己的探索、反馈、rubric 和上下文中生成监督**。

（Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision）

**Insight 8**：我感觉这篇论文的训练原则很有趣。RLVR 不是生成多少 rollouts 就一定要训练多少 rollouts；很多 rollouts 对 policy update 并不值得。GRPO 真正需要的不是更多 rollouts 本身，而是更有对比度的 rollouts。最核心的选择策略是 max-variance down-sampling：选择 reward variance 最大的 subset，让更新同时看到高 reward 成功样例和低 reward 失败样例，从而保留最强 contrastive signal。

（Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning）

**Insight 9**: 科学发现不是 average-case reasoning，而是 extreme-value search。一个非常有趣的研究方向：给定一个具体的科学或工程问题，比如数学构造、GPU kernel 优化、算法竞赛、single-cell RNA-seq denoising，模型需要在 test time 找到一个新的 state-of-the-art solution。

既然目标是找到一个极少数的 breakthrough solution，就不应该只做 frozen best-of-N，也不应该直接套优化平均 reward 的 RL（优化的是 average reward policy）；模型应该在这个 test problem 上边试边训，把经验写进权重，并且让训练和搜索都偏向 top-tail solutions（目标不是平均表现，而是最大化发现一个极好 solution 的概率）。

（Learning to Discover at Test Time）

## 二、Agent / Tool Use / Memory / Dynamic Agent RL

**Insight 10**：每个 agent 内部的 hidden state 本来可能包含很丰富的中间表征，但它必须先把这些表征压缩成文字；下一个 agent 再把这些文字重新 tokenize、重新 encode 成 hidden states。这个过程会比较慢，存在信息损失，以及继承之前的错误。

一个有趣的研究方向是：每个 agent 先在 hidden state 层面生成 latent thoughts，然后把前一个 agent 的 layer-wise KV cache 拼接到后一个 agent 的 past_key_values 里，让下一个 agent 直接 condition on 前一个 agent 的内部 working memory，最后只让最后一个 agent decode 出最终答案。

直接把 last-layer hidden states 插入浅层作为 input embeddings 肯定存在分布不匹配的问题，解决方法细节可以参考论文。大概就是 w_out 告诉我们 last-layer hidden state 如何映射到 vocab 语义，w_in 告诉我们这些 vocab 语义在 input embedding space 里长什么样，可以做一个 ridge regression 得到一个线性映射把 output hidden state 映射回 input embedding-like space、

**PS. 这篇论文的 idea 很有吸引力，值得 follow，但我发现 Github 有很多人复现失败了（以及仅仅通过一些参数优化就让 baselines 比这个方法还好），所以暂时对最终潜力保持怀疑态度**。

（Latent Collaboration in Multi-Agent Systems）

**Insight 11**：如何优化 Agent？最开始大家用直接 imitation learning 让 agent 学习 expert 路径会导致 distribution shift，agent 从未观察过 non-expert actions 的后果（类似 SFT distillation 的缺陷）；用 RL 让 agent 自己探索+环境给 reward 也可以，但是很多环境没有 reward，多轮 tool use 也经常是 long-horizon、delayed reward，credit assignment 很困难。

一个更好的 intuition 是，**agent 不一定要等到有 reward 才能从 experience 中学习。只要环境会对动作产生可观察的 state transition，这个 transition 本身就是监督信号**。具体可以有两个实现方法。

做法 1：**Implicit World Modeling（aka. next state prediction）**。比如 web agent 在订机票页面填了一个 invalid date，页面返回错误提示。IWM 就让模型预测这个错误提示。这样模型不是靠 reward 知道这个动作不好，而是通过预测环境变化，学到动作和后果之间的关系。这个可以作为 policy optimization 前的 lightweight warm-up。

做法 2：**Self-Reflection（aka. 让模型解释为什么 expert action 更好）**。比如 WebShop 里，专家点击 "$15 blue shirt"，agent alternative 点击 "$30 red shirt"。reflection 可能会说："red shirt 虽然颜色匹配，但超过 $20 预算；blue shirt 同时满足颜色和预算"。SR 本质上就是把 expert-vs-self-action 的后果差异转成 natural language decision principle。

（Agent Learning via Early Experience）

**Insight 12**：真实 agent 训练里的学习信号来自整个闭环，而不是单个 reward 函数。policy 犯错提供经验，reward model 把经验变成诊断，environment 把诊断转成更合适的训练任务。长 trajectory agent RL 不能只动态更新 policy，reward model 和 environment 也应该一起进化。

RLAnything 把 RL 系统拆成三个互相反馈的组件：policy、reward model 和 environment。policy 用最终 outcome reward 和 step-wise reward 一起训练；reward model 通过 policy trajectories、outcome consistency 和 self-consistency 继续优化，变成更好的过程评价器；environment 根据 policy 成功率和 reward model 总结出的错误反馈自动调难度，让任务不要太难也不要太简单。

（RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System）

**Insight 13**：如果 Deep Research 任务没有简单 gold answer，RL 的 reward 应该从哪里来？对于 open-ended research tasks，reward function 应该是一个不断更新的 evaluation memory。**policy 在探索答案空间，rubric 也要探索评价空间**；模型发现了新证据，rubric 要奖励；模型暴露了新型 reward hacking，rubric 要惩罚。它把 Deep Research 训练从"优化固定 judge"推进到"policy 和 judge 标准共同演化"。我感觉评分标准本身动态化这个问题比用 LLM judge 打分更有意思。

（DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research）

**Insight 14**：之前大家感觉 LLMs 调用的工具就是 web search / code interpreter / calculator / APIs，这篇论文说**模型本身也可以是工具**，例如 Qwen3-8B 解决一个复杂问题的时候，可以自动调用 GPT-5 来解决。
通过 prompt 实现效果不太行，存在一些默认调用 bias，所以这本身是一个需要 RL 训练的 sequential decision-making problem。我们可以用 GRPO 训一个小模型作为调度器，根据问题需要自动选择调用 expert models 和 tools 解决问题。

（ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration）

**Insight 15**：LLM agent 如果要长期陪伴用户、跨 session 执行任务、记住偏好和历史事件，就需要 memory system。最笨的方法就是存储所有历史交互记录。这篇论文是一个很好的 agent memory engineering principle：**write-time compression 比 retrieve-time rescue 更重要**。重点前移到 write-time，可以做许多工程上的优化，例如可以先过滤低价值对话、做各种指代消解、时间归一化、把碎片事实合成高密度 memory、根据 query 动态决定检索范围、结合 semantic、lexical、symbolic 不同视角 index…

（SimpleMem: Efficient Lifelong Memory for LLM Agents）

## 三、Interpretability / Model Understanding

**Insight 16**：非常反直觉的一个发现，我甚至感觉许多顶会论文的方法之所以 work 就是因为符合这种随机的彩票。
这篇论文发现，**大规模、充分 pretrained 的模型附近，很多随机 Gaussian perturbations 居然能提升某些 downstream tasks**。所以不要只把 pretrained weights 看成一个点，而要把它看成一个局部分布的中心。这个分布附近可能已经存在很多不同任务的 specialist models。后训练不一定必须靠梯度一步步优化；你可以随机扰动模型权重，筛出在目标任务上表现好的扰动模型，再 ensemble 它们。

（Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights）

**Insight 17**：LLM memorization 之前都是研究这两个问题：能不能诱导模型吐出训练样本？能不能判断某条数据是否在训练集中？但这篇论文说这俩指标都有问题，它们很难区分 memorization 和 generalization。
一个非常漂亮的定义 memorization 的新视角：**如果有了模型 M 之后，我能用更少 bits 描述某个样本 x，那么模型 M 就包含了关于 x 的信息。压缩省下来的 bits，就是 memorization**。

作者用随机 bitstrings 消除 generalization，测得 GPT-style 模型大约有 3.6 bits/parameter 的 memorization capacity；再用真实文本发现，模型会先 memorize，直到 capacity 接近饱和，之后才更多转向 generalization。

（How much do language models memorize?）

## 四、Multimodal / Vision / World Models

**Insight 18**：MLLM 的视觉推理可能是倒置的。人类先发展底层视觉原语，再发展语言和抽象推理；现在的 MLLM 则可能先通过图文对齐和语言知识学会高阶 benchmark 模式，但缺少 object permanence、tracking、spatial transformation 这些底层视觉结构。

很多模型能做高阶知识型多模态题，却在找相同图案、追踪线条、数 3D 积木、判断折纸展开这类儿童级任务上失败。因为很多视觉题人类不是用语言解，而是画线、圈图、补全图案。所以**未来视觉 reasoning 可能不应该完全压缩进 language CoT，而应该允许模型在 visual space 里外化和操作中间状态**。RLVR 可以带来一些提升，但主要改善能被语言化的任务，对 continuous visual tracking 这类任务帮助有限。单纯增加 scaling thinking tokens 可能解决不了真正的视觉基础问题。

（BabyVision: Visual Reasoning Beyond Language）

**Insight 19**：大量无标注视频里其实包含丰富的 action-effect 结构，只是没有显式 action labels。模型如果把视频变化拆成 state 和 latent action，就有机会把互联网视频变成 world model 预训练数据。不过现实是，在没有统一身体的视频里，学到的 action 多半是 camera-relative、spatially localized change，而不是直接可执行的机器人动作。真正用于控制时，还需要 controller 把真实 actions 映射到 latent action space。

（Learning Latent Action World Models In The Wild）

## 五、Evaluation / LLM-as-a-Judge Calibration

**Insight 20**：这篇论文把医学检测/流行病学里的 prevalence correction 搬到 LLM evaluation 里了，挺有意思的。因为现在很多论文直接报告 LLM judge 判为 correct 的比例，但只要 judge 不是完美的这个比例就是有偏估计。judge 会同时犯两类错误：把错误答案判正确，以及把正确答案判错误。可以用 sensitivity 和 specificity 建模这个 noisy channel，并用一个 human-labeled calibration set 估计这两个参数，然后把 raw judge score 反解成 bias-adjusted accuracy。

（How to Correctly Report LLM-as-a-Judge Evaluations）

## 思考 & 复盘 & 领域共识

### 1. Pretraining 不是一个能力点，而是一个 latent possibility space

我在不同论文在看同一个东西的不同切面：

- Interplay 论文：pretraining 播种 reasoning primitives，mid-training 把它们组织成 RL-ready priors，RL 只是在 edge of competence 上把已经偶尔能采样到的行为放大。
- Neural Thickets：pretrained weights 附近不是一个孤立最优点，而是一片 dense task-specialist neighborhood；很多随机 perturbation 已经能变成某些任务的 specialist。
- Memorization 论文：pretraining 把有限 capacity 分配给 sample-level memorization 和 distribution-level generalization；当数据超过 capacity 后，模型被迫从死记硬背转向学习共享结构。

合起来看，pretraining 更像是在做三件事：

1. 存储局部能力原语：reasoning primitives
2. 塑造可扰动的能力邻域：nearby task experts
3. 分配有限信息容量：memorization vs generalization

### 2. RL 的本质是把稀有成功轨迹变成稳定行为

- Interplay：定义了 RL 有效区间。太简单时 RL 只是 sharpening；太难时全是 0 reward；只有 edge of competence 才会出现真正能力扩展。

- SOAR：如果目标任务已经超出 edge，模型 pass@128=0，就不能硬上 GRPO。需要先生成 stepping-stone curriculum，把任务拉回 edge 附近。
- GDPO：当 reward 是多维的，不能过早压成一个 scalar，否则会丢掉不同目标的学习方向。
- PODS：当 rollouts 已经有了，不是每条都值得反传；GRPO 需要的是高对比度样本，而不是机械使用所有 rollouts。
- CaT：多个 rollouts 不只是 test-time vote 资源，也可以被综合成 pseudo-reference，反过来制造 RL reward。
- SDPO / SDFT / OPSD：feedback、demo、reference solution 都可以让"同一个模型的更强上下文版本"给原 policy 提供 dense token-level supervision。

所以这批论文合起来其实在定义一个新的 RLVR recipe：

1. 先判断任务是否在 edge of competence 如果太难，用 curriculum/teacher 生成 stepping stones
2. 如果有多维 reward，保留 reward 维度而不是过早 sum
3. 如果有 feedback/demo/reference，把它变成 context-conditioned teacher
4. 如果没有 reference，就用 parallel rollouts 估计 pseudo-reference
5. 如果 rollouts 很多，只把最有 contrast 的样本写进参数

### 3. Post-training 监督正在从外部标签，变成模型自己的探索痕迹

监督信号不一定来自 human label、gold answer 或固定 verifier；模型自己的 rollouts、feedback、demo-conditioned behavior、verification reasoning、rubric evolution，都可以变成监督。

不同论文的监督源不同：

- ThinkPRM：强模型的 verification CoT + process labels
- SDPO：环境 rich feedback
- SDFT：expert demonstration as privileged context
- OPSD：reference solution as privileged context
- CaT：parallel rollouts synthesized into pseudo-reference
- DR Tulu：search evidence + evolving rubrics
- RLAnything：policy trajectories + reward model diagnostics + environment adaptation 以前 post-training 是给定监督源，优化模型；现在 post-training 是模型先探索，再把探索痕迹加工成监督源。

### 4. Reward 不再是一个标量，而是一个可学习、可分解、可演化的评价系统

未来 RL 系统里，reward 不是一个固定 scalar，而是一套需要建模、校准、分解和动态更新的 evaluation infrastructure。

- GDPO：multi-reward 不能先 sum 再 normalize，否则 correctness、format、length 等不同维度会被压扁。
- ThinkPRM：只看 final answer 太粗，需要 step-wise process reward。
- DR Tulu：Deep Research 没有固定 gold answer，rubric 应该随着 policy 和搜索证据一起演化。
- RLAnything：reward model 也应该从 policy trajectories 中继续学习，不能永远当静态 judge。

这个评价系统至少有五个维度：

1. 粒度：final answer reward vs step-wise process reward
2. 维度：single reward vs multi-reward vector

3. 来源：gold label / feedback / rollout / rubric / judge
4. 动态性：static rubric vs evolving rubric
5. 可信度：raw judge score vs calibrated judge estimate 我感觉这也是 agent RL 和 open-ended RL 最核心的瓶颈。不是没有优化算法，而是我们还没有足够可靠的评价基础设施。

### 6. Test-time compute 正在从多采样变成可训练的探索过程

Test-time compute 不再只是为了当场得到更好答案，而是可以变成训练数据、reward、curriculum、甚至权重更新。

例如，不同论文的 test-time compute 用法：

- Best-of-N / majority vote 旧范式：多采样，然后选一个。
- CaT：多采样，然后合成 pseudo-reference，反过来训练。
- PODS：多采样，但只让最有 reward contrast 的 rollouts 进入 policy update。
- SOAR：生成中间问题，让 student 通过真实学习进展反馈 teacher。
- Learning to Discover at Test Time：对单个科学问题边搜索边训练，把经验写进权重，目标是找到 top-tail breakthrough solution。