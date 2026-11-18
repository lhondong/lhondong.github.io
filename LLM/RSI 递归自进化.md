# RSI 递归自进化

## 0 RSI 的目标：逐渐把人移出 improvement loop

可以粗略理解成三个阶段：

- Human-in-the-loop（人在环路）：AI 提出改进，但每次修改都需要人确认。
- Human-on-the-loop（人在环上）：data、reward、verifier 等已经可以自动产生，人主要负责监督结果和控制部署。
- Closed loop（闭环）：系统自己产生、验证并应用改进，不再需要人工审核。

目前绝大部分所谓 self-evolving 工作，其实都还停留在 human-on-the-loop。真正的 closed-loop RSI 远没有实现。

从“自我改进发生的阶段”来看，现有方法又大致可以分成两类：

- Test-Time RSI：模型在解决当前任务的过程中进化。
- Training-Time RSI：模型把自己产生的数据、reward 或反馈重新用于训练自身。

## Test-Time RSI

更严格地说，这一类属于 Deployment-Time RSI：self-improvement 直接发生在部署过程中，而不是等到下一轮 offline training。不同方法真正的差别，在于这种改进究竟能保留多久，从只作用于当前回答，到写入模型参数，再到长期沉淀进 Agent 本身。

与此同时，人类在 loop 中承担的角色也开始发生变化：早期更多是人直接参与 critique 和 revision，而随着 execution feedback、judge、memory 和 skill evolution 被自动化，人逐渐从每一步的参与者退到对整个 improvement loop 的监督者。也就是说，很多工作实际上正在从 human-in-the-loop 向 human-on-the-loop 靠近。

### Chatbot I：Self-critique & Self-refine

最经典的形式就是

>  Generate → Critique / Verify → Refine。

模型权重保持冻结，只针对当前 query 反复检查和修改自己的答案。这里真正关键的是 verification signal：单纯依赖模型自身的 self-critique 往往并不稳定，因此更有效的方法通常会引入 execution feedback、retrieval、solver 或 judge，为 refinement 提供额外 grounding。

但这种改进只停留在当前 output。当前 query refine 得再好，episode 结束之后，这些经验通常也随之消失。因此它更偏向提升单次回答，而不是让模型长期积累能力。

### Chatbot II：Test-Time Training（TTT）

TTT 再进一步，不只是修改 output，而是直接更新模型参数。这样，模型在当前 query 中获得的 experience 就有机会被写入 weights，而不是使用完直接丢弃。

于是原本的 Experience → Better Answer，开始变成

>  Experience → Update → Better Model → New Experience

相比 self-refine，TTT 让 test-time 产生的知识具有了更强的 persistence，也让当前 iteration 获得的经验能够继续影响后续 reasoning。

这一范式可以和不同后训练算法结合，目前比较热门的一类就是 Test-Time RL（TTRL）。

### Agent：Harness Evolution

到了 Agent 层面，self-improvement 的对象进一步从 output 和 weights 扩展到整个 Harness：Prompt、Tool、Memory、Skill、Workflow，甚至 Agent Code。此时优化的不再只是模型怎么回答，而是 Agent 本身如何完成任务。

一个很典型的方向是让 Agent 从真实 interaction 和失败 trajectory 中积累经验：Computer-use Agent 可以从 failure 中生成新的 training experience，Experience Graph 则把 branch、execution、failure、repair 等过程长期保存；SkillMaster、SkillSmith、Socratic-SWE 等工作进一步让 Agent 自动生成、修改甚至组合自己的 skills 和 tools。更激进的 Gödel Agent 和 Darwin Gödel Machine 则直接把 self-modification 推到 Agent code 层面，让 Agent 搜索、修改并保留更好的自身实现。

这里真正重要的变化是 accumulation：一次 response refinement 会消失，一次 test-time update 可以保留一段时间，而新的 memory、skill、tool 或 workflow 可以跨 task 持续存在，并成为下一轮 evolution 的基础。

所以从 self-refine 到 TTT，再到 harness evolution，Test-Time RSI 的发展脉络其实很清楚：

>  改进从当前 output，逐渐进入 weights，最后沉淀到 Agent 本身。

随着 persistence 不断增强，这类方法也开始从短暂的 bounded refinement，逐渐走向能够持续积累的 self-modification。同时也伴随着人类逐渐退出具体 improvement step。但目前绝大多数系统仍然需要人在外层监督和 gate deployment，因此离真正的 closed loop 还有明显距离。

## Training-Time RSI

Training-Time RSI 的核心是：系统开始自己产生 learning signal，并把这些信号重新送回 training loop，使 improvement 能够通过参数更新跨 iteration 持续积累。

相比 test-time RSI，这里的变化会直接进入下一轮 model。与此同时，人类的角色也开始从直接提供 label / reward，逐渐退到设计 training rule 和监督整个 improvement loop，更接近 human-on-the-loop。

### 场景 I：Zero-label

这一阶段 problem 仍然由外部提供，但 supervision 开始逐渐由模型自己产生。根据不同的 post-training paradigm，大致有三条路线：

- **Fine-tuning**：模型自己生成 reasoning trace，再筛选高质量 trajectory 作为下一轮 SFT 数据。经典的 STaR 仍然需要正确答案筛选 reasoning；更进一步的方法则开始只从 unlabeled questions 出发，通过 self-verification 构造训练数据。
- **RL**：对应目前很热门的 **self-rewarding RL**，核心是在没有现成 verifiable reward 的情况下，从模型自身获得 reward。一类方法利用 confidence、likelihood、entropy、self-certainty 等内生信号，另一类则利用同一 problem 的多个 rollout，通过 majority voting 等 group-level 信息构造 pseudo reward。
- **Distillation**：近期很热的一条路线是 **on-policy self-distillation**。rollout 由 student 自己生成，teacher 在 student 实际访问到的 trajectory 上提供 dense token-level supervision。OPSD 去掉独立的 external teacher，让同一个模型同时扮演 student 和 teacher，但 teacher 仍然需要 verified / ground-truth solution；进一步的 U-OPSD 连 reference label 也拿掉，通过多个 self-generated rollouts 的 majority consensus 构造 pseudo-solution。

三条路线虽然分别使用 trajectory、reward 和 token-level distribution，但本质上都在做同一件事：把原本由人提供的 supervision 搬进模型自己的 improvement loop。

随之而来的风险主要有两类：

- **self-confirming loop**：generator 和 verifier 往往共享相似的能力与 bias，因此错误可能沿着 model bias → biased evaluation → biased learning signal → stronger bias 被不断写回模型。
- **training collapse**：即使 reward 本身完全正确，training dynamics 也未必稳定。在具有真实 binary verifier 的 code RL 中，仍然出现过典型的 rise-and-collapse：performance 先持续提升，随后在同一次训练中突然崩溃，KL / EWC 等约束也无法完全阻止。

所以 zero-label 减少了人类对 supervision 的直接参与，但问题也从“如何获得 supervision”，逐渐转向 self-generated signal 是否可靠，以及这种 improvement 能否在多轮 iteration 中稳定累积。这更接近一个长期的 stability engineering 问题。

### 场景 II：Zero-data

zero-data 又把 improvement loop 向前闭合了一步。zero-label 仍然需要外部提供 problem，而到了 zero-data，模型进一步开始自己生成 problem / curriculum，也就是从“人给题、模型自己学”，进一步走向“连下一轮学什么也由模型自己决定”。

这一类通常可以归入 self-play。Absolute Zero 和 R-Zero 都采用 proposer / challenger ↔ solver 的 co-evolution：challenger 根据 solver 当前的能力生成新的 problem，solver 提升以后，又反过来改变下一轮值得生成什么问题。Agent0 则把这种关系扩展到 agentic task，让 curriculum agent 和 tool-using executor 从 zero external data 开始共同 evolve[1]。

因此，两者可以简单概括为：

- zero-label：human problem + self-generated learning signal
- zero-data：self-generated problem + self-generated learning signal

前者逐渐把 how to learn 交给模型，后者进一步把 what to learn 也交给模型。training 不再只是消费固定 dataset，而开始根据自身能力持续产生新的 learning frontier。

但这里仍然有一个边界：虽然 problem 和 learning signal 可以动态变化，整个 improvement procedure 通常仍然是人为预先设计的。majority voting、uncertainty reward、filtering rule 等 evaluation mechanism 并不会随着 solver 的提升而自动获得更强的 verification capability。

zero-data 同样存在自己的风险：

- **diversity collapse**：proposer 可能逐渐收敛到一小类最容易满足 reward 的问题，使 curriculum 越来越窄。
- **reward grounding failure**：如果 reward 无法持续锚定真实能力提升，proposer 和 solver 的 co-evolution 可能共同偏离真正有价值的 learning frontier。

所以 zero-data 已经能够回答“下一轮学什么”，但“下一轮应该采用什么方式改进模型”，通常仍然是人预先决定的。

### 进一步延伸：auto research

再往前一步，self-improvement 开始从 curriculum generation 扩展到 research process 本身。这时系统优化的不再只是 problem / curriculum，而是更完整的 improvement strategy。research agent 可以根据模型当前的表现分析 failure、提出下一步 hypothesis，选择 data / training recipe、运行实验，并根据 evaluation result 持续调整后续的 research decision。

因此可以简单理解为：

- zero-label：self-generated supervision
- zero-data：self-generated curriculum
- auto research：self-generated improvement strategy

从这个角度看，人类正在从提供 supervision、提供 problem，进一步退出部分 research decision。

而无论 improvement loop 扩展到哪一层，最终都需要回答同一个问题：

>  verification 本身是否足够可靠？

## Key Component：Verifier

把 test-time 和 training-time RSI 放在一起看，可以发现一个共同约束：系统已经能够逐步自生成 answer、experience、learning signal，甚至 problem / curriculum，但每一次 evolution 最终都需要一个可靠的标准来判断——这次更新是否真的带来了 improvement。

因此，verification 很可能比 generation 更接近 RSI 的核心瓶颈。一个 self-improvement loop 能否持续，很大程度上取决于它所依赖的 evaluation signal 是否足够可靠。

不同任务上的难度差异非常明显。math、code 等形式化任务可以依赖 proof checker、unit test 或 execution feedback，因此较容易形成稳定的自动化 loop；但这类系统通常仍然属于 bounded self-improvement，因为 evaluation criterion 本身是预先固定的。当任务逐渐走向 open-ended agent、creative task，甚至 research taste 和 research direction-setting 时，verification 会明显变得更困难。此时系统不仅需要判断结果是否正确，还要评价 novelty、usefulness、importance 等更难形式化的属性。也就是说，越接近 open-ended RSI，什么算更好本身就越难定义。

因此，近期开始出现一些直接针对 verifier 本身的 self-improvement：

- Self-Trained Verification：将 verifier 本身作为训练对象，希望 verification capability 能够随 iteration 持续提升。
- Self-evolving Deep Research Agent：在 agent 能力演化的同时，持续更新用于评价 research output 的 rubric / verifier。
- Meta-evaluation：不再只评价 output，而是进一步评价 evaluator 本身。
- Red Queen Gödel Machine：让 agent 和 evaluator 共同 evolve，使 evaluation criterion 不再始终保持固定。

这和前面的 R-Zero 仍然有所不同。R-Zero 中 evaluation signal 会随着 solver 的状态动态变化，但 majority voting、uncertainty reward 等 verification mechanism 本身仍然是人为设定的；而这里真正进入 evolution loop 的，是 verifier 的能力，甚至 evaluation criterion 本身。

沿着前两节的逻辑继续往下，RSI 的闭环实际上在逐步扩展：

>  answer → experience → learning signal → problem / curriculum → verifier

前面的过程主要是在减少人类对怎么学和学什么的直接参与，而当 verifier 也开始 evolve 时，系统才真正开始触及更深的一层：连什么算 improvement 也不再完全由一个固定的外部标准决定。

但这并没有彻底解决问题。verifier 可以由 meta-verifier 评价，evaluator 也可以与 agent co-evolve，但最终仍然需要回答：这些不断变化的 evaluation mechanism 如何保持与真实目标的 grounding？目前这一点仍然没有成熟答案。如何让 verifier 持续 evolve，同时避免 policy 和 verifier 共同 drift，可能才是 close-loop RSI 中最关键、也最难解决的问题之一。

参考文献：

[1] [https://arxiv.org/pdf/2607.07663](https://arxiv.org/pdf/2607.07663)

[2] [https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/](https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/) 

[3] [https://www.anthropic.com/institute/recursive-self-improvement](https://www.anthropic.com/institute/recursive-self-improvement)
