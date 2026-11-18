# 从 Rollout 到 Skill

好的 Agent Skill，不是写一段更像专家的 Prompt，而是把成功和失败经验，沉淀成一个可验证、可部署、可复用的外部能力层。

最近的三篇论文：From Raw Experience to Skill Consumption、SkillEvolver、SkillOpt，其实都在回答同一个问题：Agent 做任务时产生的大量 rollout，能不能被提炼成一个真正有用的 Skill？

这里的 Skill 不是普通提示词，也不是模型权重更新。它更像一种外部程序性记忆，可以被语言说明、工具使用步骤、脚本、示例、验证规则、失败机制和高风险动作黑名单。它可以被 Agent 读取、调用、执行、复用、迭代和改进。

三篇论文的主线可以概括为：Agent 执行任务，产生成功和失败的 rollout，系统从这些 rollout 中提炼出经验；再把经验写进 Skill；最后让 target Agent 消费这个 Skill，并用真实任务表现去验证它到底有没有用。区别在于，From Raw Experience 偏研究，分析 Skill 为什么有效或失效；SkillEvolver 用 meta-skill 自动产出整套 Skill；SkillOpt 偏优化，把 skill.md 当成可调整的文本参数不断改进。

## 1. From Raw Experience：Skill 有用，但不能默认有用

From Raw Experience to Skill Consumption 更像一篇系统化研究论文。它把 Skill 生命周期拆成三个阶段：experience generation、skill extraction、skill consumption。也就是先让 Agent 在任务上跑出成功和失败的轨迹，再让 extractor 从轨迹中提取 Skill，最后让 target agent 消费这个 Skill，看 held-out tasks 上是否真的有提升。

这篇论文最重要的结论是：Skill 平均有用，但不能默认有用。换句话说，把 Skill 引入到 Agent 之后，绝大多数情况是提升，但也有一部分会变差。如果 Skill 把 Agent 引向了更复杂、更长链条的工作流，而 target model 没有能力稳定执行，表现反而可能变差，这就是负迁移。

所以让 Skill 的效果不是绝对的，而取决于 Skill 本身、target model、任务复杂度、执行环境和评测方式。同一个外部知识，给强模型可能是指导，给弱模型可能就是过度复杂。这点很关键，因为它隐藏着一个 Skill 不是"写得合理就可以"，而是必须看真正 utility。

这篇论文的第二个结论是：好 Skill 不是写得漂亮，而是要编码失败机制。有些 Skill 看起来很清楚、很有道理，但实际给 Agent 用时一不一定有效。真正有用的 Skill，往往要说清楚为什么会失败，给出具体可执行步骤，并明确列出哪些高风险动作不要做。

比如低质量 Skill 可能会写："请仔细检查结果。"这句话没错，但几乎没有操作价值。高质量 Skill 会写得更具体："在 headless spreadsheet 环境中，不要把 Excel 公式字符串当作最终答案，应该用 Python 预计算路径后填回去。"这才是把失败机制转化成了可执行规则。

## 2. SkillEvolver：核心是 Meta-skill，做的是全链路 Skill 学习

SkillEvolver 的核心是一个固定的 meta-skill。这个 meta-skill 不直接解决目标任务，而是指导 Agent 去学习一个新的 domain skill。换句话说，SkillEvolver 的 meta-skill，是一个教 Agent 如何写 Skill、部署 Skill、修补 Skill 的 Skill。

它最终产出的不是某个任务的答案，而是一个可以部署的 domain skill。这个 skill 可能包含 SKILL.md、scripts、references、examples、verification checklist 等内容。SkillEvolver 更新的是 Skill 的文本和代码，而不是模型权重。

SkillEvolver 的过程可以理解成：先理解任务，再探索不同策略，然后生成或失败轨迹，提炼成 Skill。接着把候选 Skill 交给 fresh agent 去使用，观察 fresh agent 在使用 Skill 时哪里失败，最后修补 Skill，并由 auditor 检查它是否真的可部署。

它最重要的结论是：Skill 学习不是从轨迹里总结经验。普通 trace distillation 更关注从探索轨迹中总结规律，而 SkillEvolver 这一步则更接近：先把 Skill 部署给 fresh agent，然后根据 fresh agent 在使用 Skill 时的失败来修补它。

这点很关键，因为很多 Skill 的问题不是在生成时暴露的，而是在消费时暴露。比如 Skill 里写了要运行 inspect_pdf.py，但 scripts 文件夹里没有这个脚本；或者 Skill 里写了正确逻辑，但人口太深，fresh agent 根本没法调用；再或者 Skill 里用硬编码训练字段、字段名或答案模板，这些问题，单纯看文本不一定能发现，必须让一个干净上下文的新 Agent 真正使用它才行。

SkillEvolver 的另一个启发是：Skill 需要 Auditor，而不是只需要 Generator。Auditor 检查样本是否写得够不啰嗦，它会看有没有硬编码训练样本，有没有信息泄露，有没有缺脚本，主人口是否清晰，fresh agent 是否会调用，有没有 silent bypass，以及有没有把部署验伪装成通用规则。

所以 SkillEvolver 更像一个全链路 Skill 学习系统。它的重点不在某一句 Prompt，而是让 Agent 从经验中产出一个真正可部署的 Skill artifact。

## 3. SkillOpt：把 skill.md 当成可训练文本参数

SkillOpt 的核心对象是一份 skill.md。SkillOpt 把这份 skill.md 当成 frozen agent 的外部可训练状态：target model 带着当前 skill.md 去跑任务，生成 scored rollouts；optimizer model 再根据这些 rollouts 提供修改方案。

但它不是让模型随便重写整个 Skill。它的修改是受控的，主要是 add、delete、replace，也就是小步编辑，每次修改后用 held-out validation gate 验证。只有这次修改真的有提升才接受，不提升就拒绝。被拒绝的编辑也会被记录下来，避免 optimizer 之后重复犯同类错误。

SkillOpt 的核心启发是：Skill 可以被优化，但不能被随意重写。一旦一轮新让 LLM 根据失败日志重写整个 Skill，很容易覆盖原有的规则，放大局部偏好，与原有规则冲突，或者让 Skill 越改越长、关键人口越埋越深。

所以 SkillOpt 引入了类似训练系统的机制，比如 rollout batch、reflection batch、textual learning rate、validation gate、rejected-edit buffer、slow/meta update。这些机制的目的不是让 Skill 文本更漂亮，而是让 Skill 更新更稳定、更可控、更可复现。

它的另一个实用点是：训练阶段发条，部署阶段简单。Optimizer、rejected buffer、slow/meta update 都只在训练阶段存在。最终上线时，target agent 只要消费一个静态的 best_skill.md。也就是说，Skill 优化可以离线进行，但部署时仍然保持简单。

## 4. 三篇论文的共同点

三篇论文都努力在表达同一件事，但有三个共同点。

第一，它们都把 Skill 看成外部可编辑状态，而不是模型权重。Agent 能力提升不一定要写进 model weights，也可以写进非 Skill artifact。这个外部的 Skill artifact 可以读取、可审计、可迁移、可删除、可版本管理，也可以被不同 Agent 消费。相比权重更新，Skill 更像 Agent 的外部能力层。

第二，它们都依赖成功和失败 rollout。成功 rollout 提供哪些步骤应该保留，哪些工具调用顺序更稳；失败 rollout 提供哪里会错、哪些操作会导致失败、哪些策略需要被回避。哪些操作会引发高风险。所以好的 Skill 不是只从成功里学，也不是只从失败里学，而是把成功和失败两边的经验都提炼成可执行、可迁移的形态。

第三，它们都强调真实 utility，而不是文本观感。Skill 写得像样，不等于真的有用。From Raw Experience 用 held-out delta 判断 Skill 的 utility；SkillEvolver 用 fresh agent 实际消费 Skill；SkillOpt 用 validation gate 决定是否接受编辑。它们共同说明，Skill 必须被验证，而不是被故宪。

## 5. 三篇论文的不同点

最简单的区分是：From Raw Experience 研究 Skill 为什么有效或失效；SkillEvolver 用 meta-skill 自动产出好 Skill；SkillOpt 把 skill.md 当成参数稳定优化。

可以这样理解：From Raw Experience 像科学家，它研究 Skill 在什么时候有效，什么时候会让 Agent 变差。SkillEvolver 像课程设计师，它先写教材，再让小学生试用，然后修改教材，再让陌生学生写更稳的样例，最少修补。SkillOpt 像优化器，它把数据当成参数，小步修改，考试提升才接受。

所以这三篇是互补的，而不是互斥。From Raw Experience 告诉我们，Skill 要看真实 utility，不能只看文本观感。SkillEvolver 告诉我们，Skill 要经历部署、试用、失败反馈和审计，才能成为可交付的 artifact。SkillOpt 告诉我们，Skill 修改要小步、受控、可验证，不能无限制重写。

## 6. Skill 迭代器三层架构图

好的 Skill 迭代器，不是简单的线性流程，而是一个三层结构：最底层是经验积累层，中间层是 Skill 优化层，最上层是验证部署层。三层共同产出一个 Best Skill Artifact。

**第一层是经验提炼层。** 这一层对应 From Raw Experience，关键词是 Success Rollout、Failure Rollout、Failure Mechanism。它负责回答：哪些行为带来了成功？哪些经验可以让迁移到同类任务？哪些行为引发了同类失败？这一层的核心是把 rollout 变成可迁移的经验。

**第二层是 Skill 优化层。** 这一层对应 SkillOpt，关键词是 Bounded Edit、Rejected Memory、Slow Update。它负责让 Skill 稳定渐变，而不是每一轮都写整个 Skill，而是做小步修改。它要要判断应该新增什么规则、删除什么旧规则、替换什么内容，以及哪些修改曾被验证层拒绝过。

**第三层是验证部署层。** 这一层对应 SkillEvolver，关键词是 Validation Gate、Fresh Agent、Audit。它负责判断 Skill 是否能改后才可交付。它要检查改后的 Skill 是否提升、是否能正确执行、是否存在信息泄露或造成 silent bypass。

三层架构最后产出的，不是一段漂亮的 Prompt，而是一个 Best Skill Artifact。它可以是 compact best_skill.md，也可以是更完整的 Skill 包，包含 SKILL.md、scripts、references 和 verification checklist。它的核心目标是被 Agent 消费，并提升真实任务表现。

## 7. 一个好的 Skill 迭代器应该是什么样？

综合三篇论文，一个好的 Skill 迭代器至少需要七个模块。

**第一是经验提炼模块。** 好的 Skill 不能凭空写，它需要任务输入、中间观察、工具调用、动作序列、成功或失败轨迹，以及 verifier feedback 和 final answer。如果没有 rollout，比如模型凭经验写 Skill，最后很容易变成"要仔细分析""要一步步解决"这种泛泛提示词。

**第二是失败机制模块。** 失败轨迹的价值就是告诉我们哪些不要做、哪里会错。比如 spreadsheet 任务中，表面错误是写入公式正测没通过，但真正机制可能是 headless evaluator 不会自动重算 Excel 公式。于是 Skill 应该写：不要依赖 Excel 自动重算，应使用 Python 预计算路径后填回最终单元格。

**第三是正向流程模块。** 失败轨迹告诉 Agent 不要做什么，但 Skill 不能只写 Don't。它还必须告诉 Agent 应该先做什么、再做什么、什么时候校验、什么时候终止，才能稳定走向成功 rollout。

**第四是小步编辑模块。** Skill evolution 不能每轮都整篇重写，否则容易覆盖旧规则、制造冲突、放大局部偏好。更好的方式是按照一条规则、删除一个规则、替换一段以及把关键人口提前。

**第五是验证门控模块。** Skill 修改不能只靠"看起来合理"，每个候选都应该应该经过 held-out validation，判断它是否真的提升、是否会告诉 Agent 错误的样本、是否引入了 target agent 执行不了的复杂策略。

**第六是真实消费模块。** Skill 不能只给作者 Agent 用，而应该让一个全新的 fresh agent 真正使用 Skill。测试 fresh agent 是否知道扣入时候调用 Skill，是否能看懂、是否调错样本、是否会忽略关键规则，是否会 silent bypass。很多 Skill 的失败不是知识失败，而是交付失败。

**第七是审计泛化模块。** Skill 要检查有没有硬编码训练文件名、字段值和答案格式，是否把局部经验伪装成通用规则。是否包含全部脚本、是否有清晰入口、是否能被 fresh agent 使用。

## 8. 总结

三篇论文合起来，给出了一个很清晰的方向：Skill 的本质不是 Prompt，而是 Agent 的外部程序性记忆。它不更新模型参数，而是改变 Agent 的行为策略。它不是一次性生成的，而是从 rollout、失败、验证和真实消费中迭代出来的。

所以，一个好的 Skill 迭代器应该是好几条且层、总结一段接一段，直接上线。它应该集成成功和失败 rollout，提炼正向流程和失败机制，对 Skill 做小步编辑，用 held-out validation 过滤负迁移，让 fresh agent 真实消费，审计过拟合、泄露、缺脚本和 silent bypass，最后输出可部署的 best skill artifact。

如果压缩成一句话：

> 好的 Skill 迭代器 = 经验提炼层 + Skill 优化层 + 验证部署层。

Raw Experience 贡献了第一层，它告诉我们 Skill 要看真实 utility，不能只看文本观感。SkillOpt 贡献了第二层，它告诉我们 skill.md 要像参数一样小步优化，不破坏稳定性。SkillEvolver 贡献了第三层，它告诉我们 Skill 必须被 fresh agent 真实消费，并通过 auditor 检查可部署性。

未来的 Agent Skill Engineering，不只是 Prompt Engineering 的升级版。它要像一个新的训练范式：不训练模型权重，而训练 Agent 能消费的外部能力层。

## 参考论文

1. From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills
2. SkillEvolver: Skill Learning as a Meta-Skill
3. SkillOpt: Executive Strategy for Self-Evolving Agent Skills
