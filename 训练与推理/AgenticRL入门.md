# AgenticRL入门

## 推理

### 一次模型调用

对于一次典型的 agentic inference，第 $i$ 次模型调用可以分为三个层次：agent harness 维护的内部状态、按照特定协议组织的 LLM call，以及模型实际接收到的输入 token 序列。

Agent harness 维护结构化消息状态 $M_i$ ，其中可以包含 system message、user message、assistant message、tool call、tool result 和 environment observation 等信息。发起模型调用前，harness 将这些信息组织为符合协议 $v$ 的请求：

$$
Q_i^{(v)}
=
\operatorname{Adapt}_v(M_i)
$$

$Q_i^{(v)}$ 表示第 $i$ 次 LLM call。它可以包含消息、工具定义和生成参数；其中只有参与上下文构造的字段会影响模型输入 token。

> **以 Pi 为例**
>
> Pi 的内部模型上下文包含 system prompt、 `AgentMessage[]` 和工具定义。每次调用前，Pi 先把内部消息转换为模型可用的统一消息，再由 provider adapter 组织为 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 或 Google Generative AI 等协议请求。这里将这些步骤统一记为 $\operatorname{Adapt}_v$ 。

推理服务器再根据协议和模型完成输入编码，得到模型实际接收的 token ID 序列：

$$
p_i^{\mathrm{tok}}
=
\operatorname{Encode}_{m,v}\left(Q_i^{(v)}\right)
$$

其中， $m$ 表示模型， $\operatorname{Encode}_{m,v}$ 表示协议和模型相关的输入编码过程。对于显式使用 chat template 的推理服务器，这一步可以进一步分解为：

$$
p_i^{\mathrm{text}}
=
S_{m,v}\left(Q_i^{(v)}\right)
$$

$$
p_i^{\mathrm{tok}}
=
\operatorname{Tok}_m\left(p_i^{\mathrm{text}}\right)
=
\operatorname{Tok}_m\left(
S_{m,v}\left(Q_i^{(v)}\right)
\right)
$$

这里的 $S_{m,v}$ 表示 chat template 定义的序列化过程， $\operatorname{Tok}_m$ 表示模型对应的 tokenizer。对于没有公开 chat template 的托管推理服务， $\operatorname{Encode}_{m,v}$ 仍然成立，但其内部序列化过程不可直接观察。

模型以 $p_i^{\mathrm{tok}}$ 为输入，自回归生成输出 token：

$$
a_i^{\mathrm{tok}}
\sim
\pi_\theta\left(\cdot \mid p_i^{\mathrm{tok}}\right)
$$

推理服务器将生成 token 转换为文本：

$$
a_i^{\mathrm{text}}
=
\operatorname{Detok}_m\left(a_i^{\mathrm{tok}}\right)
$$

协议相关的输出解析过程 $P_v$ 将生成文本转换为 harness 使用的结构化响应 $R_i$ 。其中可以包含文本、推理内容和工具调用等字段：

$$
R_i
=
P_v\left(a_i^{\mathrm{text}}\right)
$$

Agent harness 接收到 $R_i$ 后，结合可能存在的 environment observation 或 tool result $o_i$ ，更新内部状态：

$$
M_{i+1}
=
U(M_i, R_i, o_i)
$$

其中， $U$ 表示 agent harness 的状态更新函数。它根据上一轮状态 $M_i$ 、模型响应 $R_i$ 和工具结果或环境观测 $o_i$ 构造下一轮状态 $M_{i+1}$ 。具体过程可以包括追加 assistant message、写入 tool result，以及执行上下文裁剪或压缩。

因此，一次跨轮调用的完整数据流可以写为：

$$
M_i
\xrightarrow{\operatorname{Adapt}_v}
Q_i^{(v)}
\xrightarrow{\operatorname{Encode}_{m,v}}
p_i^{\mathrm{tok}}
\xrightarrow{\pi_\theta}
a_i^{\mathrm{tok}}
\xrightarrow{\operatorname{Detok}_m}
a_i^{\mathrm{text}}
\xrightarrow{P_v}
R_i
\xrightarrow{U}
M_{i+1}
$$

### 多轮 rollout 与上下文重建

单次调用的表示可以扩展为多轮 rollout。令 $\rho$ 表示一次 agent 与环境交互的轨迹，其 message-level 记录为：

$$
\mathcal{C}^{\mathrm{msg}}(\rho)
=
\left(
(M_1, R_1, o_1),
(M_2, R_2, o_2),
\ldots
\right)
$$

其中， $M_i$ 是第 $i$ 次模型调用前的内部消息状态， $R_i$ 是该次调用返回的结构化响应， $o_i$ 是随后产生的工具结果或环境观测，也可以为空。将每次调用的文本输入与生成文本配对，可以得到 text-level 记录：

$$
\mathcal{C}^{\mathrm{text}}(\rho)
=
\left(
(p_1^{\mathrm{text}}, a_1^{\mathrm{text}}),
(p_2^{\mathrm{text}}, a_2^{\mathrm{text}}),
\ldots
\right)
$$

某些 agent harness 以只追加方式保存 session 历史，但 $M_i$ 表示调用前实际参与上下文构造的内部状态。该状态仍可能经过消息过滤、上下文裁剪、摘要压缩或动态信息注入。因此，session 历史的只追加存储不能保证模型输入保持前缀关系。要使文本输入保持前缀关系，需要排除上述上下文变换；协议转换不能改写已有消息，system prompt、工具定义和 chat template 也需要保持固定。

> **Pi 中的 session 与调用上下文**
>
> Pi 将 session 保存为 append-only JSONL tree。恢复 session、完成 compaction 或切换分支时，Pi 会沿当前分支构造有效消息；compaction 会使用摘要和保留消息替代较早消息。正常的逐轮调用使用当前内存消息，并在请求前先执行可选的 context transform，再执行消息格式转换。因此，session 文件的 append-only 性质不能保证各轮 LLM call 保持前缀关系。

## RL 训练

### 模型调用轨迹

一次完整的 agentic rollout $\rho$ 包含模型调用、harness 状态变化、工具或环境交互以及最终奖励。训练系统在模型边界直接观察到的部分可以表示为：

$$
\mathcal{C}^{\mathrm{tok}}(\rho)
=
\left(
(p_1^{\mathrm{tok}}, a_1^{\mathrm{tok}}),
(p_2^{\mathrm{tok}}, a_2^{\mathrm{tok}}),
\ldots
\right)
$$

其中，每一项表示一次模型调用实际使用的输入 token 和采样得到的输出 token。 $\mathcal{C}^{\mathrm{tok}}(\rho)$ 是 rollout 的模型调用轨迹，不包含未经过模型边界的环境状态变化。

令 $a_{i,t}^{\mathrm{tok}}$ 表示第 $i$ 次调用生成的第 $t$ 个 token， $\pi_{\mathrm{roll}}$ 表示实际执行采样的 rollout policy，则：

$$
a_{i,t}^{\mathrm{tok}}
\sim
\pi_{\mathrm{roll}}\left(
\cdot
\mid
p_i^{\mathrm{tok}},
a_{i,<t}^{\mathrm{tok}}
\right)
$$

若训练目标依赖采样策略概率，还需要保存或可复现地计算采样 token 在 rollout policy 下的 log probability：

$$
\ell_{i,t}^{\mathrm{roll}}
=
\log \pi_{\mathrm{roll}}\left(
a_{i,t}^{\mathrm{tok}}
\mid
p_i^{\mathrm{tok}},
a_{i,<t}^{\mathrm{tok}}
\right)
$$

训练样本至少需要准确恢复每次调用的输入 token、输出 token 和 response/action mask；该 mask 用于标记参与损失计算的生成位置。从结构化消息或文本重建 token 时，需要固定 tokenizer、chat template 和协议转换版本。若训练目标使用旧策略概率，还需要记录对应的 log probability，或保留 rollout policy checkpoint 后重新计算。

异步采样时还需要记录实际使用的策略版本。trainer 可以据此判断轨迹与当前参数之间的关系，并按照所选训练目标决定样本是否可用；模型名称不能替代策略版本。

> **通过 serving gateway 采集调用轨迹**
>
> 通过 Pi、Claude Code 或 OpenClaw 等现成 harness 进行黑盒 RL 时，常见做法是在 harness 与 rollout inference service 之间部署协议兼容的 serving gateway 或 proxy。harness 只需将 OpenAI Chat Completions、OpenAI Responses 或 Anthropic Messages 等客户端指向该服务，原有的工具调用、上下文管理和环境交互逻辑可以保持不变。
>
> gateway 将协议请求转换为推理服务调用，并使用 rollout ID、attempt ID 和模型版本标识每次请求。对于每次模型调用，它需要从推理服务取得或直接控制实际输入 token、采样输出 token，以及训练目标所需的 rollout log probability。response/action mask 可以在后续构造样本时，根据模型输出与环境输入的 token 来源生成。gateway 同时将协议响应返回给 harness，并将 token 级调用记录写入 trajectory store，供训练系统读取。
>
> 只记录结构化消息或响应文本的普通 API gateway 不能恢复可靠的训练样本。gateway 需要使用与推理服务一致的 tokenizer 和 chat template 生成 `input_ids` ，或者由推理服务返回其实际使用的 token ID 和生成概率。白盒 agent loop 也可以在自定义 rollout 函数中直接完成这些记录，因此独立 gateway 是黑盒 harness 的常见接入方式，并非 Agentic RL 的必要组件。

### 跨轮重编码与轨迹构造

下一轮的 $p_{i+1}^{\mathrm{tok}}$ 通常不是通过直接拼接 $p_i^{\mathrm{tok}}$ 和 $a_i^{\mathrm{tok}}$ 得到的。模型生成 $a_i^{\mathrm{tok}}$ 后，系统会依次经历 detokenization、输出解析、内部状态更新、协议转换和输入编码：

$$
a_i^{\mathrm{tok}}
\xrightarrow{\operatorname{Detok}_m}
a_i^{\mathrm{text}}
\xrightarrow{P_v}
R_i
\xrightarrow{U}
M_{i+1}
\xrightarrow{\operatorname{Adapt}_v}
Q_{i+1}^{(v)}
\xrightarrow{\operatorname{Encode}_{m,v}}
p_{i+1}^{\mathrm{tok}}
$$

在前文所列的上下文不变条件成立，且序列化可以简化为直接文本拼接时，文本序列可能满足：

$$
p_i^{\mathrm{text}}
\Vert
a_i^{\mathrm{text}}
\preceq_{\mathrm{text}}
p_{i+1}^{\mathrm{text}}
$$

其中， $\Vert$ 表示序列拼接， $\preceq_{\mathrm{text}}$ 表示精确的字符串前缀关系。该关系不能推出 token 序列满足：

$$
p_i^{\mathrm{tok}}
\Vert
a_i^{\mathrm{tok}}
\preceq_{\mathrm{tok}}
p_{i+1}^{\mathrm{tok}}
$$

chat template 可能在消息边界插入控制 token，detokenization 后重新编码也可能改变 token 切分，协议解析还可能规范化模型输出。因此，上式是线性合并相邻调用训练样本时需要满足的条件，并非多轮调用的一般性质。

将每次调用作为独立样本可以直接保持上述条件。线性合并、serving 阶段的 token buffer 和训练阶段的 prefix tree 也可以减少重复计算，但都需要保持每个生成 token 在 rollout 时实际使用的条件序列。训练阶段事后替换下一轮真实输入中的 token 会改变该条件序列。

### 轨迹聚合、评分与训练输入

模型调用轨迹还不是 trainer 可以直接使用的训练批次。从原始 token 记录到参数更新，需要依次建立调用、rollout、rollout group 和训练样本之间的映射：

```text
token-level call records
        ↓
complete rollout records
        ↓
rollout groups
        ↓
verifier scores
        ↓
reward signals
        ↓
credit assignment
        ↓
training samples
        ↓
trainer
```

首先，将同一次 agent 与环境交互产生的模型调用、环境轨迹和版本信息聚合为完整 rollout 记录：

$$
\mathcal{R}_\rho
=
\left(
x_\rho,
\mathcal{C}^{\mathrm{tok}}(\rho),
e_\rho,
v_\rho
\right)
$$

其中， $x_\rho$ 表示任务输入， $\mathcal{C}^{\mathrm{tok}}(\rho)$ 表示模型调用轨迹， $e_\rho$ 表示环境状态、工具结果和最终产物等可验证记录， $v_\rho$ 表示 rollout policy 及相关编码组件的版本。只有调用和环境记录均已完成， $\mathcal{R}_\rho$ 才能进入评分阶段。

训练系统再按照任务、初始状态或采样批次等预先定义的分组键，将多个完整 rollouts 聚合为 rollout group：

$$
\mathcal{G}_g
=
\left\{
\mathcal{R}_\rho
\mid
g(\rho)=g
\right\}
$$

分组定义了哪些 rollouts 可以共同参与比较、归一化或批处理。它不要求 verifier 必须执行组内相对评分：verifier 可以独立评估每条 rollout，也可以结合整个 group 进行排序、成对比较或一致性检查。将评分过程抽象为：

$$
\mathcal{S}_g
=
\operatorname{Score}(\mathcal{G}_g)
$$

$\mathcal{S}_g$ 保存每条 rollout 对应的原始评分记录。评分可以是标量、多个指标或结构化判定；原始评分应当独立保存，使后续奖励变换可以复查，而不必重新执行环境或 verifier。

原始评分经过训练配置定义的组合、过滤或归一化，得到奖励记录：

$$
\mathcal{Y}_g
=
\operatorname{Reward}
\left(
\mathcal{G}_g,
\mathcal{S}_g
\right)
$$

$\operatorname{Reward}$ 只负责把 verifier 的原始输出转换为训练使用的奖励，同时保留从每项奖励到原始评分的映射。信用分配再决定这些奖励如何作用于 rollout 中的不同决策：

$$
\mathcal{K}_g
=
\operatorname{Credit}
\left(
\mathcal{G}_g,
\mathcal{Y}_g
\right)
$$

$\mathcal{K}_g$ 可以包含 rollout-level、call-level 或 token-level 的权重、回报或优势。评分、奖励变换与信用分配分开后，同一组原始结果可以用于比较不同训练方法，也能明确区分环境评价与算法假设。

样本构造器最后结合真实 token 轨迹和训练信号，生成 trainer 输入：

$$
\mathcal{D}_g
=
\operatorname{Build}
\left(
\mathcal{G}_g,
\mathcal{S}_g,
\mathcal{Y}_g,
\mathcal{K}_g
\right)
$$

每个训练样本需要保留 `group_id` 、 `rollout_id` 、策略版本、输入与输出 token、response/action mask，以及所选训练目标需要的概率或权重。 `Build` 可以按模型调用拆分样本，也可以在满足 token 条件一致性时合并样本；样本布局属于数据构造问题，不应隐式改变评分或信用分配结果。

trainer 接收多个 groups 产生的样本，并按照具体训练算法计算目标函数和更新参数。该边界允许不同算法复用同一套 rollout、评分和样本追踪接口，同时保留以下检查条件：

1. 每个参与训练的生成 token 都能回指产生它的模型调用和真实条件序列。
2. 每个 rollout 都能回指所属 group、原始评分、训练信号和策略版本。
3. 奖励变换与信用分配具有明确配置，原始 verifier 评分保持不变。
4. 轨迹拆分、合并或打包不会在未定义的情况下改变 rollout 在训练批次中的权重。
