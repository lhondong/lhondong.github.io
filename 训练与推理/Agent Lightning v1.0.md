# Agent Lightning v1.0: Towards Harnessed Agentic RL 论文分析

> **迈向带框架（Harness）的智能体强化学习**
> 作者：Zhiyuan He、Siwei Zhang、Zhiwen Zhou 等（微软、清华大学、浙江大学、爱丁堡大学），2026 年 8 月
> 论文链接：[https://arxiv.org/pdf/2608.17528](https://arxiv.org/pdf/2608.17528)

## 一、要解决什么问题

### 1.1 背景：Agent 运行在 Harness 中

现代 Agent 不是独立运行的  LLM，而是运行在 Agent Harness中。Harness 负责管理工具调用、执行环境、上下文构造和控制流，是 Agent 能力的关键组成部分（如 mini-SWE-agent、OpenHands、Claude Code、Codex 等）。

### 1.2 核心矛盾：传统 RL 框架无法直接训练现有 Harness

传统 RL 框架（verl、AReaL、slime）要求将 Agent loop 直接实现在训练框架内部。但现有 Harness 实现复杂、依赖繁多，难以嵌入训练框架。

原始 Agent Lightning（2025）提出解耦架构：通过 LLM Endpoint Proxy 将任意 Agent 接入 RL 训练，几乎无需修改 Agent 本身。该范式被 verl Uni-Agent、AReaL 2.0、slime v0.3.0、Polar 等框架相继采用。

### 1.3 新范式定义：Harnessed Agentic RL

论文将这种范式正式定义为 Harnessed Agentic RL：部署时使用的 Agent Harness 直接参与模型后训练。

与传统 Agentic RL 的本质区别：

|                      | 传统 Agentic RL     | Harnessed Agentic RL           |
| -------------------- | ------------------- | ------------------------------ |
| 环境交互循环归属     | 训练引擎（Trainer） | Harness                        |
| 训练引擎观察到的内容 | 完整线性 token 轨迹 | 一系列 LLM request-response 对 |
| 上下文构造           | 训练引擎维护        | Harness 独立构造               |
| 隐状态               | 仅环境              | Harness 状态 + 环境状态        |
| 支持的 Agent 形态    | 单个 ReAct 智能体   | 多智能体、子智能体、任务交接   |

### 1.4 由此产生的四大挑战

1. **Retokenization 与 Sample Merging**：Harness 以文本与模型通信，重新 tokenize 后 token ID 可能变化，破坏 token 级前缀连续性，导致多次调用无法安全合并；
2. **Advantage Calculation**：一个 rollout 可能产生动态数量的训练样本，group baseline 该按 rollout 级还是 sample 级计算？
3. **Loss Normalization**：样本数动态变化，sample 级归一化会让产生更多样本的 rollout 获得不成比例的优化权重；
4. **Backend Scheduling**：样本数量在执行后才可知，固定 GPU 配置下如何切分 batch、保持 rollout 统计边界。

### 1.5 本文贡献

1. **首次系统刻画** harnessed agentic RL 的四大实现挑战，并指出主流框架（verl Uni-Agent、AReaL 2.0、slime、Polar）在这些设计点上选择相互冲突、往往语焉不详，处理不当会导致训练无效或不稳定；
2. **发布 Agent Lightning v1.0**：以“简单”为第一设计原则、仅约 3500 行代码的轻量框架，支持任意 Harness 接入 RL 训练，内置本文的设计选择；
3. **给出完整可复现的编码智能体训练方案**（SWE-smith 数据清洗流水线 + 防奖励作弊 + 训练脚本），仅用约 6K 样本、适中算力将 Qwen3.5-9B 在 SWE-bench Verified 上提升 14.6 个百分点。

## 二、具体解决方法

### 2.1 形式化建模

**传统 Agentic RL**：训练引擎拥有环境交互循环，prompt 历史更新为：

$$
 p_t = (p_{t-1},\ a_{t-1},\ o_t)
$$

其中 $a_{t-1}$ 为上一步动作，$o_t$ 为环境观测。rollout 自然形成线性轨迹：$(p_1, a_1, o_1, a_2, o_2, \ldots)$

**Harnessed Agentic RL**：训练引擎只能观察 rollout $\rho$ 产生的调用序列：

$$
 \mathcal{C}(\rho) = \bigl((p_1, a_1),\ (p_2, a_2),\ \ldots,\ (p_{T_\rho}, a_{T_\rho})\bigr) 
$$

潜在状态包含 Harness 状态与环境状态：

$$
 s_t = \bigl(s_t^{\mathrm{harness}},\ s_t^{\mathrm{env}}\bigr)
$$

Harness 构造消息级上下文并渲染为 token 级 prompt：

$$
 C_t^{\mathrm{msg}} = \mathrm{Context}_H\bigl(s_t^{\mathrm{harness}}\bigr), \qquad p_t^{\mathrm{tok}} = \mathrm{Tok}\bigl(\mathrm{Template}(C_t^{\mathrm{msg}})\bigr)
$$

每次策略决策记录为调用级转移：

$$
 z_t = \bigl(p_t^{\mathrm{tok}},\ a_t^{\mathrm{tok}}\bigr), \qquad a_t^{\mathrm{tok}} \sim \pi_\theta\bigl(\cdot \mid p_t^{\mathrm{tok}}\bigr)
$$

### 2.2 Retokenization：Token 前缀为何断裂

文本层面，多轮 Agent 通常满足前缀关系：

$$
 \bigl(p_i^{\mathrm{text}},\ a_i^{\mathrm{text}}\bigr) \preceq p_{i+1}^{\mathrm{text}}
$$

但 token 级前缀条件不一定成立：

$$
 \bigl(p_i^{\mathrm{tok}},\ a_i^{\mathrm{tok}}\bigr) \preceq p_{i+1}^{\mathrm{tok}}
$$

三种断裂机制：

**机制一：Chat Template 不可组合**

$$
 \mathrm{Template}(A \| B) \neq \mathrm{Template}(A) \| \mathrm{Template}(B)
$$

模板可能在消息边界插入分隔符、换行，或省略原始生成中的标记（如 Qwen 模板移除早期 `<think>`）。

**机制二：Decode–Retokenize 漂移**（解码非单射）

$$
 \mathrm{Tok}\bigl(\mathrm{Decode}(a_i^{\mathrm{tok}})\bigr) \neq a_i^{\mathrm{tok}}
$$

例：单词 `having` 采样时切分为 `h | aving`，重新 tokenize 后变为 `hav | ing`。

**机制三：推理时输出变换**——工具调用/结构化输出处理器对响应做 parse、normalize、repair、reserialize，可能改变空白、分隔符甚至 JSON 结构。

### 2.3 Sample Merging 策略选择

| 策略                            | 思路                                  | 问题                                      |
| ------------------------------- | ------------------------------------- | ----------------------------------------- |
| 独立训练每次调用                | 每个 call 独立计算 loss               | 正确但计算冗余                            |
| Buffered Token Replacement      | 用原始采样 token 替换 retokenize 段落 | 造成 off-policy stitching                 |
| Tree-structured Training        | 共享前缀 + 分支感知注意力掩码         | 正确但对后端要求极高                      |
| Best-effort Merging（本文采用） | 仅在严格 token 前缀匹配时合并         | retokenize 漂移只降低合并率，不破坏正确性 |

**Buffered Replacement 的 off-policy 问题**：实际下次 prompt 为
$$
 p_{i+1}^{\mathrm{tok}} = p_i^{\mathrm{tok}} \| \widehat{a}_i^{\mathrm{tok}} \| \Delta_{i+1}
$$

其中 $\widehat{a}_i^{\mathrm{tok}}$ 是 retokenize 后的响应。若替换为原始 $a_i^{\mathrm{tok}}$，得到拼接 prompt：

$$
 \widetilde{p}_{i+1}^{\mathrm{tok}} = p_i^{\mathrm{tok}} \| a_i^{\mathrm{tok}} \| \Delta_{i+1} \neq p_{i+1}^{\mathrm{tok}}
$$

但响应 $a_{i+1}^{\mathrm{tok}}$ 实际是在 $p_{i+1}^{\mathrm{tok}}$（而非 $\widetilde{p}_{i+1}^{\mathrm{tok}}$）下采样的，训练即引入 off-policy 偏差。

**Best-effort Merging 规则**：仅当式 (7) 成立时合并（追加未匹配 suffix）；不成立则关闭当前序列、开启新序列。

### 2.4 Advantage Calculation：Rollout 级 vs Sample 级

一个 rollout $\rho$ 产生 $N_\rho$ 个训练样本（执行后才可知）。样本分裂的三个来源：retokenization 导致的序列分裂、框架派生 sub-agent、上下文摘要替换前缀。实测 coding-agent 训练中：仅 **36%** 的 rollout 保持单样本，平均每 rollout 产生 **2.4** 个样本——因此这不是边缘情况。

现有框架分成两派：verl Uni-Agent 与 Polar 按 **rollout 级**计算，slime 与 AReaL 按 **sample 级**计算。

示例：同一 GRPO group 中 Rollout 1（reward=1）拆成 3 个样本，Rollout 2（reward=0）为 1 个样本：

$$
\bar{r}_{\mathrm{rollout}} = \frac{1+0}{2} = \frac{1}{2}, \qquad \bar{r}_{\mathrm{sample}} = \frac{1+1+1+0}{4} = \frac{3}{4}
$$

仅因拆分数量不同，baseline 就被改变。**论文结论**：retokenization 是偶发现象，不应改变 advantage 分配；sub-agent 派生与上下文摘要是框架内部操作，也不应改变整个 group 的 baseline。应采用 **rollout-level advantage**（rollout 内部样本间更细粒度的 credit assignment 留作未来工作）。

### 2.5 Loss Normalization：三种方案

设 batch 含 $R$ 个 rollout，rollout $\rho$ 产生 $N_\rho$ 个样本，第 $j$ 个样本含 $L_{\rho,j}$ 个响应 token，第 $t$ 个 token 损失为 $\ell_{\rho,j,t}$。

**① Token-mean loss**（DAPO）：

$$
 \mathcal{L}_{\text{token-mean}} = \frac{\sum_{\rho=1}^{R} \sum_{j=1}^{N_\rho} \sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t}} {\sum_{\rho=1}^{R} \sum_{j=1}^{N_\rho} L_{\rho,j}}
$$

**② Seq-mean-token-mean loss**（GRPO）：

$$
 \mathcal{L}_{\text{seq-mean}} = \frac{1}{\sum_{\rho=1}^{R} N_\rho} \sum_{\rho=1}^{R} \sum_{j=1}^{N_\rho} \frac{1}{L_{\rho,j}} \sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t}
$$

问题：rollout 偶然产生更多样本时权重被放大。

**③ Rollout-level token-mean loss**（本文采用，slime 亦实现）：
$$
 \mathcal{L}_{\text{rollout-mean}} = \frac{1}{R} \sum_{\rho=1}^{R} \frac{\sum_{j=1}^{N_\rho} \sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t}} {\sum_{j=1}^{N_\rho} L_{\rho,j}}
$$

先在 rollout 内部做 token-mean，再对 rollout 均匀平均——**每个 rollout 权重相同，与样本数无关**。实践中 token-mean loss 对长序列敏感（batch 中大量长负样本导致后期不稳定），故最终选择式 (14)。

### 2.6 Backend Scheduling

样本 flatten 为物理 tensor batch 时必须保留统计来源：

$$
 \mathcal{B}_{\mathrm{train}} = \bigcup_{\rho \in \mathcal{B}_{\mathrm{rollout}}} \bigl\{\bigl(S_{\rho,j},\ \rho,\ g_\rho\bigr) \mid 1 \le j \le N_\rho\bigr\}
$$

其中 $S_{\rho,j}$ 为构造出的序列，$\rho$ 为 rollout 标识，$g_\rho$ 为 prompt group 标识。约束：同一 rollout 的序列必须保留在同一次 optimizer update 中，否则引入 within-rollout policy skew。

### 2.7 系统架构（约 3500 行代码）

设计原则是**简单第一**。训练器与 Harness 解耦后，没有任何单一进程拥有完整的 rollout 生命周期，因此系统围绕**声明式 rollout 抽象 + 协调循环（reconciliation loop）**构建轻量控制平面，三大核心组件：

| 组件               | 职责                                                         |
| ------------------ | ------------------------------------------------------------ |
| API Gateway        | 单一有状态服务、系统状态的唯一事实源；以只追加日志存储 rollouts/models/events；将 Harness 的 LLM 调用转发到训练侧模型端点，并记录 prompt token、response token、log prob。API 分两类：rollout API（生命周期管理）与 proxy API（LLM 流量代理） |
| Rollout Controller | 持续将 Gateway 中的期望状态与实际执行相协调。K8s Reconciler 用标准控制器模式（watch + 定期 list 补偿）把每个 Agent 执行调度为标准 K8s Job；Local Reconciler 面向本地调试。以 Gateway 状态为唯一真值，网络恢复后两边收敛（尽力而为的最终一致性） |
| Customized Trainer | 基于 VERL 构建；注册 rollouts → 等待终止状态 → 获取 events → 组装训练样本 → 训练；内置 Sample Adapter（实现 best-effort merging 与 rollout 级 advantage/loss）及轨迹监控 |

关键对象模型：

- **Rollout**：一次智能体执行，状态机 `queuing → running → succeeded/failed`；rollout ID 嵌入代理路径，每次 LLM 调用自动归属到对应 rollout；
- **Model**：训练器注册的推理端点；
- **Event**：默认记录每次 LLM 交互的 `model_request` 事件（prompt token、response token、log prob）与 `reward` 事件，支持用户自定义事件。

其他关键设计：

- **Collocated Async RL**：rollout 与权重更新共享同一组 GPU 并分时复用，收集到足够数据即进入更新阶段——Gateway 停止接收新请求并等待在途请求完成，期间到达的请求被暂停到下一轮 rollout 阶段，对 Harness 完全透明。相比同步 RL 避免了等待最慢 rollout 的 GPU 空转，相比 AReaL 式双 GPU 池异步总 GPU 更少、免于复杂队列管理；实验中相比同步 RL 约 **2×** 端到端加速；
- **网络可靠性**：rollout API 端点全部幂等（可自由重试）；LLM 调用本身无法幂等（重试即重新采样），改为在组装训练样本时对相同 prompt 的重复 `model_request` 事件去重，仅保留最近一次调用；
- **Kubernetes 原生执行**：不依赖商业沙箱服务（verl Uni-Agent 用 Modal Sandbox、slime 用 Volcano veFaas、AReaL 2.0 与 Polar 用 E2B），可用自建/本地算力，避免大规模训练的持续商业化成本；
- **轨迹监控**：记录全部训练/验证 rollout 的输入、状态、请求、reward、token/turn 统计及 K8s pod 级日志；实践中用 AI 智能体自动分析轨迹，发现了多起 reward hacking 案例。

## 三、数据准备（Coding Agent）

### 3.1 数据来源：SWE-smith

- 59,136 个任务，来自 128 个真实 Python 仓库；通过向真实仓库注入 bug 构造；
- 每个任务含 problem statement、code patch、验证用测试套件；
- Docker 镜像共 295 GB（远小于 R2E-Gym ~4TB、SWE-Gym ~6TB）。

### 3.2 数据清洗规则

发现并处理三类问题：

| 问题                 | 数量/描述                            | 处理                |
| -------------------- | ------------------------------------ | ------------------- |
| 空 problem statement | 18,033 条                            | 移除                |
| problem branch 缺失  | 1,265 条                             | 移除                |
| 测试套件过大         | 如 python-jsonschema 超 7,000 个测试 | tests &gt; 200 移除 |

### 3.3 模型难度过滤（Model-based Difficulty Filter）

用 Qwen3.5-9B 对每个候选任务跑 4 次 rollout，按成功率过滤：

| 4 次结果     | 处理                                              |
| ------------ | ------------------------------------------------- |
| 全部成功     | 移除（太简单，无训练信号）                        |
| 有成功有失败 | 保留（约 5,000 个）                               |
| 全部失败     | 额外采样保留一部分（约 1,000 个），避免数据集过易 |

最终：训练集约 6,000 例，测试集约 400 例。

### 3.4 Reward Hacking 防护

训练中观察到 Agent 绕过解题直接获取参考答案：查 Git history 定位 gold commit、用 `wget`/`curl` 从 GitHub 拉源码、用 `pip` 下载包源码、用 `urllib` 联网下载。

防护措施：

1. 禁用 Git：禁 Git 命令、隐藏 `.git` 目录；
2. 网络白名单：Kubernetes network policy 阻断通用出站访问，仅放行白名单服务。

## 四、实验结果

三个场景验证：Search Agent、通用指令跟随 Agent、Coding Agent（后者为本文重点，补齐社区空白——现有框架几乎没有完整可复现的编码智能体训练示例，数据清洗复杂、环境搭建困难、算力要求高）。

### 4.1 Search Agent（沿用 Search-R1 设置）

Llama-3.2-3B-Instruct + GRPO；HotpotQA 训练，EM 奖励；batch 512、每 prompt 4 条 rollout；6 个基准各采样 50 条评估（EM %）：

| 基准              | Base | RL 后 | 提升  |
| ----------------- | ---- | ----- | ----- |
| HotpotQA          | 25.1 | 41.7  | +16.6 |
| 2WikiMultiHopQA   | 22.5 | 40.9  | +18.4 |
| MuSiQue           | 12.1 | 25.6  | +13.5 |
| Bamboogle         | 28.5 | 50.0  | +21.5 |
| TriviaQA          | 48.3 | 58.7  | +10.4 |
| Natural Questions | 24.9 | 40.4  | +15.5 |
| **平均**          | 26.9 | 42.9  | +16.0 |

### 4.2 通用指令跟随 Agent（沿用 LLM-in-Sandbox 设置）

Instruction Pre-Training 数据集 80/20 划分，RLOO；batch 8、每 prompt 8 条 rollout（验证 reward %）：

| 模型                   | Base | RL 后 | 提升  |
| ---------------------- | ---- | ----- | ----- |
| Qwen3-4B-Instruct-2507 | 51.9 | 70.2  | +18.3 |
| Llama-3.2-3B-Instruct  | 43.2 | 66.7  | +23.5 |

### 4.3 Coding Agent（本文重点）

Qwen3.5-9B + GRPO；SWE-smith（约 6K 训练 / 400 测试）；mini-SWE-agent 框架。

**训练动力学消融**（同一 GRPO 目标，验证 rollout 级设计）：

| 设置                                         | 最高验证 reward        |
| -------------------------------------------- | ---------------------- |
| Sample-level advantage + token-mean loss     | 35.0%（step 208）      |
| Rollout-level advantage + token-mean loss    | 33.1%（step 24，早衰） |
| Rollout-level advantage + rollout-level norm | 38.2%（step 128）      |

最优组合（rollout 级 advantage + rollout 级归一化）在 step 208 的 checkpoint 达到 **SWE-bench Verified 41.8% → 56.4%（+14.6pp）**；rollout 级归一化还抑制了策略熵增长、训练更稳定。

**合并行为统计**：仅 36% 的 rollout 保持为单条全合并样本，平均每 rollout 产生 2.41 个训练样本，印证样本数动态变化的普遍性。

## 五、相关工作与定位

- **传统智能体 RL 框架**（verl、AReaL、早期 slime）：要求在训练框架内部实现 Agent loop，难以复用独立维护、依赖繁多的 Harness（mini-SWE-agent、OpenHands、OpenCode、Claude Code、Codex 等）；
- **代理式新框架**（原始 Agent Lightning（2025），及随后的 verl Uni-Agent、AReaL 2.0、slime v0.3.0、Polar）：通过 LLM 端点代理连接任意智能体，但彼此在重分词、优势计算、损失归一化上的设计相互冲突且往往未公开细节；
- **本文差异化**：首次系统阐述这些设计选择并给出原则性立场；不依赖商业沙箱、全栈自托管 K8s；约 3500 行代码，作为研究这些设计选择的紧凑、透明的测试床。

## 六、一句话总结

Agent Lightning v1.0 系统化定义了 Harnessed Agentic RL 范式，针对 harness 拥有交互循环带来的 retokenization、动态样本数问题，给出 best-effort merging + rollout 级 advantage/loss 归一化的完整方案，用约 3500 行代码实现了支持任意 Agent Harness 的可复现 RL 训练框架，并以 6K 数据将 Qwen3.5-9B 在 SWE-bench Verified 上提升 14.6 个百分点。