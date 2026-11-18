# Market Maker Alpha

把 alpha 理解为“未来中间价对当前中间价的预测偏移”，度量单位建议用 tick 或 bps。一个稳健的报价结构是把“基准价（reservation price）”与“非对称边际价差”分开：

- 预测偏移（含库存与微观结构修正）：

$$
r_t = mid_t + k_1\,\alpha_t - \phi\,Inv_t + \psi\,(MicroBias_t)
$$

其中 $Inv_t$ 是库存（用名义张数或美元），$\phi$ 为库存风控系数，$MicroBias_t$ 可用 microprice 偏移等。

- 非对称价差（把 alpha 变为“哪一边更积极/更保守”）：

$$
s^{bid}_t = s_0 + \gamma\,\sigma_{H} - \eta\,\alpha_t + Fees + AS_{penalty} \\
s^{ask}_t = s_0 + \gamma\,\sigma_{H} + \eta\,\alpha_t + Fees + AS_{penalty}
$$

其中 $\sigma_{H}$ 为目标持有期 H 的波动（tick），$AS_{penalty}$ 为逆向选择的保守项。

- 最终报价：

$$
bid_t = r_t - \frac{s^{bid}_t}{2},\qquad
ask_t = r_t + \frac{s^{ask}_t}{2}
$$

这样当 $\alpha_t>0$（看涨）时：两边的报价整体上移（更愿意持多），同时 bid 更紧、ask 更松（更想买、少想卖）。这比“mid * alpha * scaling ± skew”的一条式子更稳健：$k₁$ 对应你的 scaling，$\eta$ 对应“按方向改变两侧紧松的 skew”，而 $\phi$ 则是库存偏移。

小技巧

- 对 $\alpha_t$ 做 clip（如 $|\alpha_t|\le 3\sigma_{H}$）+ tanh 压缩，避免极值导致过度收敛/扩张。
- 在线标定 $k_1,\eta,\phi,\gamma$ 使单位风险 Sharpe 或每笔期望利润最大，且满足撤单率/成交率/库存上限约束。


## 五大类 alpha

### 1. ML 预测 20–60s 收益（回归）

目标：预测 $y_t=\log(mid_{t+H})-\log(mid_t)$ 或以 tick 计的 $\Delta p_{t\to t+H}$，$H$ 常取 20–60s。

特征：L2 订单簿快照（多档价量、价差、队列位置）、OFI/OBI、短期动量、成交簇强度、波动、跨所价、资金费率/资金流入（加密）、宏微合成因子等。

模型：树模型（RF/XGBoost/LightGBM）最主流；为降低延迟，本地 C++/Rust 推理或 tiny 模型蒸馏；线上可做增量学习（样本权衰减）。

映射到出价：$\alpha_t = \mathbb{E}[\Delta p|X_t]$（tick），直接进上面的 $r_t, s^{bid/ask}_t$。

优点：可综合多源信息，信号稳健、覆盖面广。

缺点：决策延迟比纯规则大几毫秒；训练/更新有成本；需要严格反偷看切分与滑点/队列回放。

要点：

- 用 isotonic regression 或 Platt 标定回归输出到“期望 tick”，避免过度自信。
- 线上尽量 0 拷贝特征（共享内存）、批内向量化，把推理压到 < 0.5–1ms；再慢就会“跟在行情后面”。
- 选择 H：既要覆盖你订单平均存活时间，又要与撤单策略一致（如挂单最多活 2–5s，H 取 5–15s 更贴合）。

### 2. ML 预测“归一化方向”，做单腿抢盘口（分类）

目标：预测 $p_t = \mathbb{P}(\Delta p>0|X_t)$ 或离散标签（up/flat/down）。

用法 1（做市）：把 $p_t$ 映射为方向性偏移 $\alpha_t = \mu_{H}\cdot(2p_t-1)$（$\mu_H$ 为同类样本的平均绝对变动 tick），再走“非对称价差”：$\eta \alpha_t$ 让有利方向更紧。

用法 2（taker 单腿）：若 $p_t>0.5+\theta$ 直接抢一腿，或在有利方向Join/Improve 最优价，另一侧少挂/不挂。

优点：决策面更简洁，容错性强。

缺点：纯方向往往需要更快的执行与撤单；异动时容易过度追涨杀跌。

要点：
- 方向概率要做 温度校准（ECE 小很重要）；
- 作为做市的“gate”：当 $p_t$ 置信不足时自动回退到中性对称报价，减少抖动；
- 延迟优化同上，尽量用“小而快”的树/线模 + AOT 编译。

### 3. Layer-2 套餐：OBI/OFI/动量/强度/VPIN

这类几乎是“万王之王”的基础盘面 alpha，低延迟、可解释、可并行。

- OBI（Order Book Imbalance）：多档加权不平衡。常用一级形式：

$$
OBI^{(1)}=\frac{Vol_{bid,1}-Vol_{ask,1}}{Vol_{bid,1}+Vol_{ask,1}}
$$

多档可做距离加权 $\sum_k w_k(Vol_{bid,k}-Vol_{ask,k})$。

    - 映射：$\alpha_t \propto OBI\cdot\frac{spread}{2}$ 或 $\eta\propto OBI$，OBI>0 时 压紧 bid、放松 ask。
    - 坑：刷单/诱单、层间流转；需与**取消事件**联合看（见 OFI）。

OFI（Order Flow Imbalance）：事件级净订单流（Cont & de Larrard）

$$
OFI_\Delta = \sum_{\text{events}\in\Delta}\big( \mathbf{1}{add,bid} - \mathbf{1}{add,ask} - \mathbf{1}{rmv,bid} + \mathbf{1}{rmv,ask}\big)\cdot size
$$

也可细分市价单/撤单/挂单对 mid 的边际冲击。

    - 映射：短窗累加的 OFI → $\alpha$（tick）；与 OBI 联合更强。
    - 坑：馈送丢包/乱序要补齐；跨档移位需对齐到价格网格。

- 短期动量：如 $mom_{0.5s,2s}=\frac{mid_t-mid_{t-\tau}}{tick}$ 的 zscore。
    - 映射：作为 $\alpha$ 的一部分或作为 $\eta$ 的放大因子（趋势强时更不对称）。
    - 坑：与反转类市场冲突；建议和**波动状态做门控**。
- 交易强度（intensity）/Hawkes：$\lambda^{buy},\lambda^{sell}$ 估计；强买入簇 → 正 alpha。
    - 实现：核密度或简化 Hawkes（低配即可）；
    - 坑：极端行情下强度爆表，要做上限与冷却。
- VPIN（体积同步的信息交易概率）：分桶看买卖不平衡。
    - 用法：作为 AS_penalty 的增益项：VPIN 高 → 适当加宽两侧基础价差，保护自己。
    - 坑：在高频里噪声较大，别把它当方向信号。

实盘要点：

- 这类信号延迟极低，最适合做 $\eta$ 非对称 与 $r_t$ 的微调。
- “大家都会用”的拥挤风险：做去噪（时间/价距多尺度平滑）、和 OFI×OBI、强度×动量 等联乘；
- 做好 撤单节流 和 队列位置建模（后面评估指标里会提）。

### 4. Basis / 价差（期现、跨所、CEX–DEX、USDT-M vs Coin-M、主网-子网）

核心思想：跟随-回归与传导-延迟。流动性好的市场先动，弱者后跟；或在网络/撮合延迟下发生“延迟套利”的可观察价差。

- 期现：Basis = Fut - Spot - FundingAdjust（对永续要扣资金费率预期）；
- 跨所同币：$Spread = P^{ex1} - P^{ex2}$（考虑手续费/提现不可行时做合成对冲即可）；
- USDT-M vs Coin-M：同标的不同保证金计价带来的资金/风险偏好差；
- 主网 vs 子网：主网 ticker 与高相关子网/桥接衍生的 ticker 之间的 lead-lag。
- 统计：
    - 计算 半衰期（ECM/ADF）与 误差修正项：$\alpha_t \propto -\beta\cdot z_t$，$z_t$ 为标准化价差；
    - lead-lag：做滞后互相关或 HY 协方差，确定“谁领谁跟”与相位；
    - 风险：滑点、费率、撮合与资金费率、可借券/可开仓约束。
- 映射：把回归信号作为 $\alpha_t（tick）$，或仅用于 一侧更积极（如预期 ex1→ex2 传导，则在 ex2 更激进）。
- 优点：结构性强回归，常年有效；
- 缺点：需要多市场接入、延迟管理与一致定价；极端时会“断裂”，要止损与断路。

### 5. Microprice（加权中间价）

$$
Micro_t = \frac{A_1\cdot Vol_{bid,1} + B_1\cdot Vol_{ask,1}}{Vol_{bid,1}+Vol_{ask,1}}
= mid_t + \frac{Vol_{bid,1}-Vol_{ask,1}}{Vol_{bid,1}+Vol_{ask,1}}\cdot\frac{spread}{2}
$$

- 用法：$\alpha_t \approx Micro_t - mid_t$（tick）。对 大 tick、浅簿 的品种尤其有用，直接当作 $MicroBias_t$ 项进入 $r_t$。
- 优点：极快、极稳健；
- 缺点：容易被刷量扰动，建议与 OFI 联合验证。

## 评估与风控：别让“纸面 alpha”在实盘蒸发

- 每笔信息增益/逆向选择：

$$
RealizedAlpha = \text{side}\cdot (mid_{t+\Delta}-fill\_px)
$$

分解为：点差捕获 – 手续费 – 逆向选择损失。
- 成交率/撤单率/队列位置：
- 建 fill 曲线：入队深度、盘口人数、持有时间对成交概率的影响（logit/树）；
- 你的 $\eta$ 不能只看方向，还要看 fill 代价（太紧→成交多但常被打脸）。
- 库存与风险预算：$\phi$ 动态调节，超阈后自动只挂一侧或减仓模式。
- 延迟预算：[feature\ extraction] + [inference] + [wire] < 你的撤单-更新节拍（比如 5–20ms）。
- 回放回测：必须是 L2 事件级回放 + 排队/撤单模型（queue-reactive），避免“神级成交”。

## 组合与权重（含“随机控制/查表”）

把多路 alpha 做线上加权，两层结构最好用：

1.	信号融合（输出 $\alpha_t$ 与置信度 $w_t$）

$$
\alpha_t = \sum_i w_{i,t}\,\alpha_{i,t}, \quad
w_{i,t} \propto \frac{IR_{i,\text{recent}}}{\hat{\sigma}_{i,t}} \ \text{或做}\  \text{Ridge/Kalman/动态贝叶斯}
$$

- 设 门控：波动/点差/成交强度等状态变量不满足时，某些子 alpha 直接关断或降权。
- 方向类（2）可不直接入 $\alpha$，而去调 $\eta$（不对称度）。

2.	参数查表 + 随机控制（North-America 风格）

- 定义 状态向量 $z_t=\big(spread,\,vol,\,OBI,\,OFI,\,queue\_pos,\,fee\_tier,\ldots\big)$。
- 维护一个 参数表 $\mathcal{T}(z)\Rightarrow (k_1,\eta,\phi,\gamma,s_0)$。
- 在线用 上下文 bandit / Thompson Sampling 做轻微随机化探索（$\epsilon-greedy$ 也可），根据真实成交与 PnL 反馈微调表项。
- 好处：鲁棒、工程简单、超低延迟；坏处：需要持续 AB，且对状态离散划分要用心。

## Crypto CEX示例

- 目标持有期 H=5–10s；
- 波动刻度：$\sigma_H$ = 过去 60s 的中位绝对变动在 H 上的投影（tick）；
- 预测映射：$\alpha_t=\text{clip}\big(\text{IsoReg}(f(X_t)),\,3\sigma_H\big)$；
- 价差骨架：$s_0=1$ tick，$\gamma=0.3$，$AS_{penalty}= \mathbf{1}{VPIN>p95}\times 0.5$ tick；
- 非对称：$\eta = 0.6$（随 OBI/OFI 放大到 0.6–1.0）；
- 预测注入：$k_1 = 0.8$（强信号），弱信号时逐步降到 0.3；
- 库存：$\phi = c \times \frac{\sigma_{1m}}{Notional\_Cap}$（经验先取让满仓位移 ~0.5–1 tick）；
- 撤单节拍：10ms；挂单存活：1–2s；队列>前 20%或spread 收窄立即替换；
- Kill-switch：瞬时波动>p99、成交深度骤降>p99、异常滑点立即中性化 + 缩表。

常见坑位清单

- 信号慢半拍：模型再准，慢 2–5ms 也会被已发生的跳价碾碎 → 上线前必须做端到端延迟测量与控件内联（intrinsics）优化。
- 回测偷看：L2 重放若不模拟排队/可撤单延迟/撮合顺序，业绩会虚高一截。
- 拥挤与同质化：OBI/OFI 大家都会；做多尺度平滑 + 门控 + 与强度/动量/VPIN 联动。
- 库存“慢性中毒”：单边行情里 $\eta$ 过强但 $\phi$ 太小，会越贴越满仓。
- 费用与返佣：同一参数在不同费率/返佣层级下最优点差不同，参数表要分层。
- 跨所价差“断裂”：网络/风控/限仓导致回归失效 → 设 价差容忍带 + 断路器。