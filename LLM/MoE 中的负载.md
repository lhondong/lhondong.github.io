# MoE 中的负载

假设一个 batch（或一个路由 group）中有 $T$ 个 token、$N$ 个 routed experts，每个 token 选择 $k$ 个 Expert。

Router 先产生 logits：

$$
 z_{t,i}=h_t^\top e_i 
$$

再通过 softmax 或 sigmoid 得到路由分数：

$$
 p_{t,i}=G(z_{t,i}) 
$$

Top-k 产生离散选择：

$$
 a_{t,i}= \begin{cases} 1,&i\in\operatorname{TopK}(p_t)\\ 0,&\text{otherwise} \end{cases} 
$$

于是，Expert $i$ 的实际 token 负载为：

$$
 L_i=\sum_{t=1}^{T}a_{t,i} 
$$

理想平均负载是：

$$
 \bar L=\frac{Tk}{N} 
$$

如果没有均衡机制，训练早期一个很小的随机优势可能被不断放大：

>  某个 Expert 多接到一点 token
>  $\rightarrow$ 它训练得更充分
>  $\rightarrow$ Router 更愿意选它
>  $\rightarrow$ 它收到更多 token。

最终，少数 Expert 过载，其他 Expert 长期“饿死”，形成 routing collapse。

负载不均衡会导致一些不好的后果：

- **计算瓶颈**：同步执行时，整层速度由最忙的设备决定；
- **Token 溢出**：Expert 容量有限，超出的 token 需要丢弃或重路由；
- **专家退化**：冷门 Expert 得不到足够训练，模型的参数容量名义上很大，实际只用了少数部分。

接下来我们来看看一些 MoE 中的负载均衡策略。

## 1. Importance Loss

2017 年的 Sparsely-Gated MoE 先定义了 Expert 的 **Importance**：

$$
 \operatorname{Importance}_i(X) = \sum_{x\in X}G(x)_i 
$$

也就是把一个 batch 内分给 Expert $i$ 的门控权重全部加起来。

来举个例子。假设有 3 个 Expert、4 个 token。每个 token 的门控权重分别为：

$$
G(x_1) = [0.8, 0.2, 0]
$$

$$
G(x_2) = [0.7, 0, 0.3]
$$

$$
G(x_3) = [0, 0.6, 0.4]
$$

$$
G(x_4) = [0.9, 0.1, 0]
$$

其中，$G(x_t)_i$ 表示 token $x_t$ 分配给 Expert $i$ 的门控权重。

将一个 batch 内分配给同一 Expert 的门控权重相加，可以得到各个 Expert 的 Importance：

$$
\operatorname{Importance}_1 = 0.8 + 0.7 + 0 + 0.9 = 2.4
$$

$$
\operatorname{Importance}_2 = 0.2 + 0 + 0.6 + 0.1 = 0.9
$$

$$
\operatorname{Importance}_3 = 0 + 0.3 + 0.4 + 0 = 0.7
$$

因此，完整的 Importance 向量为：

$$
\operatorname{Importance}(X) = [2.4, 0.9, 0.7]
$$

从这个向量可以看出，Expert 1 获得的总门控权重明显高于另外两个 Expert，说明 Router 对 Expert 1 存在明显偏好。

然后使用变异系数的平方惩罚不均衡：

$$
 \mathcal L_{\mathrm{importance}} = w_{\mathrm{importance}}\cdot \operatorname{CV}(\operatorname{Importance})^2 
$$

其中：

$$
 \operatorname{CV}(v)=\frac{\operatorname{std}(v)}{\operatorname{mean}(v)} 
$$

CV 的好处是与绝对尺度无关。所有 Expert 的 Importance 相同时，CV 为 0；分布越偏，CV 越大。

它想表达的意思是：

>  不要让 Router 把几乎全部“信任”都押在少数 Expert 上。

![](https://pic2.zhimg.com/v2-dd3b15ab798048ba347115e07cd15cf7_r.jpg)

但 Importance 只衡量了**权重总和**，并不能衡量 token 数。

例如：

- Expert A 接收 2 个 token，每个权重都是 0.9，Importance 为 1.8；
- Expert B 接收 9 个 token，每个权重都是 0.2，Importance 也是 1.8。

二者 Importance 相同，实际计算负载却完全不同。

因此，只做 Importance Loss 还不够。

## 2. Expected Load Loss

原始 Sparsely-Gated MoE 使用 Noisy Top-k Gate。由于 Top-k 是离散操作，直接对“实际被选中的次数”求梯度并不容易。

它的做法是利用门控噪声估计 Expert $i$ 进入 Top-k 的概率：

$$
 \Pr(i\in\operatorname{TopK}) 
$$

再定义可微的期望负载：

$$
 \operatorname{Load}_i(X) = \sum_{x\in X}\Pr(i\in\operatorname{TopK}(x)) 
$$

对应的负载损失为：

$$
 \mathcal L_{\mathrm{load}} = w_{\mathrm{load}}\cdot \operatorname{CV}(\operatorname{Load})^2 
$$

举个例子。假设有 3 个 Expert，一个 batch 中有 4 个 token，每个 token 只选择 1 个 Expert，即 Top-1 路由。

考虑 Noisy Top-1 Gate 中的随机噪声后，假设每个 Expert 被选中的概率如下：

| Token | Expert 1 | Expert 2 | Expert 3 |
| ----- | -------- | -------- | -------- |
| x_1   | 0.80     | 0.15     | 0.05     |
| x_2   | 0.70     | 0.20     | 0.10     |
| x_3   | 0.10     | 0.60     | 0.30     |
| x_4   | 0.75     | 0.15     | 0.10     |

例如：

$$
\Pr\left(E_1 \in \operatorname{Top1}(x_1)\right)=0.8
$$

它表示，对于 token $x_1$，如果不断重新采样门控噪声，大约有 80% 的情况下 Expert 1 会成为得分最高的 Expert。

注意与之前提到的权重区分，这里是指 Expert 1 **进入 Top-1 的概率**。

Expert 1-3 的期望负载为：

$$
\operatorname{Load}_1=0.80+0.70+0.10+0.75=2.35
$$

$$
\operatorname{Load}_2=0.15+0.20+0.60+0.15=1.10
$$

$$
\operatorname{Load}_3=0.05+0.10+0.30+0.10=0.55
$$

因此，完整的期望负载向量为：

$$
\operatorname{Load}(X)=[2.35,\,1.10,\,0.55]
$$

这一次均衡的是：

>  按照当前含噪路由分布，预计每个 Expert 会收到多少 token。

![](https://picx.zhimg.com/v2-2d34611e9019060d4b111342bbd07169_r.jpg)

我们来和上一种方法对比一下：

- Importance 在意的是“权重有多大”；
- Expected Load 在意的是“预计会来多少 token”。

在早期 Noisy Top-k MoE 中，两者经常一起使用。现代大模型 Router 更常使用接下来要介绍的 $f\cdot P$ 形式，因为实现更简单，不需要显式推导含噪 Top-k 的选择概率。

## 3. $f\cdot P$ Auxiliary Loss

Switch Transformer 和 GShard 一类实现采用了更紧凑的辅助损失。

先定义 Expert $i$ 的实际路由比例：

$$
 f_i = \frac{1}{Tk} \sum_{t=1}^{T}a_{t,i} 
$$

再定义 Router 给它的平均软概率：

$$
 P_i = \frac{1}{T} \sum_{t=1}^{T}p_{t,i} 
$$

均衡损失为：

$$
 \mathcal L_{\mathrm{bal}} = \alpha N\sum_{i=1}^{N}f_iP_i 
$$

不同论文会把 $N$、$k$ 吸收到 $f_i$ 或 $\alpha$ 中，公式外观看起来略有差别，但核心都是 $f_iP_i$。

举个例子。假设有 3 个 Expert，即 $N=3$；一个 batch 中有 4 个 token，即 $T=4$；每个 token 只选择 1 个 Expert，即 $k=1$。

假设实际路由结果是：

```text
token 1 -> Expert 1
token 2 -> Expert 1
token 3 -> Expert 2
token 4 -> Expert 1
```

那么：

- Expert 1 收到 3 个 token；
- Expert 2 收到 1 个 token；
- Expert 3 收到 0 个 token。

因为 $T=4$、$k=1$，所以总共有 $Tk=4$ 次路由分配。

因此：

$$
f_1=\frac{3}{4}=0.75
$$

$$
f_2=\frac{1}{4}=0.25
$$

$$
f_3=\frac{0}{4}=0
$$

于是：

$$
f=[0.75,\,0.25,\,0]
$$

这里的 $f_i$ 表示 Expert $i$ 实际承担了多少比例的 token 路由。并且：

$$
\sum_{i=1}^{N}f_i=1
$$

对于一般的 Top-$k$ 路由，每个 token 会产生 $k$ 次 token-to-expert 分配，因此总分配次数是 $Tk$。

假设 4 个 token 的 Router 概率如下：

| Token | Expert 1 | Expert 2 | Expert 3 |
| ----- | -------- | -------- | -------- |
| t_1   | 0.70     | 0.20     | 0.10     |
| t_2   | 0.60     | 0.30     | 0.10     |
| t_3   | 0.40     | 0.50     | 0.10     |
| t_4   | 0.70     | 0.20     | 0.10     |

那么 Expert 1-3 的平均软概率为：

$$
P_1=\frac{0.70+0.60+0.40+0.70}{4}=0.60
$$

$$
P_2=\frac{0.20+0.30+0.50+0.20}{4}=0.30
$$

$$
P_3=\frac{0.10+0.10+0.10+0.10}{4}=0.10
$$

因此：

$$
P=[0.60,\,0.30,\,0.10]
$$

这里的 $P_i$ 表示从整个 batch 来看，Router 平均把多少软概率分给了 Expert $i$。

![](https://pic3.zhimg.com/v2-46ec978930951030831baf2f9684b58a_r.jpg)

为什么要同时保留硬计数 $f$ 和软概率 $P$？

- $f_i$ 反映真实 dispatch 结果，但 Top-k/argmax 本身不可微；
- $P_i$ 是连续值，可以给 Router 提供梯度；
- 如果某个 Expert 的 $f_i$ 很大，损失会推动 Router 降低它的 $P_i$；
- 理想状态下，两者都接近 $1/N$。

这种形式非常实用，但它有一个超参数需要权衡：

- $\alpha$ 太小，Router 可能继续偏科；
- $\alpha$ 太大，均衡梯度会干扰语言模型主目标，让“本来应该去某个 Expert 的 token”被迫改道。

## 4. Multi-level Auxiliary Loss

只让每个 Expert 的 token 数接近，并不一定能让系统跑得快。

假设 8 个 Expert 部署在 4 张 GPU 上：

$$
 \mathcal E_1=\{E_1,E_2\},\quad \mathcal E_2=\{E_3,E_4\},\quad \mathcal E_3=\{E_5,E_6\},\quad \mathcal E_4=\{E_7,E_8\} 
$$

这是就会涉及到多种粒度的均衡。

**(1) Expert-Level Balance**

目标是防止某些 Expert 过热、某些 Expert 饿死：

$$
 \mathcal L_{\mathrm{exp}} = \alpha_1\sum_i f_iP_i 
$$

这是模型层面的均衡。

**(2) Device-Level Balance**

把同一设备上所有 Expert 的负载聚合：

$$
 f'_d = \sum_{i\in\mathcal E_d}f_i 
$$

再约束每张设备的总计算量接近：

$$
 \mathcal L_{\mathrm{dev}} = \alpha_2\sum_d f'_dP'_d 
$$

这是计算系统层面的均衡。

**(3) Communication-Level Balance**

Expert Parallel 通常需要 All-to-All：

1. 每张 GPU 把 token 发给目标 Expert 所在设备；
2. Expert 计算完成；
3. 再把输出发回原设备。

即使发送量有上限，某张设备仍可能收到远多于其他设备的 token，形成网络热点。

于是还可以按“每张设备接收的 token/字节数”定义通信负载：

$$
 \mathcal L_{\mathrm{comm}} = \alpha_3\sum_d f''_dP''_d 
$$

![](https://pic3.zhimg.com/v2-901a5f2ae4c596f16d0fba1f517e504c_r.jpg)

DeepSeek-V2 就显式讨论了 Expert-Level、Device-Level 和 Communication Balance Loss。是否需要全部启用，取决于 Expert 的放置方式、并行规模和网络拓扑。

## 5. Expert Capacity + Token Dropping

前面几种辅助损失只能“鼓励”均衡，不能保证这一批 token 一定均匀。

工程系统通常还会给每个 Expert 设置容量：

$$
 C = \left\lceil \mathrm{CF}\cdot\frac{Tk}{N} \right\rceil 
$$

其中 CF 是 Capacity Factor。

例如：

$$
 T=12,\quad N=4,\quad k=1,\quad \mathrm{CF}=1.0 
$$

则：

$$
 C=3 
$$

如果 Router 的实际请求为：

$$
 [6,3,2,1] 
$$

那么 E1 收到了 6 个请求，但只能处理 3 个。常见策略是保留路由分数最高的 Top-C token，其余 token 可以有如下几种选择：

- 直接跳过 Expert，走 residual 分支；
- 丢弃该次 Expert 计算；
- 尝试重路由到次优 Expert；
- 或在支持动态 shape 的系统里扩容，但牺牲吞吐可预测性。

![](https://pic2.zhimg.com/v2-e6f2b556d1f594c724277091f797b8fb_r.jpg)

Capacity Factor 基于容量一定 buffer：

- CF 小：预留的计算和显存少，但 token drop 增多；
- CF 大：drop 减少，但 padding、显存与最坏计算量增加。

但要注意，这只是工程上的方法，它能保护吞吐，却不会让 Router 自己学会均衡。如果没有其他均衡机制，热门 Expert 仍然会不断溢出，冷门 Expert 仍然闲置。

## 6. Expert Choice

传统 Token Choice 是每个 token 选择 Top-k Expert。

它保证了“每个 token 激活 $k$ 个 Expert”，却没有保证每个 Expert 收到多少 token。

Expert Choice 则反过来考虑：

>  每个 Expert 从候选 token 中选择分数最高的 Top-C token。

这样，每个 Expert 都有一个固定 bucket，Expert 负载由构造直接保证。

![](https://pic4.zhimg.com/v2-d1f01eaf1d4c1c3e6f1fcdce35db9a75_r.jpg)

它有两个明显变化。

第一，每个 Expert 的计算量固定，但每个 token 的计算量不再固定。有些 token 可能被多个 Expert 选中；有些 token 可能只被一个 Expert 选中；甚至可能没有 Expert 选择某个 token。

第二，选择过程需要看到一组 token 后再排名。

对双向编码、部分视觉任务或允许整组全局分配的场景，这可能没问题；但对因果语言模型，如果前面 token 的 Expert 选择取决于同一序列中的未来 token，就可能造成 future-token leakage。

## 7. Balanced Assignment

Expert Choice 是每个 Expert 独立选 token。还可以更进一步，把所有 token 到 Expert 的路由统一写成一个分配问题。

设 $A_{t,i}\in\{0,1\}$ 表示是否把 token $t$ 分给 Expert $i$，$S_{t,i}$ 是 affinity score。

目标是最大化总 affinity：

$$
 \max_A \sum_{t,i}A_{t,i}S_{t,i} 
$$

同时满足：

$$
 \sum_iA_{t,i}=1 
$$

$$
 \sum_tA_{t,i}=C 
$$

也就是每个 token 分配给一个 Expert，每个 Expert 恰好收到 $C$ 个 token。

![](https://pica.zhimg.com/v2-33df5153e12466bb04525a717408cc68_r.jpg)

BASE Layers 将 token-to-expert routing 表述为线性分配问题，在规则上得到均衡计算负载，不再依赖额外辅助损失。

如果严格离散求解成本太高，可以使用 Sinkhorn 一类行列归一化，把 affinity 矩阵变成近似满足行、列约束的软分配，再离散化。

这一类方法的优点是均衡目标清晰、约束严格；缺点是：

- 需要跨一组 token 做全局协调；
- 分配求解本身有额外计算和通信；
- 分布式实现复杂；
- 在因果语言模型中，同样要避免未来 token 通过全局分配影响过去 token。

如果 $T$ 不能被 $N$ 整除，也不必强行让每列完全相同，可以让各 Expert 容量取 $\lfloor T/N\rfloor$ 或 $\lceil T/N\rceil$，使最大差值不超过 1。

## 8. Auxiliary-Loss-Free

传统辅助损失把“均衡”写进优化目标：

$$
 \mathcal L = \mathcal L_{\mathrm{LM}} + \alpha\mathcal L_{\mathrm{balance}} 
$$

问题在于，均衡梯度可能和语言建模梯度冲突。

Auxiliary-Loss-Free 的思路是：**不再通过反向传播教 Router 均衡，取而代之在 Top-k 之前加一个按 Expert 维护的控制 bias。**

Router 产生原始分数：

$$
 s_{t,i} 
$$

然后真正决定 Top-k 的是：

$$
 \operatorname{TopK}(s_{t,i}+b_i) 
$$

但 Expert 输出的组合权重仍然使用原始分数 $s_{t,i}$，而不是 $s_{t,i}+b_i$。

每个训练 step 结束后统计上一批负载 $c_i$，再更新：

$$
 b_i \leftarrow b_i + u\cdot \operatorname{sign}(\bar c-c_i) 
$$

因此：

- 过载 Expert：$c_i>\bar c$，bias 降低；
- 欠载 Expert：$c_i<\bar c$，bias 提高；
- 负载接近平均：bias 不变。

![](https://picx.zhimg.com/v2-74ccbf7e41b62bab3382edaff97587e1_r.jpg)

可以把它理解成在线反馈控制器：

- 上一批谁太忙，下一批就降低谁的“被选优先级”；

- 上一批谁太闲，下一批就提高谁的“被选优先级”。

由于 $b_i$ 不由语言模型 loss 的反向传播更新，均衡过程不产生额外的干扰梯度。

需要注意：

1. bias 只影响 Top-k 选择，不应直接加进最终 Expert 输出权重；
2. 更新要依据历史负载，而不是用当前序列的未来 token 反过来修改当前路由，否则会破坏因果性。

DeepSeek-V3 主要使用这种 loss-free 策略，但仍保留了很小的 sequence-wise balance loss，用来防止单条序列内部出现极端偏斜。

## 总结

| 类别     | 方法                                                  | 能否严格保证 Expert 负载 | 主要代价                              |
| -------- | ----------------------------------------------------- | ------------------------ | ------------------------------------- |
| 软正则   | Importance<br>Expected Load<br>f·P<br>Multi-level Aux | 否                       | 要调 loss 系数，会引入额外梯度        |
| 硬兜底   | Capacity + Token Dropping                             | 只保证上限               | token 丢弃、padding 或重路由          |
| 改变路由 | Expert Choice                                         | 是，固定 Expert bucket   | 每个 token 计算量不固定，需处理因果性 |
| 改变路由 | Balanced Assignment                                   | 是或近似严格             | 全局求解、额外协调与通信              |
| 在线控制 | Loss-Free Bias                                        | 通常为近似均衡           | 要调 bias 更新速率，存在反馈滞后      |
