## Linear Attention

Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

目标是提升长序列下 Transformer 的计算速度。

Softmax 形式的 Attention

$$
\begin{aligned}
Q & =x W_Q \\
K & =x W_K \\
V & =x W_V \\
A_l(x)=V^{\prime} & =\operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right) V
\end{aligned}
$$
标准 Attention 瓶颈，时间复杂度：

- 计算 $QK^\top$：$O(N^2 d)$
- 乘上 $V$：$O(N^2 d)$

**内存复杂度**：要显式或隐式存 $N\times N$ 的 attention matrix（或相应的中间结果），也是 $O(N^2)$。

标准 Attention 的权重：

$$
\alpha_{ij} \propto \exp(q_i^\top k_j)
$$

从核方法角度，可以把 $\exp(q_i^\top k_j)$ 看成某个核函数的内积：

$$
\exp(q_i^\top k_j) = \phi(q_i)^\top \phi(k_j)
$$

其中 $\phi(\cdot)$ 把向量映射到一个（可能更高维的）特征空间。很多 Linear Attention 的核心就是 **找一个合适的 $\phi$**，把 softmax kernel 近似掉。

假设我们有一个正值特征映射 $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^r_+$，那么可以近似：

$$
\mathrm{softmax}(QK^\top)V \approx D^{-1}\Phi(Q)\big(\Phi(K)^\top V\big)
$$

其中 $\Phi(Q)$ 表示对每个 token 的 $q_i$ 做 $\phi$ 映射，从 $N \times d$ 矩阵映射为 $N\times r$ 的矩阵。

一般将 $\phi$ 设置为 elu(⋅) 函数，不设置为 Relu⁡(⋅) 是为了避免在 x 为负时梯度为 0。

$D^{-1}$ 是 Linear Attention 核函数近似过程中，用于**归一化**的关键矩阵（或向量），核心作用是模拟标准 Attention 中 softmax 的“概率归一化”效果，确保注意力权重之和为 1（或近似满足该性质）。

**$D$ 本质是“归一化因子矩阵”**，其元素 $D_{ii}$ 对应第 $i$ 个 query token 的“注意力权重总和”，即： $$D_{ii} = \phi(q_i)^\top \cdot \sum_{j=1}^N \phi(k_j)$$ 其中 $\sum_{j}\phi(k_j)$ 是所有 key 经 $\phi$ 映射后的累加向量，可预计算。 因此 $D$ 是对角矩阵（或简化为向量，因每个 $D_{ii}$ 仅与第 $i$ 个 query 相关），$D^{-1}$ 就是 $D$ 的逆矩阵（对角矩阵的逆即对角线元素取倒数）。 

作用：标准 Attention 中，softmax 会将每个 query 对所有 key 的相似度（$QK^\top$ 元素）转化为“和为 1 的概率权重”； Linear Attention 用 $\phi(q_i)^\top \phi(k_j)$ 近似相似度，但直接计算 $\Phi(Q)(\Phi(K)^\top V)$ 会缺少“归一化”。此时 $D^{-1}$ 就是用来弥补这一点： 对每个 query $q_i$，用 $D_{ii}^{-1}$（$D_{ii}$ 的倒数）乘以 $\phi(q_i)^\top \cdot (\Phi(K)^\top V)$，相当于将“未归一化的注意力输出”除以“所有 key 对该 query 的相似度总和”，从而让结果近似满足“权重归一化”，避免因不同 query 的相似度规模差异导致的输出偏差。

相比标准 Attention 中 $O(N^2)$ 的 softmax 计算，$D^{-1}$ 只需通过预计算的 $\sum_{j}\phi(k_j)$ 与 $\phi(q_i)$ 点积得到，复杂度仅 $O(Nr)$，是 Linear Attention 实现“线性复杂度（$O(Nrd)$）”的关键之一。

### 换计算顺序

原式（核近似后）：  

$$
O_i = \sum_{j=1}^N \mathrm{softmax}_j(q_i^\top k_j), v_j
\approx \frac{\sum_j \phi(q_i)^\top \phi(k_j), v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}
$$

将分子展开：  

$$
\sum_{j} \phi(q_i)^\top \phi(k_j), v_j
= \phi(q_i)^\top \left( \sum_j \phi(k_j), v_j^\top \right)
$$

于是可以先预计算一个 **“Key-Value 累加矩阵”**：  

$$
S = \sum_{j=1}^N \phi(k_j), v_j^\top \in \mathbb{R}^{r \times d}
$$

再对每个 token：  

$$
O_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

如果把 $\sum_j \phi(k_j)$ 也预先累加，整个计算只需要：

- **一步**对整个序列的累加：
	- $S = \sum_j \phi(k_j)v_j^\top$ （复杂度 $O(N r d)$）
	- $z = \sum_j \phi(k_j)$ （复杂度 $O(N r)$）
- 对每个 token：
	- $\phi(q_i)^\top S$ （复杂度 $O(r d)$）
	- $\phi(q_i)^\top z$ （复杂度 $O(r)$）

总复杂度从 $O(N^2 d)$ 变成 $O(N r d)$，通。取 $r \ll N$，从而是**线性于序列长度 (N)**。

代表工作包括 _Transformers are RNNs_、Performer、TransNormer / TransNormerLLM、LaplacianFormer 等。

> 1. 多项式注意力：是一种通过对输入向量进行多项式变换来计算注意力权重的方法。这种机制一般会将输入向量经过一个多项式函数（如二次函数 $f(x)=ax^2+bx+c$ 或三次函数）来增强或抑制特定的特征，优点在于它能够捕捉输入数据中更复杂的非线性关系。注意：多项式函数的参数 a/b/c 会在训练中不断更新。
> 2. RBF 核注意力：一种基于径向基函数（Radial Basis Function）的注意力机制。RBF 核是一种常用的核函数，可以用于度量向量之间的相似性。RBF 核的一个典型形式是高斯核：$K(x,y) = exp⁡(−||x−y||^2 / \sigma^2)$，其中，$\sigma$ 是一个可调的参数，控制核函数的“宽度”。

### cumsum 瓶颈

因果（自回归）场景的问题

自回归时，第 $i$ 个位置只能看到 $\{1,…,i\}$：

$$
S_i = \sum_{j=1}^i \phi(k_j), v_j^\top
$$
$$
z_i = \sum_{j=1}^i \phi(k_j)
$$
$$
O_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

这看似还是 $O(N)$，但**这里的累加是前缀和（cumsum）**，在 GPU 上意味着强序依赖：

- 不能很容易让不同 $i$ 并行，特别在训练时，长序列 cumsum 会成为内存 & IO 瓶颈。
- 实际 wall-clock 很大一部分时间卡在这个 cumsum 上，导致“理论线性，实际不一定快”。

实际过程：

$$
\begin{aligned}
o_1 &= q_1(k_1^\top v_1) \\
o_2 &= q_2(k_1^\top v_1 + k_2^\top v_2) \\
o_3 &= q_3(k_1^\top v_1 + k_2^\top v_2 + k_3^\top v_3) \\
o_4 &= q_4(k1^\top v_1 + k_2^\top v_2 + k_3^\top v_3 + k_4^\top v_4)
\end{aligned}
$$

```python
for t in 1 to n:
	hidden_state += K[t].T * V[t］ # 仅累积历史信息 
	output[t] = Q[t] * hidden_state
```

这是 Lightning Attention 要解决的核心问题。

##  Lightning Attention

Lightning Attention 系列的主线工作：

- TransNormerLLM 引入 Lightning Attention 1，宣称线性 attention 在 LLM 上首次做到“精度不输 softmax + 显著更快”。([arXiv](https://arxiv.org/abs/2307.14995?utm_source=chatgpt.com "TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer"))
- Lightning Attention-2 进一步把线性 attention 做成“序列长度无关的恒定速度 + 近似无限上下文”，是现在很多长上下文 LLM 和 reasoning 模型（如 MiniMax-M1）的基础组件。([arXiv](https://arxiv.org/abs/2401.04658?utm_source=chatgpt.com "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"))

> _Various Lengths, Constant Speed_  
> ——在固定显存下，对不同序列长度保持**几乎恒定的训练速度**。([arXiv](https://arxiv.org/abs/2405.17381?utm_source=chatgpt.com "Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention"))

问题根源：**因果线性 attention 里的 cumsum 无法很好并行**，导致长序列时速度明显下降。Lightning 的目标就是：

> 用 **块化(block)** + **分解(intra / inter)** + **tiling & Triton kernel**，干掉 cumsum 的串行瓶颈。

### Intra-block vs Inter-block 分解

设序列长度 (N)，我们用 block size (B) 分块：

- 第 $b$ 个块：tokens $[bB, (b+1)B)$

Lightning Attention 把 attention 计算拆成两部分：([GitHub](https://raw.githubusercontent.com/mlresearch/v235/main/assets/qin24c/qin24c.pdf?utm_source=chatgpt.com "Efficient Language Modeling with Lightning Attention - GitHub"))

1. **块内 (intra-block)**

   - 一个 block 内长度只有 (B)，可以直接用**标准 softmax attention**（甚至配合 FlashAttention）算局部交互：  
      $$
      O_i^{\text{local}} = \sum_{j \in \text{same block}} \alpha_{ij}^{\text{local}} v_j
      $$
   - 复杂度 $O(B^2 d)$，但 B 通常远小于 N，且每个 block 之间可以**完全并行**。

2. **块间 (inter-block)**

   - 对远距 token 的影响，使用 **线性 attention 的核 trick** 做“块级别”的累加：
     - 每个 block 汇总出一个 “块级 KV 状态”：  
        $$
        S^{(b)} = \sum_{j \in \text{block } b} \phi(k_j) v_j^\top,\quad
       z^{(b)} = \sum_{j \in \text{block } b} \phi(k_j)
        $$
     - 然后对 block 顺序做累加（但这是在 block 级别而非 token 级别），复杂度 $O(\frac{N}{B})$，且可用更高效的扫描 / tiling 技巧。
   - 每个 token 再用自身 $\phi(q_i)$ 去乘以“过去所有 block 的汇总”，得到远距部分 $O_i^{\text{global}}$。

最终输出：  

$$
O_i = O_i^{\text{local}} + O_i^{\text{global}}
$$

**关键点**：

- 把 **“全序列级别的 cumsum”** 换成 **“块级别的 prefix / tiling 操作”**。
- 块内用普通 attention，把表达能力（局部细粒度）保住。
- 块间靠线性 attention 做超长距离交互。

### Tiling & IO-aware 实现

Lightning Attention-1 / 2 都非常强调“IO-aware”，思路和 FlashAttention 很像：

- 在 GPU 上，**HBM（显存）带宽远小于算力**，所以：
  - 尽量把 Q/K/V 分块后搬到 on-chip SRAM / shared memory。
  - 在 Tiling 上完成尽可能多的运算（$QK^\top$、softmax、与 V 相乘、线性 kernel 的乘积）。
  - 最后只把结果写回 HBM，减少大矩阵在显存和 SM 之间来回搬运。

Lightning Attention-2：

- 对任意长的输入，**训练/推理速度几乎不随长度变化**（只要显存允许 tiling）。
- 在同一显卡上，支持更长上下文，且比 FlashAttention-2 仍快一截。

### Lightning Attention-V1

TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer

Lightning Attention 的基本思想是 Linear Attention

- Lightning Attention-V1 解决 GPU 上的内存访问（I/O）可能影响 Attention 计算的整体速度；
	- 方法：取消 Softmax，采用 Swish 激活函数，结合 Flash Attention 的分块计算（SRAM 优化）。
	- 局限：复杂度仍为 $O(n^2)$，未解决因果模型根本问题。
- Lightning Attention-V2 解决因果注意力下，所需的累积求和（cumsum）操作使其无法达到理论训练速度的问题。

考虑到因果 mask 的情况下右乘的计算很低效，在训练过程中，继续沿用了传统的左乘版本。同时，为了加速 Attention 计算，受到 FlashAttention 的启发，引入了 Lightning Attention 算法，使 Linear Attention 具有更好的 IO 交互性。

![](https://pica.zhimg.com/v2-1149c1da7ae567108ae77a2f64ff5872_r.jpg)

从算法流程俩看，外层 Q 循环，内层 K、V 循环，一个计算示例如下。其中，所有的参数配置在图中右上角，且仅展示了第二个外层的计算示意图。

![](https://pic3.zhimg.com/v2-b4138fedab7d28f7fd1fa05f51510664_r.jpg)

然而，这并没有充分利用线性注意力的"右乘"优点。

### Lightning Attention-V2

V1 版本的不足在 V2 版本得到了解决，核心创新：分块处理序列，分为块内（intra-block）和块间（inter-block）注意力：

- 块内：传统左乘+掩码，保证单向性。
- 块间：缓存历史 KV 并右乘，实现 $O(nd^2)$ 复杂度。

前向传播：

- 分块加载 Q/K/V 到 SRAM。
- 计算块内注意力（带掩码）和块间注意力（利用缓存 KV）。
- 更新 KV 缓存并输出结果。

反向传播：

- 分块计算梯度时，分别处理块内和块间贡献。

硬件优化: SRAM tiling 技术

- 将数据从 HBM (高带宽内存) 加载至 SRAM (片上高速内存) 计算
- 块内：密集矩阵乘法，充分利用 GPU 张量核心
- 块间：维护滚动累积器 KV，大幅减少内存读写
- 最终输出在 SRAM 合并后写回 HBM

优化：

- 掩码处理：分块后 M 矩阵仍保持下三角结构，确保单向性。
- 硬件加速：通过分块减少内存访问，提升并行效率。

![](https://pica.zhimg.com/v2-936929b29e242d6d4e222ebd71b4bcf4_r.jpg)

块内（intra）块间（inter）之间计算

![](https://pic1.zhimg.com/v2-a9ca9b6a0c84127619c5a5a4b56c0798_r.jpg)

v2 通过块间与块内分离，**完全消除了全局 cumsum 操作** :

- 块内：无 cumsum，直接计算
- 块间：使用数学归纳法更新 KV 状态，并行高效

代码：[核心代码](https://github.com/OpenNLPLab/lightning-attention/blob/main/lightning_attn/ops/triton/lightning_attn2.py) 和 [示例](https://github.com/OpenNLPLab/lightning-attention/blob/main/tests/ops/test_lightning2.py)

## MiniMax

MiniMax-Text-01，包含 456B 参数，其中有 45.9B 的激活参数。架构的核心点：

- Attention 模块：集成了 Lightning Attention 和 Softmax Attention，以结构化的方式组合。具体来说：每 7 个 transnormer 块的 Lightning Attention 之后跟随一个带有 Softmax Attention 的 transformer 块，共 80 层；每个 Attention 模块由 64 个 head 组成，每个 head 的维度为 128。
- 共享 KV：采用 GQA，且分组大小为 8。
- 残差：采用 DeepNorm，即 $output = \alpha \times Layer(input) + input$ ，以保持训练过程中梯度的稳定性。
- 位置编码：采用 RoPE，并将其应用于一半的注意力 head 维度，基频设置为 10000。
- FFN 模块：采用 MOE，模型的 hidden size 被设置为 6144，每个 layer 包含 32 个专家，采用 top-2 路由策略。每个专家中 FFN 的 hidden size 为 9216。

![](https://pica.zhimg.com/v2-90a142bd56cbdf7cbe7a2334c69b53de_r.jpg)
