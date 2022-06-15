
# Transformer 中的 attention 为什么需要 scaled?

> 向量的点积结果会很大，将 softmax 函数 push 到梯度很小的区域，scaled 会缓解这种现象。

## 1. 为什么比较大的输入会使得 softmax 的#梯度变得很小？

对于一个输入向量 $\mathbf{x}\in \mathbb{R}^d$，Softmax 函数将其映射/归一化到一个分布 $\hat{\mathbf{y}}\in\mathbb{R}^d$。在这个过程中，Softmax 先用一个自然底数 $e$ 将输入中的元素间差距先“拉大”，然后归一化为一个分布。假设某个输入 $\mathbf{x}$ 中最大的的元素下标是 $k$，如果输入的数量级变大（每个元素都很大），那么 $\hat{y}_k$ 会非常接近 1。

可以用一个小例子来看看 $\mathbf{x}$ 的数量级对输入最大元素对应的预测概率 $\hat{y}_k$ 的影响。假定输入 $\mathbf{x}=[a,a,2a]^T$），我们来看不同量级的 $a$ 产生的 $\hat{y}_3$ 有什么区别。

- a=1 时，$\hat{y}_3= 0.5761168847658291$;
- a=10 时，$\hat{y}_3=0.999909208384341$;
- a=100 时，$\hat{y}_3\approx 1.0$  （计算机精度限制）。

可以看到，数量级对 Softmax 得到的分布影响非常大。**在数量级较大时，Softmax 将几乎全部的概率分布都分配给了最大值对应的标签。**

然后我们来看 Softmax 的梯度。不妨简记 Softmax 函数为 $g$，Softmax 得到的分布向量 $\hat{\mathbf{y}}=g(\mathbf{x})$ 对输入 $\mathbf{x}$ 的梯度为：

$$
\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}}=\operatorname{diag}(\hat{\mathbf{y}})-\hat{\mathbf{y}} \hat{\mathbf{y}}^{\top} \quad \in \mathbb{R}^{d \times d}
$$
把这个矩阵展开：
$$
\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}}=\left[\begin{array}{cccc}
\hat{y}_{1} & 0 & \cdots & 0 \\
0 & \hat{y}_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \hat{y}_{d}
\end{array}\right]-\left[\begin{array}{cccc}
\hat{y}_{1}^{2} & \hat{y}_{1} \hat{y}_{2} & \cdots & \hat{y}_{1} \hat{y}_{d} \\
\hat{y}_{2} \hat{y}_{1} & \hat{y}_{2}^{2} & \cdots & \hat{y}_{2} \hat{y}_{d} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{y}_{d} \hat{y}_{1} & \hat{y}_{d} \hat{y}_{2} & \cdots & \hat{y}_{d}^{2}
\end{array}\right]
$$

根据前面的讨论，当输入 $\mathbf{x}$ 的元素均较大时，softmax 会把大部分概率分布分配给最大的元素，假设我们的输入数量级很大，最大的元素是 $x_{1}$, 那么就将产生一个接近 one-hot 的向量 $\hat{\mathbf{y}} \approx[1,0, \cdots, 0]^{\top}$, 此时上面的矩阵变为如下形式：

$$
\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}} \approx\left[\begin{array}{cccc}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{array}\right]-\left[\begin{array}{cccc}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{array}\right]=\mathbf{0}
$$
也就是说，在输入的数量级很大时，梯度消失为 0, 造成参数更新困难。

## 2. 维度与点积大小的关系是怎么样的，为什么使用维度的根号来放缩？

针对为什么维度会影响点积的大小，在论文的脚注中其实给出了一点解释：

> To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean O and variance 1. Then their dot product, $q\cdot k= \sum_{i=1}^{d_k}q_ik_i$, has mean 0 and variance $d_k$.

假设向量 $q$ 和 $k$ 的各个分量是互相独立的随机变量，均值是 0, 方差是 1, 那么点积 $q \cdot k$ 的均值是 0, 方差是 $d_{k}$ 。这里给出一点更详细的推导：

对 $\forall i=1, \cdots, d_{k}, q_{i}$ 和 $k_{i}$ 都是随机变量，为了方便书写，不妨记 $X=q_{i}, Y=k_{i}$ 。 

这样有：$\quad D(X)=D(Y)=1, \quad E(X)=E(Y)=0$ 。

则：

$$
E(X Y)=E(X) E(Y)=0 \times 0=0
$$

$$
\begin{aligned}
D(X Y) &=E\left(X^{2} \cdot Y^{2}\right)-[E(X Y)]^{2} \\
&=E\left(X^{2}\right) E\left(Y^{2}\right)-[E(X) E(Y)]^{2} \\
&=E\left(X^{2}-0^{2}\right) E\left(Y^{2}-0^{2}\right)-[E(X) E(Y)]^{2} \\
&=E\left(X^{2}-[E(X)]^{2}\right) E\left(Y^{2}-[E(Y)]^{2}\right)-[E(X) E(Y)]^{2} \\
&=D(X) D(Y)-[E(X) E(Y)]^{2} \\
&=1 \times 1-(0 \times 0)^{2} \\
&=1
\end{aligned}
$$

这样 $\forall i=1, \cdots, d_{k}; q_{i} \cdot k_{i}$ 的均值是 0, 方差是 1, 又由期望和方差的性质，对相互独立的分量 $Z_{i}$, 有

$$
E\left(\sum_{i} Z_{i}\right)=\sum_{i} E\left(Z_{i}\right)
$$

以及

$$
D\left(\sum_{i} Z_{i}\right)=\sum_{i} D\left(Z_{i}\right)
$$

所以有 $q \cdot k$ 的均值 $E(q \cdot k)=0$, 方差 $D(q \cdot k)=d_{k}$ 。方差越大也就说明，点积的数量级越大（以越大的概率取大值）。那么一个自然的做法就是把方差稳定到 1，做法是将点积除以 $\sqrt{d}_{k}$ , 这样有：
$$
D\left(\frac{q \cdot k}{\sqrt{d}_{k}}\right)=\frac{d_{k}}{\left(\sqrt{d}_{k}\right)^{2}}=1
$$

将方差控制为 1 , 也就有效地控制了前面提到的梯度消失的问题。