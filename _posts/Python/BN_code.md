# Batch Normalization

## 算法流程

- 输入：批处理 (mini-batch) 输入 $x: \mathcal{B}=\left\{x_{1, \ldots, m}\right\}$
- 输出：规范化后的网络响应 $\left\{y_{i}=\mathrm{BN}_{\gamma, \beta}\left(x_{i}\right)\right\}$

1. $\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i} / /$ 计算批处理数据均值
2. $\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2} / /$ 计算批处理数据方差
3. $\hat{x_{i}} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} / /$ 规范化
4. $y_{i} \leftarrow \gamma \hat{x_{i}}+\beta=\mathrm{BN}_{\gamma, \beta}\left(x_{i}\right) / /$ 尺度变换和偏移
5. return 学习的参数 $\gamma$ 和 $\beta$.

## 前向传播代码

```python
import numpy as np
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #为了后向传播求导方便，这里都是分步进行的
  #step1: 计算均值
  mu = 1./N * np.sum(x, axis = 0)

  #step2: 减均值
  xmu = x - mu

  #step3: 计算方差
  sq = xmu ** 2
  var = 1./N * np.sum(sq, axis = 0)

  #step4: 计算 x^的分母项
  sqrtvar = np.sqrt(var + eps)
  ivar = 1./sqrtvar

  #step5: normalization->x^
  xhat = xmu * ivar

  #step6: scale and shift
  gammax = gamma * xhat
  out = gammax + beta

  #存储中间变量
  cache =  (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache
```

反向传播代码

```python
def batchnorm_backward(dout, cache):

  #解压中间变量
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  N,D = dout.shape

  #step6
  dbeta = np.sum(dout, axis=0)
  dgammax = dout
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step5
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar #注意这是 xmu 的一个支路

  #step4
  dsqrtvar = -1. /(sqrtvar**2) * divar
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step3
  dsq = 1. /N * np.ones((N,D)) * dvar
  dxmu2 = 2 * xmu * dsq #注意这是 xmu 的第二个支路

  #step2
  dx1 = (dxmu1 + dxmu2) 注意这是 x 的一个支路

  #step1
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  dx2 = 1. /N * np.ones((N,D)) * dmu 注意这是 x 的第二个支路

  #step0 done!
  dx = dx1 + dx2

  return dx, dgamma, dbeta
```

Batch Norm 即批规范化，目的是为了解决每批数据训练时的不规则分布给训练造成的困难，对批数据进行规范化，还可以在梯度反传时，解决梯度消失的问题。

Batchnorm 也是一种正则的方式，可以代替其他正则方式如 dropout，但通过这样的正则化，也消融了数据之间的许多差异信息。

### batchnorm 的几个参数，可学习的参数有哪些？

第四步加了两个参数 $\gamma$ 和 $\beta$，分别叫做缩放参数和平移参数，通过选择不同的 $\gamma$ 和 $\beta$ 可以让隐藏单元有不同的分布。这里面的 $\gamma$ 和 $\beta$ 可以从你的模型中学习，可以用梯度下降，Adam 等算法进行更新。

### Batch Normalization 的作用

神经网络在训练的时候随着网络层数的加深，激活函数的输入值的整体分布逐渐往激活函数的取值区间上下限靠近，从而导致在反向传播时低层的神经网络的梯度消失。而 Batch Normalization 的作用是通过规范化的手段，将越来越偏的分布拉回到标准化的分布，使得激活函数的输入值落在激活函数对输入比较敏感的区域，从而使梯度变大，加快学习收敛速度，避免梯度消失的问题。

不仅极大提升了训练速度，收敛过程大大加快；还能增加分类效果，一种解释是这是类似于 Dropout 的一种防止过拟合的正则化表达方式，所以不用 Dropout 也能达到相当的效果；另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等。

### BN 一般用在网络的哪个部分

先卷积再 BN

Batch normalization 的 batch 是批数据，把数据分成小批小批进行 stochastic gradient descent. 而且在每批数据进行前向传递 forward propagation 的时候，对每一层都进行 normalization 的处理。

### BN 为什么要重构

恢复出原始的某一层所学到的特征的。因此我们引入了这个可学习重构参数 $\gamma$、$\beta，让我们的网络可以学习恢复出原始网络所要学习的特征分布。

### BN 层反向传播，怎么求导

反向传播需要计算三个梯度值，分别是：

$$
\frac{\partial \ell}{\partial x_{i}}, \frac{\partial \ell}{\partial y}, \frac{\partial \ell}{\partial \beta} 
$$

定义 $\frac{\partial \ell}{\partial y_{i}}$ 为从上一层传递过来的残差。

$$
\frac{\partial \ell}{\partial \gamma}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} \cdot \overline{x_{i}}
$$

$$
\frac{\partial \ell}{\partial \beta}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}}
$$

下面计算 $\frac{\partial \ell}{\partial x_{i}}$，观察缩放和移位与归一化公式，可以看到从 $x_i$ 到 $y_i$ 的链式计算过程：

$$
\begin{gathered}
有 \quad x_{i} \gg \mu_\mathcal{B} \gg \sigma_{\mathcal{B}}^{2} \gg \overline{x_{i}} \gg y_{i}\\
同时 x_{i} \gg \mu_\mathcal{B} \gg \overline{x_{i}} \gg y_{i}\\
同时 x_{i} \gg \overline{x_{i}} \gg y_{i}\\
则 \frac{\partial \ell}{\partial x_{i}}=\frac{\partial \ell}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial x_{i}}\left(\frac{\partial \overline{x_{i}}}{\partial x_{i}}+\frac{\partial \overline{x_{i}}}{\partial \mu_\mathcal{B}} \cdot \frac{\partial \mu_\mathcal{B}}{\partial x_{i}}+\frac{\partial \overline{x_{i}}}{\partial \sigma_{B}^{2}} \cdot \frac{\partial \sigma_{B}^{2}}{\partial \mu_\mathcal{B}} \cdot \frac{\partial \mu_\mathcal{B}}{\partial x_{i}}\right)
\end{gathered}
$$

### batchnorm 训练时和测试时的区别

训练阶段：首先计算均值和方差（每次训练给一个批量，计算批量的均值方差），然后归一化，然后缩放和平移。

测试阶段：每次只输入一张图片，这怎么计算批量的均值和方差，于是，就有了代码中下面两行，在训练的时候实现计算好 mean、 var，测试的时候直接拿来用就可以了，不用计算均值和方差。

### 先加 BN 还是激活，有什么区别（先激活）

目前在实践上，倾向于把 BN 放在 ReLU 后面。也有评测表明 BN 放 ReLU 后面效果更好。