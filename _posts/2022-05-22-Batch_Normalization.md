---
title: "Batch Normalization"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-22.jpg"
mathjax: true
tags:
  - Batch Normalization
---

# Batch Normalization

机器学习领域有个很重要的假设：IID 独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。因此，在把数据喂给机器学习模型之前，“白化（whitening）”是一个重要的数据预处理步骤，其中最典型白化方法是 PCA。白化一般包含两个目的： 

1. 去除特征之间的相关性：独立； 
2. 使得所有特征具有相同的均值和方差 ：同分布。 

每批训练数据的分布各不相同，那么网络需要在每次迭代中去学习适应不同的分布，这样将会大大降低网络的训练速度。对于深度网络的训练是一个非常复杂的过程，只要网络的前面几层发生微小的改变，那么这些微小的改变在后面的层就会被累积放大下去。一旦网络某一层的输入数据的分布发生改变，那么这一层网络就需要去适应学习这个新的数据分布，所以如果训练过程中，训练数据的分布一直在发生变化，那么将会影响网络的训练速度。

## Internal Covariate Shift（内部协变量转移）

在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如 transfer learning/domain adaptation 等。而 covariate shift 就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有 $x \in \mathcal{X}$,

$$
P_{s}(Y \mid X=x)=P_{t}(Y \mid X=x)
$$

但是：

$$
P_{s}(X) \neq P_{t}(X)
$$

对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了 covariate shift 的定义。由于是对层间信号的分析，也即是“internal”的来由。 

简单点说就是：对于深度学习这种包含很多隐层的网络结构，在训练过程中，因为各层参数不停在变化，所以每个隐层都会面临 covariate shift 的问题，也就是在训练过程中，隐层的输入分布老是变来变去，这就是所谓的“Internal Covariate Shift”，Internal 指的是深层网络的隐层，是发生在网络内部的事情，而不是 covariate shift 问题只发生在输入层。 

那么 Internal Covariate Shift 会导致什么问题呢？ 

1. 每个神经元的输入数据不再是“独立同分布”；
2. 上层参数需要不断适应新的输入数据分布，降低学习速度；
3. 下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止；
4. 每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

Batch Normalization 调整了数据的分布，不考虑激活函数，它让每一层的输出归一化到了均值为 0 方差为 1 的分布，这保证了梯度的有效性，防止“梯度弥散”。

BN 的优点是：

1. 可以选择比较大的初始学习率，加快网络的收敛；同时使用更大的学习率，可以跳出局部极值，增强泛化能力；
2. 减少正则化参数的 Dropout、L2 正则项参数的选择问题，BN 具有提高网络泛化能力的特性；
3. 不需要使用局部响应归一化层（局部响应归一化是 Alexnet 网络用到的方法），因为 BN 本身就是一个归一化网络层；
4. 可以把训练数据彻底打乱，防止每批训练的时候，某一个样本经常被挑选到，在 ImageNet 上提高 1% 的精度。 

BN 的核心思想不是为了防止梯度消失或者防止过拟合，其核心是通过**对系统参数搜索空间进行约束来增加系统鲁棒性**，这种约束压缩了搜索空间，约束也改善了系统的结构合理性，这会带来一系列的性能改善，比如加速收敛，保证梯度，缓解过拟合等。

## BN 的计算

神经网络中传递的张量数据，其维度通常记为 [N, H, W, C]，其中 N 是 batch_size，H、W 是行、列，C 是通道数。那么上式中 BN 的输入集合 B 就是下图中蓝色的部分：

<div align=center><img src="/assets/Batch_normalization-2022-05-22-15-19-58.png" alt="Batch_normalization-2022-05-22-15-19-58" style="zoom:50%;" /></div>

均值的计算，就是在一个批次内，将每个通道中的数字单独加起来，再除以 N×H×W。举个例子：该批次内有 10 张图片，每张图片有三个通道 RBG，每张图片的高、宽是 H、W，那么均值就是计算 10 张图片 R 通道的像素数值总和除以 10×H×W，再计算 B 通道全部像素值总和除以 10×H×W，最后计算 G 通道的像素值总和除以 10×H×W。方差的计算类似。

可训练参数  的维度等于张量的通道数，在上述例子中，RBG 三个通道分别需要一个 $\gamma$ 和一个 $\beta$，所以 $\vec\gamma,\vec\beta$  的维度等于 3。

## 1. BN 在训练和测试时的差异？

对于 BN，在训练时，是对每一个 batch 的训练数据进行归一化，也即用每一批数据的均值和方差。

而在测试时，比如进行一个样本的预测，就并没有 batch 的概念，因此，这个时候用的均值和方差是在训练过程中通过滑动平均得到的均值和方差，这个会和模型权重一起，在训练完成后一并保存下来。

对于 BN，是对每一批数据进行归一化到一个相同的分布，而每一批数据的均值和方差会有一定的差别，而不是用固定的值，这个差别实际上也能够增加模型的鲁棒性，并会在一定程度上减少过拟合。

但是一批数据和全量数据的均值和方差相差太多，又无法较好地代表训练集的分布，因此，BN 一般要求将训练集完全打乱，并用一个较大的 batch 值，去缩小与全量数据的差别。

## 2. BN 中的移动平均 Moving Average 是怎么做的？

训练过程中的每一个 batch 都会进行移动平均的计算 [1]：

```python
moving_mean = moving_mean * momentum + batch_mean * (1 - momentum)
moving_var = moving_var * momentum + batch_var * (1 - momentum)
```

式中的 momentum 为动量参数，在 TF/Keras 中，该值为 0.99，在 Pytorch 中，这个值为 0.9 初始值，moving_mean=0，moving_var=1，相当于标准正态分布。

在实际的代码中，滑动平均的计算会以下面这种更高效的方式，但实际上是等价的：

```python
moving_mean -= (moving_mean - batch_mean) * (1 - momentum)
moving_var -= (moving_var - batch_var) * (1 - momentum)
```

## 3. 移动平均中 Momentum 参数的影响

整个训练阶段滑动平均的过程，（moving_mean, moving_var）参数实际上是从正态分布，向训练集真实分布靠拢的一个过程。

理论上，训练步数越长是会越靠近真实分布的，实际上，因为每个 batch 并不能代表整个训练集的分布，所以最后的值是在真实分布附近波动。

一个更小的 momentum 值，意味着更大的更新步长，对应着滑动平均值更快的变化，能更快地向真实值靠拢，但也意味着更大的波动性，更大的 momentum 值则相反。

训练阶段使用的是（batch_mean, batch_var），所以滑动平均并不会影响训练阶段的结果，而是影响预测阶段的效果。

如果训练步数不够，一个大的 momentum 值可能会导致（moving_mean, moving_var）还没有靠拢到真实分布就停止了，这样对预测阶段的影响是很大的，也会是欠拟合的状态。如果训练步数足够，一个大的 momentum 值对应小的更新步长，最后的滑动平均的值是会更接近真实值的。

如果 batch size 比较小，那单个 batch 的（batch_mean, batch_var）和真实分布会比较大，此时滑动平均单次更新的步长就不应过大，适用一个大的 momentum 值，反之可类比分析。

## 4. Norm 中的标准化、平移和缩放的作用

Internal Covariate Shift 导致的问题：

1. 上层参数需要不断适应新的输入数据分布，降低学习速度。
2. 下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。
3. 每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

Norm 的使用，可以很好的改善这些问题，它包括两个操作：标准化 + 平移和缩放。

首先，抛开神经网络的背景，标准化这个操作，可以消除不同数据之间由于量纲/数量级带来的差异，而且它是一种线性变换，并不会改变原始数据的数值排序关系。

经过标准化，每层神经网络的输出变成了均值为 0，方差为 1 的标准分布，不再受到下层神经网络的影响（注意是数据分布不受影响，但具体的值肯定还是依赖下层神经网络的输出），可以有效改善上述的几个问题。

那平移和缩放的作用在哪里呢？主要还是为了：**保证模型的表达能力不因为标准化而下降**。

在旧参数中，$x$ 的均值和方差取决于下层神经网络的复杂关联；但在新参数中，在 Norm 中引入了两个新参数 $\gamma$ 和 $\beta$，$x$ 的均值和方差仅由  $\gamma$ 和 $\beta$ 来确定，去除了与下层计算的密切耦合。新参数很容易通过梯度下降来学习，简化了神经网络的训练。

如果直接只做标准化不做其他处理，神经网络是学不到任何东西的，因为标准化之后都是标准分布了，但是加入这两个参数后就不一样了。

先考虑特殊情况，如果 $\gamma$ 和 $\beta$ 分别等于此 batch 的标准差和均值，那么 $y$ 不就还原到标准化前的 $x$ 了吗，也即是缩放平移到了标准化前的分布，相当于 Norm 没有起作用。这样就保证了每一次数据经过 Norm 后还能保留学习来的特征，同时又能完成标准化这个操作，从而使当前神经元的分布摆脱了对下层神经元的依赖。

注：该问题主要参考 [2]，讲解非常清晰和系统。

## 5. 不同 Norm 方法中都有哪些参数要保存？

BN 的参数包括：

1. 每个神经元在训练过程中得到的均值和方差，通过移动平均得到
2. 每个神经元上的缩放参数 $\gamma$ 和平移参数 $\beta$

LN 只包括上面的第 2 部分参数，因为它和 batch 无关，无需记录第 1 部分参数。

## 6. BN 和 LN 有哪些差异？

Batch Norm 是对不同样本间的同一维特征做归一化，即标准化某一特征整体的分布。
Layer Norm 是对同一样本的不同维特征问做归一化，即标准化某一样本特征的分布。

<div align=center><img src="/assets/Batch_normalization-2022-05-21-23-36-25.png" alt="Batch_normalization-2022-05-21-23-36-25" style="zoom:50%;" /></div>

1. 两者做 Norm 的维度不一样，BN 是在不同样本的同一特征维度，而 LN 一般是在样本的维度。
2. BN 需要在训练过程中，滑动平均累积每个神经元的均值和方差，并保存在模型文件中用于推理过程，而 LN 不需要。
3. 因为 Norm 维度的差异，使得它们适用的领域也有差异，BN 更多用于 CV 领域，LN 更多用于 NLP 领域。

## 7. 为什么 Transformer/BERT 使用 LN，而不使用 BN？

最直白的原因，还是因为用 LN 的效果更好，这个在 Transformer 论文中作者有说到。但背后更深层次的原因，这个在知乎上有广泛的讨论 [3]，不过并没有一个统一的结论，这里我摘录两个我相对认可的回答 [4][5] ：

**图像数据是自然界客观存在的，像素的组织形式已经包含了 “信息”，而 NLP 数据不一样，网络对 NLP 数据学习的真正开端是从'embedding'开始的，而这个'embedding'并不是客观存在，它也是通过网络学习出来的。**

下面只从直观理解上从两个方面说说个人看法： 

1. layer normalization 有助于得到一个球体空间中符合 0 均值 1 方差高斯分布的 embedding，batch  normalization 不具备这个功能；
2. layer normalization 可以对 Transformer 学习过程中由于多词条 embedding 累加可能带来的 “尺度” 问题施加约束，相当于对表达每个词**一词多义的空间施加了约束**，有效降低模型方差。batch normalization 也不具备这个功能。 

Emmbedding 并不存在一个客观的分布，那我们需要考虑的是：**我们希望得到一个符合什么样分布的 embedding?** 

很好理解，**通过 layer normalization 得到的 embedding 是以坐标原点为中心，1 为标准差，越往外越稀疏的球体空间中。**

说简单点，其实深度学习里的正则化方法就是 “通过把一部分不重要的复杂信息损失掉，以此来降低拟合难度以及过拟合的风险，从而加速了模型的收敛”。Normalization 目的就是让分布稳定下来（降低各维度数据的方差）。 

不同正则化方法的区别只是操作的信息维度不同，即选择损失信息的维度不同。 

在 CV 中常常使用 BN，它是在 N×H×W 维度进行了归一化，而 Channel 维度的信息原封不动，因为可以认为在 CV 应用场景中，数据在不同 channel 中的信息很重要，如果对其进行归一化将会损失不同 channel 的差异信息。 

而 NLP 中不同 batch 样本的信息关联性不大，而且由于不同的句子长度不同，强行归一化会损失不同样本间的差异信息，所以就没在 batch 维度进行归一化，而是选择 LN，只考虑的句子内部维度的归一化。可以认为 NLP 应用场景中一个样本内部维度间是有关联的，所以在信息归一化时，对样本内部差异信息进行一些损失，反而能降低方差。 

总结一下：选择什么样的归一化方式，取决于你关注数据的哪部分信息。如果某个维度信息的差异性很重要，需要被拟合，那就别在那个维度进行归一化。

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

## 参考文献

[1] https://jiafulow.github.io/blog/2021/01/29/moving-average-in-batch-normalization/ 
[2] https://zhuanlan.zhihu.com/p/33173246
[3] https://www.zhihu.com/question/395811291
[4] https://www.zhihu.com/question/395811291/answer/1260290120
[5] https://www.zhihu.com/question/395811291/answer/1251829041
[6] https://zhuanlan.zhihu.com/p/242086547