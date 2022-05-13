---
title: "The Perception-Distortion Tradeoff"
subtitle: "图像超分率失真"
layout: post
author: "L Hondong"
header-img: "img/post-bg-3.jpg"
mathjax: true
tags:
  - 图像超分
  - 率失真
  - GAN
---

# The Perception-Distortion Trade-off

Israel Institute of Technology

## 摘要

SR 问题，对于算法的评估主要是分为两类，一类是是像素维度上的，对于 MSE 的优化，但这会导致过平滑、模糊等效应。而另外一类，就是基于人眼的直观感受，比如 MOS tset、NIQE，图片更加的 sharp，更加符合人眼的直观感受，更加的自然，但是会存在的问题，比如原来是直的线，会变成倾斜等。而这篇论文就是为了证明感知质量和图像失真程度并不是完全对应的（实际上是对立的）。

基于深度网络的特征失真可能更好的表示图像感知质量的好坏，而使用 GAN 网络可以获得具有较好感知质量效果的图像。

## 一、简介

超分问题中的 loss 分为以下几种：

- Distortion (fullreference) measures——指重建图像与原始图像之间的，重建图像的整体与参考图像越像，评价指标越高。以 MSE 为代表
- Perceptual quality（感知评判因子）——is the degree to which it looks like a natural image, and has nothing to do with its similarity to any reference image.
- Human opinion based quality assessment
- No-reference quality measures——No-reference quality measures are commonly based on estimating deviations from natural image statistics.，such as NIQE
- GAN-based image restoration——an adversarial loss，which minimizes some distance between the distribution of images produced by the generator and the distribution of images in the training dataset.（最小化了由生成器生成的图像的分布与训练数据集中的图像分布之间的一些距离。）

作者证明了 perception 和 distortion 之间存在下面这样的一条曲线，并且左下角的区域是任何算法都无法达到的。一些一味注重优化 distortion 的算法可能既不有效又损害视觉质量（在曲线的右上方区域），说明了 GAN 方法的有效性（去逼近这个 bound）。对于不同的领域应该有不同的侧重点，比如对于医学领域可能会更注重 distortion accuracy，即与原图像的接近程度。这个图像也指导给出了一个新的衡量算法的方法，将算法的表现绘制到该坐标轴上（同时考虑 perceptual quality 和 distortion）。 

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-34-01.png" alt="Perception-Distortion_Tradeoff-2022-01-11-23-34-01" style="zoom:50%;" /></div>

### 1.1 Motivation

### 1.2 Contributions

## 二、相关工作

### 2.1 失真

**Distortion**：指的是重建图像 $\hat{x}$ 与原图像 $x$ 之间的不相似度。

失真用来衡量给定图像和参考图像之间的差异程度。在衡量 distortion 中使用的是 full-reference 方法，最常见的是 MSE，但是与图像之间的语义相似性比较差。其他衡量失真的标准包括 SSIM、MS-SSIM 等，最近基于神经网络的距离误差可以捕获更多的语义信息，从而获得高质量的重建。 (IFC， VIF， VSNR， FSIM)

### 2.2 感知质量

**Perceptual quality** ：仅指 $\hat{x}$ 的图像质量，与原图像无关。或者说是指 $\hat{x}$ 与真实图像的相似程度，实际上是与重建图像的分布和真实图像的分布的距离有关。

感知质量是使图像看起来和自然图像更像而不考虑其与参考图像之间的相似性。目前主要的评价方法包括基于人类评价的质量评估方法，无参评价方法（如 KL 散度，DIIVINE， BRISQUE， BLIINDS-II， NIQE 等）和基于 GAN 网络的评价方法，这些方法基本都是利用统计学的知识进行评价。

## 三、问题定义

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-34-25.png" alt="Perception-Distortion_Tradeoff-2022-01-11-23-34-25" style="zoom:50%;" /></div>

实际上自然图像可以看做是自然图像 $p(X)$ 的自然分布的一个实现，可以把失真后的图像 $y$ 看成是给定原图像 $x$ 在条件分布 $p(Y\mid X)$ 下产生的结果，失真后还原的图像 $x’$ 可以看成 $y$ 在条件分布 $p(X\mid Y)$ 产生的结果。失真的公式可以表示如下： 

$$
E(\Delta(X,\hat X)) \tag{1}
$$

感知质量指标的表达式如下（值越低越好）：

$$
d(p_X,p_{\hat X}) \tag{2}
$$

论文的目标是建立上面两个公式的平衡，首先说明为什么减少（1）不能必然导致较低的（2）。

这里论文主要使用了两种失真方式去衡量图像失真。一种是均方误差失真（MMSE），另外一种是 0-1 失真（MAP）。$x$ 的原始分布是 $\{-1,0,1\}$ 时，前者导致产生的结果是连续的，后者导致结果只有 $\{-1,1\}$。换言之就是失真评估的方法会使图像掉落自然分布的”流型“，从而使分布与原始图像不同。具体见下图所示： 

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-34-54.png" alt=/Perception-Distortion_Tradeoff-2022-01-11-23-34-54" style="zoom:50%;" /></div>

以 MSE 和 MAP 为例，说明了使用这两种方式进行复原的图像分布不一定等于原分布。虽然 MAP 在某些条件下 $p_{\hat X}=p_X$ 成立，但我们需要的是一个 stable distribution peserving distortion measure， 即对每一个 $p_{X,Y}$ 都成立。作者证明了这样的衡量标准是不存在的， 并在附录中给出了相关证明。

**由于这样的 stably distribution preserving 的衡量方法并不存在，因此 low distortion 不一定会导致好的 perception quality**。那么我们可以找到在某一个 distortion level 下的最佳 perceptual quality 吗？ 

## 四、感知-失真平衡

从上面可以看出低失真不一定能带来较好的感知质量，如何在给定失真的情况下得到最好的感知质量需要研究。信号恢复任务的感知-平衡函数由下式给出： 

$$
P(D)=\min_{P_{\hat X\mid Y}} d(p_X,p_{\hat X}),s.t.E[\Delta(X,\hat X)]\leq D \tag{3}
$$

distortion 为 MSE， $d(⋅,⋅)$ 为 KL divergence。

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-35-30.png" alt=/Perception-Distortion_Tradeoff-2022-01-11-23-35-30" style="zoom:30%;" /></div>

在这个曲线中，$D$ 增大， $P(D)$ 减小。曲线为 convex 并且对于更大的噪声现象更严重。

作者指出虽然这个任务很难进行分析，但上面例子的现象普遍存在，并在附录中给出了一定的证明。并且不是所有的 distortion measure 都有相同的 tradeoff function。对于一些捕捉了图像间语义关系的衡量方法，这个现象是 less severe 的。

> **定理**：如果 $d(p,q)$ 对于他的第二个参数是 convex 的（对任意的 $𝑝,q1,q2,\lambda \in[0,1]$ 有 $d(p,\lambda q1+(1−\lambda)q2)≤\lambda d(p,q1)+(1−\lambda)d(p,q2))$， 那么 $P(D)$ 是不单调增的且凸的。这条定理中的假设 $d(p,q)$ 是 convex 的条件并不是非常严苛，即使没有这个条件 $P(D)$ 也是 monotonically non-increasing 的。

该定理不需要对失真度量进行假设，也就是对任意的感知-失真都存在这样的度量。虽然这并不意味着所有的失真都存在相同的感知-失真函数。

$P(D)$ 的凸特性说明其在低失真和高感知的情况下存在非常严重的平衡。例如低失真情况下略微的失真改善会导致感知质量大幅度下降，同理，高感知情况下略微的感知改善会导致失真大量增加。 

然后讨论了感知-失真和速率失真理论的联系。信号的率失真函数和互信息密切相关。当然率失真和要讨论的情况有一些不同。

## 五、通过 GAN 进行平衡

GAN 损失函数定义为：

$$
\min_G \max_D \mathbb E_{X\sim p_{data}}[\log D(x)] + \mathbb E_{Z\sim p_{z}}[\log (1-D(G(z)))]
$$

可以通过 GAN 来设计接近感知-失真平衡曲线的估计器，实际上使用 GAN 方法就是一个 systematic way 来设计 estimator 逼近这个界限。具体可以通过修改损失函数得到：

$$
\ell_{gen}= \ell_{distortion} + \lambda\ell_{adv}\tag{4}
$$

第一项表示失真，第二项的生成器学习 $d(p_X,p_{\hat X})$，所以（4）式接近于学习目标：

$$
\ell_{gen}=E(\Delta(X,\hat X))+\lambda d(p_X,p_{\hat X})\tag{5}
$$

这里最小化（5）可以等效为最小化（3）, 变化的 $λ$ 产生变化的 $D$，从而产生感知-失真函数的估计量。可以使用这种方法去获得感知-失真平衡，将 $λ$ 设定为 [0,0.3] 之间，改变参数可得到曲线，如下图所示： 

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-36-11.png" alt=/Perception-Distortion_Tradeoff-2022-01-11-23-36-11" style="zoom:30%;" /></div>

同样可以看出，失真函数使用 MMSE 比使用 MAP 在产生相同失真情况下，感知质量更好，同时也比 Random draw 的失真要小。

去噪 WGAN estimator（D）与 MAP estimator 具有相同的失真，但具有更好的感知质量；与 Random draw estimator 有几乎相同的感知质量，但失真显著降低。

## 六、实用的评估算法

> 定义：如果算法 A 在失真和感知质量上优于算法 B，则称算法 A dominate 算法 B。（注意：如果无法同时在失真和感知质量上有较好的效果，那么算法 A 和算法 B 无法一方 dominate 另一方，认为它们具有同样好的效果）。 

> 定义：如果一个算法不被一组算法中的其他算法 dominate，那么这个算法被认为是可以接受的。 

论文这里说明失真一般通过全参考（FR）方式来度量，包括（RMSE/SSIM/MS-SSIM/IFC/VIF/VGG2.2) 等。为了评价图像的感知质量，这里采用了无参考 NR (NIQE) 指标。这里对 16 种 SR 算法从 FR 和 NR 进行评估，结果如下图所示： 

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-36-51.png" alt=/Perception-Distortion_Tradeoff-2022-01-11-23-36-51" style="zoom:50%;" /></div>

上图各部分都有共同的特点：

1. 左下方是空白的，这反映感知-失真平面中的不可到达区域。
2. NR 和 FR 指标是负相关的，这反映感知-失真的平衡。虽然 IFC 和 VIF 相比于 SSIM 和 MSE 可以更好的捕获视觉质量，但是也存在这种平衡。VGG2.2 的平衡现象比 MSE 略弱，说明其是一个相对更感知评价指标。从上图中还可以看出，从左到右失真增加，但是感知质量是提高的。 

当远离不可到达区域时，FR 和感知质量可以达到正相关，接近不可达达区域时却不符合，FR 此时不能用来衡量图像感知质量，而 NR 可以用来表示感知质量好坏。因此在评估时需要对 FR 和 NR 统一进行评估，兼顾失真和感知质量。具体如下图所示： 

<div align=center><img src="/assets/Perception-Distortion_Tradeoff-2022-01-11-23-37-12.png" alt=/Perception-Distortion_Tradeoff-2022-01-11-23-37-12" style="zoom:30%;" /></div>

在 2017 年之前，IFC 指标可以很好的匹配感知质量，2017 年以后就开始反相关。这篇论文说明了失真和感知质量之间存在矛盾，可以使用一对 NR 和 FR 指标进行评价比较。