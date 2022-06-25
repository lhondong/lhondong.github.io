---
title: "MobileNet"
subtitle: "Efficient Convolutional Neural Networks for Mobile Vision Applications"
layout: post
author: "L Hondong"
header-img: "img/post-bg-33.jpg"
mathjax: true
tags:
  - 模型压缩
  - 轻量化
---

输入图片（Input）大小为 $I\times I$，卷积核（Filter）大小为 $K\times $，步长（stride）为 s，填充（Padding）的像素数为 p，那卷积层输出（Output）的特征图大小为：

$$
O=\frac{I-K+2p}{s}+1
$$

## 可分离卷积

可分离卷积用于某些神经网络体系结构中，例如 MobileNet。可以在空间上（空间可分离卷积）或在深度上（深度可分离卷积）进行可分离卷积。

### 空间可分离卷积

空间可分离卷积在图像的 2D-空间维度（即高度和宽度）上运行。从概念上讲，空间可分离卷积将卷积分解为两个单独的运算。对于下面显示的示例，将 Sobel 的 kennel（3x3 的 kennel）分为 3x1 和 1x3 的 kennel。

$$
\left[\begin{array}{rrr}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{array}\right]=\left[\begin{array}{l}
1 \\
2 \\
1
\end{array}\right] \times\left[\begin{array}{lll}
-1 & 0 & 1
\end{array}\right]
$$

在卷积中，3x3 的 kennel 直接与图像卷积。在空间可分离卷积中，3x1 的 kennel 首先与图像进行卷积，然后应用 1x3 的 kennel。在执行相同操作时，空间可分离卷积只需要 6 个参数而不是 9 个参数。

此外，在空间可分离卷积中需要比卷积更少的矩阵乘法。举一个具体的例子，用 3 x 3 的 kennel（步长=1，填充=0）在 5 x 5 图像上进行卷积，需要在水平 3 个位置（垂直 3 个位置）上扫描的 kennel，总共 9 个位置（在下图中以点表示）。在每个位置上，将应用 9 个按元素的乘法，总体来说，这是 9 x 9 = 81 个乘法运算。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-01-38-18.png" alt="MobileNet-2022-04-01-01-38-18" style="zoom:100%;" /></div>

另一方面，对于空间可分离卷积，首先在 5 x 5 图像上应用 3 x 1 的 filter。在水平 5 个位置和垂直 3 个位置扫描这样的 kennel，总的位置是 5×3 = 15（表示为下面的图像上的点）。在每个位置，应用 3 个逐元素的乘法，那就是 15 x 3 = 45 个乘法运算。

现在，我们获得了一个 3 x 5 的矩阵，此矩阵与 1 x 3 的 kennel 卷积，该 kennel 在水平 3 个位置和垂直 3 个位置扫描矩阵，总共 9 个位置。对于这 9 个位置中的每一个，将应用 3 个按元素的乘法，此步骤需要 9 x 3 = 27 个乘法运算。因此，总的来说，空间可分离卷积需要 45+27=72 个乘法运算，小于标准卷积的 81 个乘法运算。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-01-38-52.png" alt="MobileNet-2022-04-01-01-38-52" style="zoom:100%;" /></div>

比起卷积，空间可分离卷积要执行的矩阵乘法运算也更少。假设现在在 $m\times m$ 卷积核、卷积步长=1 、填充=0 的 $N\times N$ 图像上做卷积。传统的卷积需要进行 $(N-2)\times (N-2) \times m \times m$ 次乘法运算，而空间可分离卷积只需要进行 $N\times (N-2) \times m + (N-2)\times (N-2) \times m=(2N-2)\times (N-2) \times m$ 次乘法运算。

空间可分离卷积与标准的卷积的计算成本之比为：

$$
\frac{2}{m}+\frac{2}{m(N-2)}
$$

对于图像的大小 $N$ 远远大于 filters 大小 $m$（$N >> m$）的图层，比值变为 $2 / m$。这意味着在这种渐近情况下（$N >> m$），空间可分离卷积的计算成本是标准卷积的 2/3（3 x 3 的 filters）。对于 5 x 5 的 filters 为 2/5，对于 7 x 7 的 filters 为 2/7，依此类推。

尽管空间可分离卷积节省了成本，但很少在深度学习中使用它。主要原因之一是并非所有 kennels 都可以分为两个较小的 kennels。如果用空间可分离卷积代替所有传统的卷积，在训练过程中，我们将限制卷积核的类型，训练结果可能不是最佳的。

### 深度可分离卷积

深度可分离卷积是深度学习中更常用的方法（例如，在 MobileNet 和 Xception 中），它包括两个步骤：深度卷积和 1x1 卷积。

首先回顾一下在前几节中讨论的 2D-卷积和 1 x 1 卷积。标准的 2D 卷积，举一个具体的例子，假设输入层的大小为 $7\times 7 \times 3$，而 filters 的大小为 $ 3 \times 3 \times 3$。用一个 filters 进行 2D 卷积后，输出层为大小为 $ 5\times 5 \times 3$ （只有 1 个通道）。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-01-54-52.png" alt="MobileNet-2022-04-01-01-54-52" style="zoom:100%;" /></div>

通常，在两个神经网络层之间应用多个 filters。假设这里有 128 个 filters，应用这 128 个 2D 卷积后，我们得到 128 个 $5\times 5 \times 1$ 输出特征图。然后，将这些特征图堆叠到大小为 $5\times 5 \times 128$ 的单层中。我们将输入层（$7\times 7 \times 3$）转换为输出层（$5\times 5 \times 128$），在扩展深度的同时，空间尺寸（即高度和宽度）会缩小。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-02-01-29.png" alt="MobileNet-2022-04-01-02-01-29" style="zoom:100%;" /></div>

接下来看看使用深度可分离卷积如何实现同样的转换。

#### 一、逐通道卷积 Depthwise Convolution

首先第一步，在输入层上应用深度卷积。在 2D-卷积中分别使用 3 个卷积核（每个 filter 的大小为 $3 \times 3 \times 1$ ），而不使用大小为 $3 \times 3 \times 3$ 的单个 filter。每个卷积核仅对输入层的 1 个通道做卷积，这样的卷积每次都得出大小为 $ 5 \times 5 \times 1$ 的映射，之后再将这些映射堆叠在一起创建一个 $5\times 5 \times 3$ 的特征图，最终得出一个大小为 $5\times 5 \times 3$ 的输出图像。这样的话，图像的深度保持与原来的一样。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-02-01-08.png" alt="MobileNet-2022-04-01-02-01-08" style="zoom:100%;" /></div>

#### 二、逐点卷积 Pointwise Convolution

深度可分离卷积的第二步是扩大深度，用大小为 $1 \times 1 \times 3$ 的卷积核做 $1\times 1$ 卷积。每个 $1 \times 1 \times 3$ 卷积核对 $5 \times 5 \times 3$ 输入图像做卷积后都得出一个大小为 $5 \times 5 \times 1$ 的特征图。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-02-03-36.png" alt="MobileNet-2022-04-01-02-03-36" style="zoom:100%;" /></div>

这样的话，做 128 次 $1\times 1$ 卷积后，就可以得出一个大小为 $5 \times 5 \times 128$ 的层。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-02-04-56.png" alt="MobileNet-2022-04-01-02-04-56" style="zoom:100%;" /></div>

深度可分离卷积完成这两步后，同样可以将一个 $7 \times 7 \times 3$ 的输入层转换为 $5 \times 5 \times 128$ 的输出层。

下图展示了深度可分离卷积的整个过程：

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/MobileNet-2022-04-01-02-05-56.png" alt="MobileNet-2022-04-01-02-05-56" style="zoom:100%;" /></div>

从本质上说，深度可分离卷积就是 3D 卷积 kernels 的分解（**在深度 channel 上的分解**），而空间可分离卷积就是 2D 卷积 kernels 的分解（**在 WH 上的分解**）。

那么，进行深度可分离卷积有什么好处？效率！与标准的 2D 卷积相比，深度可分离卷积所需的运算量少得多。

标准的 2D 卷积例子中的计算成本：128 个 $3 \times 3 \times 3$ 的卷积核移动 $5 \times 5$ 次，总共需要进行的乘法运算总数为 $128 \times 3 \times 3 \times 3 \times 5\times 5 =86400$ 次。

对于深度可分离卷积呢，在深度卷积这一步，有 3 个 $3 \times 3 \times 1$ 的卷积核移动 $5 \times 5$ 次，总共需要进行的乘法运算次数为 $3 \times 3 \times 3 \times 1 \times 5 \times 5=675$ 次；在第二步的 $1\times 1$ 卷积中，有 128 个 $1 \times 1 \times 3$ 的卷积核移动 $5 \times 5$ 次，总共需要进行的乘法运算次数为 $128 \times 1 \times 1 \times 3 \times 5 \times 5=9600$ 次。因此，深度可分离卷积共需要进行的乘法运算总数为 10275 次，花费的计算成本仅为 2D 卷积的 12%。

同样，我们对深度可分离卷积进行归纳。假设输入为 $h \times w \times d$，应用 $n$ 个 $k \times k \times d$ 的 filters（步长为 1，填充为 0，$h$ 为偶数），同时输出层为 $(h-k+1) \times (w-k+1) \times n$。

- 2D 卷积的计算成本：$n \times k \times k \times d \times (h-k+1) \times (w-k+1)$
- 深度可分卷积的计算成本：$d \times k \times k \times 1 \times (h-k+1) \times (w-k+1) + n \times 1 \times 1 \times d \times (h-k+1) \times (w-k+1)=(k\times k +n)\times d \times (h-k+1) \times (w-k+1)$
- 深度可分卷积和 2D 卷积所需的计算成本比值为：$\frac{1}{n}+\frac{1}{k^2}$

目前大多数网络结构的输出层通常都有很多通道，可达数百个甚至上千个。该情况下（$n>>k$），则上式可简写为 $\frac{1}{k^2}$。基于此，如果使用 $3\times3$ 的 filters，则 2D 卷积的乘法计算次数比深度可分离卷积多 9 倍。对于 $5\times5$ 的 filters，2D 卷积的乘法次数是其 25 倍。

#### 深度可分离卷积缺点

深度可分离卷积可大幅度减少卷积的参数。因此对于规模较小的模型，如果将 2D 卷积替换为深度可分离卷积，其模型大小可能会显著降低，模型的能力可能会变得不太理想，因此得到的模型可能是次优的。但如果使用得当，深度可分离卷积能在不牺牲模型性能的前提下显著提高效率。
