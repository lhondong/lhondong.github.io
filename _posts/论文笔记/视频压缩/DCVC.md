---
title: "Deep Contextual Video Compression"
subtitle: "DCVC"
layout: post
author: "L Hondong"
header-img: "img/post-bg-46.jpg"
mathjax: true
tags:
  - 笔记
---

# DCVC

Deep Contextual Video Compression

[NIPS 2021]

## 摘要

过去的神经网络视频编码方法还是用的传统的思路，先生成预测帧，然后编码当前帧的残差。然而预测编码可能只是一个次优解，因为只使用了简单的减法去除冗余。

从预测编码到条件编码。

提出一个问题：

> How to define, use, and learn condition under a deep video compression framework.

设计了一种高效的条件编码框架，将时域（特征域 feature domain）上下文特征作为条件输入去帮助编解码器编码当前帧，从而充分挖掘条件编码的潜力。而这种设计也便于充分利用高维特征来帮助视频高频细节获得更好的重建质量。

与此同时，DCVC 是一个拓展性非常强的框架，其里面的上下文特征可以灵活设计。实验表明，在标准 1080p 视频上，所提出的 DCVC 相比 x265 (veryslow) 获得了 26.0% 的码率节省（PSNR 为指标）。在 DCVC 下，最新的方法相比 H.265-HM 有 14.4% 的码率节省（PSNR 为质量评价指标）。如果以 MS-SSIM 为质量评价指标，相比 H.266-VTM 则有 21.1% 的码率节省。

## 一、简介

### 1.1 Motivation

#### 残差编码

从 1988 年的 H.261 到 2020 年发布的 H.266，近 30 年来所有传统的视频编码标准都是基于残差编码的框架。在残差编码中，预测帧先会从之前已经解码的帧中生成出来，然后再计算当前帧与预测帧的残差。该残差会被编码变成码流，解码器将码流解码并获得重建后的残差，最后和预测帧相加获得解码帧。残差编码是一种简单高效的方式，但它的熵大于或等于条件编码的熵，并不是最优的方式。

因为给定预测帧 $\tilde{x}_t$，编码当前帧 $x_t$，使用手工相减的熵为 $H(x_t-\tilde{x}_t)$，而使用条件编码的熵为 $H(x_t \vert \tilde{x}_t)$，一般有：

$$H(x_t - \tilde{x}_t) \geq H(x_t \vert \tilde{x}_t)$$

另外从理论上讲，当前帧 $x_t$ 待编码的像素与之前的重建帧和当前帧 $x_t$ 已经重建的像素都可能有相关性。对于传统编码器，由于搜索空间巨大，使用人为制定的规则去显示地挖掘这些像素之间的相关性是一件非常困难的事情。因此残差编码假设当前像素只和预测帧对应位置的像素具有强相关性，这是条件编码的一个特例。

考虑到残差编码的简单性，最近的基于深度学习的视频压缩方法也大多采用残差编码，使用神经网络去替换传统编码器中的各个模块。

而本文认为，与其将视野局限在残差编码，更应该充分利用深度学习的优势去挖掘这些像素之间的相关性来设计条件编码，所以设计了一种基于条件编码的视频压缩方法，方法对比如图所示。

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-30-51.png" alt="DCVC-2022-02-10-08-30-51" style="zoom:50%;" /></div>

#### 条件编码

通常，在设计一个基于深度学习的条件编码框架时会遇到以下问题：“什么是条件？如何利用条件？如何学习条件？”

准确来讲，条件可以是任何能帮助当前帧编码的信息。一种简单直接的方法是把预测帧作为条件，虽然这样可行，但预测帧只包含 RGB 三个通道的像素信息，这会限制条件编码的潜力。既然已经采纳条件编码，为什么不可以让网络自动学习它所需要的条件？在 DCVC 里，网络在时域上学习生成上下文特征，该下文特征作为条件输入去帮助当前帧的编码和解码。

DCVC 实现框架如下图所示。至于如何学习上下文特征，首先设计了一个特征提取器将之前解码帧从像素域转换到特征域，同时利用运动估计去学习运动向量。该运动向量在经过编码和解码之后会指导网络从哪里提取特征。考虑到运动补偿引发的空间不连续性，本文又设计了一个上下文改进模块去生成最终的上下文特征。该上下文特征通过并以并联的方式作为编码器和解码器的条件输入。

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-35-56.png" alt="DCVC-2022-02-10-08-35-56" style="zoom:50%;" /></div>

### 1.2 Contributions

1. We design a deep contextual video compression framework based on conditional coding. The definition, usage, and learning manner of condition are all innovative. Our method can achieve higher compression ratio than previous residue coding-based methods.
2. We propose a simple yet efficient approach using context to help the encoding, decoding, as well as the entropy modeling. For entropy modeling, we design a model which utilizes spatial-temporal correlation for higher compression ratio or only utilizes temporal correlation for fast speed.
3. We define the conditionas the context in feature domain.The context with higher dimensions can provide richer information to help reconstruct the high frequency contents.
4. Our framework is extensible. There exists great potential in boosting compression ratio by better defining, using, and learning the condition

## 二、相关工作

### 2.1 

### 2.2 

## 三、方法

### 3.1 

在 DCVC 框架中，时域高维上下文特征为编码条件。相比传统的像素域的预测帧，高维特征可以提供更丰富的时域信息，不同的通道也可以有很大的自由度去提取不同类型的信息，从而帮助当前帧高频细节获得更好的重建。图4展示了在像素域的残差编码和在特征域的条件编码的误差对比。可以发现，研究员们的方法在背景和前景中的高频内容可以获得明显的重建误差减小，这主要得益于使用更丰富的高维特征。

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-51-47.png" alt="DCVC-2022-02-10-08-51-47" style="zoom:50%;" /></div>

### 3.2 

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-52-15.png" alt="DCVC-2022-02-10-08-52-15" style="zoom:50%;" /></div>

## 四、实验

### 4.1 

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-52-37.png" alt="DCVC-2022-02-10-08-52-37" style="zoom:50%;" /></div>

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-52-59.png" alt="DCVC-2022-02-10-08-52-59" style="zoom:50%;" /></div>

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/DCVC-2022-02-10-08-53-46.png" alt="DCVC-2022-02-10-08-53-46" style="zoom:50%;" /></div>

### 4.2 

## 五、总结

## 六、思考
