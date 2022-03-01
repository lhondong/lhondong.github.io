---
title: "Generative Adversarial Networks for Extreme Learned Image Compressiona"
subtitle: "GAN 极限图像压缩"
layout: post
author: "L Hondong"
header-img: "img/post-bg-3.jpg"
mathjax: true
tags:
  - 图像压缩
  - GAN
---

# Generative Adversarial Networks for Extreme Learned Image Compressiona

ICCV 2019

ETH Zürich

## 摘要

提出了一种基于对抗生成网络 (GAN) 可学习的极限压缩模型，在低比特率 (bpp<0.1) 时也能在视觉上获得令人满意的结果。极限压缩模型有两种方式，第一种称作生成压缩，第二种称作选择生成压缩，其中选择性压缩需要使用语义分割图。

## 一、简介

### 1.1 Motivation

传统的有损图像压缩技术（如 JPEG 和 WebP) 并不是专门为要压缩的数据设计的，因此并没有达到图像的可能最佳压缩率。在已有深度学习模型的基础上，设计了一个基于 GAN 的图像压缩系统，在满足恢复质量的前提下，实现了极高的压缩率。模型结合了编码器、解码器/生成器和判别器，同时训练解码器/生成器和鉴别器，以此来实现压缩目标，与高质量的 JPEG 等标准方法相比，压缩率提高了数量级，同时能够生成更直观的重建，保持对原始图像的更高保真度。最后，使用 PSNR 和 MS-SSIM 标准度量方法对模型进行评估，并将其与 JPEG、WebP 和已有的深度学习压缩模型等作比较，结果证实基于 GAN 的压缩模型在压缩率更高的情况下，比已有的压缩方案结果更好。

### 1.2 Contributions

## 二、相关工作

### 2.1

### 2.2

### 2.3

## 三、方法

### 3.1 生成压缩 (Generative Compression, GC)

生成压缩模式下会保留整个图像的内容，生成不同规模的结构比如树上的树叶或者建筑物上的窗户，并且生成模式不需要语义图的参与，不论是训练还是实际的编解码过程。生成压缩模式典型用于带宽受限的情况下。

GC 模式的基本结构和流程如下：

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/Generative Adversarial Networks for Extreme Learned Image Compression-2022-01-12-00-03-57.png" alt="Generative Adversarial Networks for Extreme Learned Image Compression-2022-01-12-00-03-57" style="zoom:50%;" /></div>

流程：输入 → 编码器 → 量化 → 解码器 → 输出

结构：

- 编码器 c7s1=60，d120，d240，d480，d960，c3s1，C，q
- 解码器/生成器：c3s1-960,R960,R960,R960,R960,R960,R960,R960, R960,R960,u480,u240,u120,u60,c7s1-3

其具体的参数配置方式详见 (Wang et al., 2018)[1]。量化器采用聚类量化方式，其中量化长度 L=5，聚类中心为{-2，-1, 0, 1, 2}。对于编码最后一层 bottleneck 的通道数 C 可取 2,4,8 和 16，其值越大，所对应的的 bpp 也越大，训练时默认 C=8，相对应的 bpp=0.08 左右。

对于输入 x 编码和量化之后得到特征图 $\hat{w}$，此时有一个可选择的采样噪声特征图 z（此特征图是由固定的先验分布比如高斯分布生成）与 $\hat{w}$ 相拼接，然后送入解码器/生成器生成重构图像。注意，此处的采样噪声时可选择的。

在训练模型时，都是由 GAN 模型训练方式，本文中一种是传统的训练方式，还有一种方式就是添加语义图作为额外信息送入判别器的方式。

### 3.2 选择生成压缩 (Selective Generative Compression, SC)

选择生成压缩模式是当保留用户定义的具有高度细节的区域，由语义图来生成图像中不保留的那部分区域。SC 模式可用于视频通话的场景，当视频流中需要完全保留人，而背景则可以合成。

选择生成模型的网络框架如下图所示：

<div align=center><img src="https://cdn.jsdelivr.net/gh/lhondong/Assets/Images/Generative Adversarial Networks for Extreme Learned Image Compression-2022-01-12-00-04-58.png" alt="Generative Adversarial Networks for Extreme Learned Image Compression-2022-01-12-00-04-58" style="zoom:50%;" /></div>

其中图像编码器 E(Image encoder)，语义图编码器 (Semantic label map encoder) 以及解码器 (Generator/decoder) 的具体结构如下：

- Encoder GC: c7s1-60, d120, d240, d480, d960, c3s1-C, q
- Encoders SC:
  -  Semantic label map encoder: c7s1-60, d120, d240, d480, d960
  - Image encoder: c7s1-60, d120, d240, d480, c3s1-C, q, c3s1-480, d960

The outputs of the semantic label map encoder and the image encoder are concatenated and fed to the generator/decoder.

- Generator/decoder: c3s1-960, R960, R960, R960, R960, R960, R960, R960, R960, R960, u480, u240, u120, u60, c7s1-3

对于 SC 模式，作者构建一个单通道的二值 heatmap 与量化后的特征图的空间尺寸 (height 与 width)，其中 0 对应的区域应该完全合成，而 1 对应的区域则保留特征图中相应区域的内容。

然而对于压缩而言，要求完全合成的区域与原图的语义相同。假设语义 s 分别存储，将它们送入生成器之前先经过特征提取器 F，为了由语义来引导网络，作者对失真 d 作 mask 操作，这样便可以只计算所保留区域的损失。而且，压缩后的特征图 $\hat{w}$ 中需要合成的区域的值置 0。假设 heatmap 也被存储，那么只需要对 $\hat{w}$ 中需要保留的区域进行编码（指的是熵编码），这样可以大大的减少需要存储的比特数。通常 $\hat{w}$ 的比特数要远大于存储语义与 heatmap，这种方法在比特数上可以节省很多。

### 3.3

## 四、实验

在训练时有两种模式：

1. Random Instance(RI)：在语义标签图中的实例随机选择 25%，然后保存这部分；
2. Randim Box(RB)：随机均匀的选取一个图像位置然后一个随机维度的 box；

RI 模式适合大部分使用情况；而 RB 模式对生成器会造成更多具有挑战性的情形，因为它需要将所保存的 box 无缝的整合到生成的内容中。

### 训练细节

损失函数构成：

失真损失：MSE * 10

特征匹配损失：L_FM * 12

感知损失：L_VGG * 12

在 OpenImages 数据集上训练的后半部分（训练 14W Iterations 之后），生成器/解码器的归一化（本方法使用的是 Instance Normalization) 操作固定，不进行参数更新，作者发现这样可以减少伪影和色差。

## 五、总结

## 参考文献

- Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High resolution image synthesis and semantic manipulation with conditional gans. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

## [实现](https://github.com/Justin-Tan/generative-compression)

网络的架构是基于论文 Perceptual Losses for Real-Time Style Transfer and Super-Resolution 中的附录中提供的描述完成的，项目中最初提到的多规格鉴别器的损失是基于论文 High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs 完成的。

生成器网络从低维潜在空间映射到高维图像空间。

对于下面的公式，我们将 latent space 潜空间称为 z 空间，将图像空间称为 X 空间。从 X 到 z 的映射并不是 GAN 网络的固有特性，但是这种映射正是压缩所必需的编码步骤。 因此，从 z 空间中随机初始化的向量通过随机梯度下降进行训练，其重建目标是在适当的损失函数下尽可能接近原始图像。

在生成器模型中，条件变量 y 实际上是作为一个额外的输入层（additional input layer），它与生成器的噪声输入 p(z) 组合形成了一个联合的隐层表达；在判别器模型中，y 与真实数据 x 也是作为输入，并输入到一个判别函数当中。实际上就是将 z 和 x 分别于 y 进行 concat，分别作为生成器和判别器的输入，再来进行训练。

CGAN 在 mnist 数据集上进行了实验，对于生成器：使用数字的类别 y 作为标签，并进行了 one-hot 编码，噪声 z 来自均均匀分布；噪声 z 映射到 200 维的隐层，类别标签映射到 1000 维的隐层，然后进行拼接作为下一层的输入，激活函数使用 ReLU；最后一层使用 Sigmoid 函数，生成的样本为 784 维。

## Feature matching Loss

论文：[Improved Techniques for training GANS](https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)

GANs 的 loss 是建立在 discriminator 的输出上的，即 discriminator 输出的交叉熵，这可能导致 GANs 训练不稳定，毕竟给予 generator 的信息太少了，而图像空间又太大了。为了让训练更稳定，作者提出了 feature matching 的方法。所谓的 feature matching，即是要求 generator 产生的图像在经过 discriminator 时，提取的特征尽可能地接近（匹配）自然图像经过 discriminator 时提取的特征。设 discriminator 的某个中间层对于输入 x 的输出为 f(x)，作者提出的 feature matching，实际上是用下面的 loss function 替换以前的交叉熵 loss：这个 loss 比交叉熵 loss 更加直接，对 generator 的指导作用更大一些。