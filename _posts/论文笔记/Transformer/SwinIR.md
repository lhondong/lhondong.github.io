---
title: "论文笔记模板"
subtitle: "论文笔记模板"
layout: post
author: "L Hondong"
header-img: "img/post-bg-3.jpg"
mathjax: true
tags:
  - Template
  - 笔记
---

# SwinIR: Image Restoration Using Swin Transformer

Swin Transformer 提出了一种针对视觉任务的通用的 Transformer 架构，在分类，检测，分割任务上均达到了最优。SwinIR 继承了 Swin Transformer 的结构，是一个用于图像恢复的强基线模型，在图像超分辨率、去噪等任务上表现 SOTA。

图像复原 (Image restoration) 是一个长期存在的 low-level 视觉问题，旨在从低质量退化的图像 （例如，缩小、嘈杂和压缩图像） 中恢复高质量干净的图像。虽然最先进的图像恢复方法基于卷积神经网络，但很少有人尝试使用在 high-level 视觉任务中表现出优越性能的 Transformer。

基于卷积网络的图像复原模型有两个缺点：

1. 图像和卷积核之间的交互是与内容无关的。使用相同的卷积核来恢复不同的图像区域可能不是最佳选择。
2. 受到卷积核感受野大小的限制，卷积无法建模长距离的相关性。

针对第二个问题，目前已有一些基于 Transformer 网络的图像复原模型，比较有效的比如：IPT, U-Transformer, Video super-resolution transformer。

作为 CNN 的一个替代品，Transformer 设计了一个自注意力机制来获取内容之间全局交互的信息，并且在多种视觉任务上取得了不错的性能。但是针对图像复原的视觉 Transformer 通常将输入图像分割为固定大小的 patch（比如 48×48)，并且独立的处理每个 patch。这种策略不可避免的引入两类缺陷。

1. 边界像素点不能使用 patch 范围之外的邻接像素来做图像复原。
2. 复原后的图像中容易在 patch 周围引入边界伪影。虽然这个问题可以通过 patch 交叠来减轻，但是这会引入额外的计算负担。

Swin Transformer 在多种视觉任务上展示了巨大的前景，因为它集成了 CNN 和 Transformer 的优势。

1. 由于局部注意机制 (local attention mechanism)，它具有 CNN 处理大尺寸图像的优势。
2. 具有 Transformer 的优势，可以用 shift window 方案来建模远程依赖关系。
3. 这种基于注意力机制的模型图像和卷积核之间的交互是与内容有关的，可以理解成一种 spatially varying 的卷积操作。所以在一定程度上解决了 CNN 和 Transformer 模型的缺点。

SwinIR 由三部分组成：浅层特征提取，深层特征提取和高质量图像重建。特别的，深层特征提取模块由一系列的残差 Swin Transformer 块组成，每个 RSTB 包含一些 Swin Transformer 层及一个残差连接。

- 浅层特征提取模块使用 CNN 提取浅层特征，这些特征最后会直接传递到重建模块以便保留低频信息；
- 深层特征提取模块主要由 RSTB 模块组成，每个 RSTB 块使用多个 Swin Transformer 层用于局部注意力和窗口间交互；
- 在模块的最后增加了一个卷积层用来做特征增强，并且使用了一个残差连接用来提供特征融合的 shortcut。

最终，浅层和深层特征融合进重建模块重建高质量图像。相比于流行的基于 CNN 的图像复原模型，基于 Transformer 的 SwinIR 有如下几个好处：

1. 图像内容和注意力权重之间基于内容的交互，可以被解释为空间变化卷积。
2. 移动窗口机制有助于长范围依赖的建模。
3. 更少的参数，更好的性能，相比于已存在的图像复原模型，SwinIR 使用更少的参数得到的更好的 PSNR。

## 网络结构

<div align=center><img src="/assets/SwinIR-2022-05-28-19-55-51.png" alt="SwinIR-2022-05-28-19-55-51" style="zoom:50%;" /></div>

### 1. 浅层特征提取模块 (shallow feature extraction)

给定输入的低质量图片 $I_{LQ} \in \mathbb{R}^{H \times W \times C_{in}}$, 使用一个 3×3 卷积 $H_{SF}(\cdot)$ 来提取它的浅层特征 $F_{0} \in \mathbb{R}^{H \times W \times C}$:

$$
F_{0}=H_{S F}\left(I_{L Q}\right)
$$

### 2. 深层特征提取模块 (deep feature extraction)

给定上一阶段的输出特征 $F_{0}$, 使用深层特征提取模块 $H_{DF}$ 进行深层特征提取：

$$
F_{DF}=H_{DF}\left(F_{0}\right)
$$

深层特征提取模块由 $K$ 个 Residual Swin Transformer Blocks (RSTB) 和一个卷积操作组成。每个 RSTB 模块的输出 $F_{1}, F_{2}, \ldots, F_{K}$ 和最终的输出特征是：
$$
F_{i}=H_{RSTB_{i}}\left(F_{i-1}\right), \quad i=1,2, \ldots, K
$$

$$
F_{DF}=H_{CONV}\left(F_{K}\right)
$$

式中，$H_{RSTB_{i}}(\cdot)$ 表示第 $i$ 个 RSTB 模块，$H_{CONV}$ 表示最终的卷积层，使用它的目的是将卷积网络的归纳偏差 (inductive bias) 融入基于 Transformer 的网络，并为后来的浅层和深度特征奠定了更好的基础。

每个 Residual Swin Transformer Block 的内部结构如上图所示。它由一堆 STL (Swin Transformer Layer) 和一个卷积操作，外加残差链接构成。写成表达式就是：

$$
F_{i,j}=H_{STL_{i, j}}\left(F_{i, j-1}\right), \quad j=1,2, \ldots, L
$$

$$
F_{i, out}=H_{C O N V_{i}}\left(F_{i, L}\right)+F_{i, 0}
$$

式中，$H_{STL_{i,j}}(\cdot)$ 代表第 $i$ 个 RSTB 的第 $j$ 个 STL (Swin Transformer Layer), $H_{CONV_{i}}(\cdot)$ 代表第 $i$ 个 RSTB 的卷积操作，$F_{i, 0}$ 代表残差连接。

每个 RSTB 的残差链接使得模型便于融合不同级别的特征，卷积操作有利于增强平移不变性。

Swin Transformer Layer 由 2 个连续的 Swin Transformer Layer 组成：

$$
\begin{aligned}
&\hat{\mathbf{z}}^{l}=\mathrm{W}-\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}^{l-1}\right)\right)+\mathbf{z}^{l-1} \\
&\mathbf{z}^{l}=\operatorname{MLP}\left(\operatorname{LN}\left(\hat{\mathbf{z}}^{l}\right)\right)+\hat{\mathbf{z}}^{l} \\
&\hat{\mathbf{z}}^{l+1}=\operatorname{SW}-\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}^{l}\right)\right)+\mathbf{z}^{l} \\
&\mathbf{z}^{l+1}=\operatorname{MLP}\left(\operatorname{LN}\left(\hat{\mathbf{z}}^{l+1}\right)\right)+\hat{\mathbf{z}}^{l+1}
\end{aligned}
$$

### 3. 高质量图像重建模块 (high-quality image reconstruction)

对于图像超分任务，通过浅层特征 $F_{0}$ 和深层特征 $F_{DF}$ 重建高质量图像 $I_{RHQ}$:

$$
I_{R H Q}=H_{REC}\left(F_{0}+F_{DF}\right)
$$

式中，$H_{REC}(\cdot)$ 代表高质量图像重建模块。

浅层特征 $F_{0}$ 主要含有低频信息，而深层特征 $F_{DF}$ 专注于恢复丢失的高频信息。通过长距离的跳变连接，SwinIR 可以直接将低频道信息直接传输到重建模块，这可以帮助深层特征 $F_{DF}$ 专注于提取高频信息并稳定训练。使用 sub-pixel 的卷积层实现高质量图像重建模块。

对于图像去噪任务和压缩图像任务，仅仅使用一个带有残差的卷积操作作为高质量图像重建模块：

$$
I_{RHQ}=H_{SwinIR}\left(I_{L Q}\right)+I_{L Q}
$$

## 损失函数

对于超分任务，直接优化生成的高质量图片和 GT 的 $L_1$ 距离：

$$
\mathcal{L}=\left\|I_{R H Q}-I_{H Q}\right\|_{1}
$$

对于图像去噪任务和压缩图像任务，使用 Charbonnier Loss:

$$
\mathcal{L}=\sqrt{\left\|I_{R H Q}-I_{H Q}\right\|^{2}+\epsilon^{2}}
$$

式中，$\epsilon$ 通常取 $10^{-3}$ 。

## 模型参数：

- RSTB 模块数： K=6 (轻量级超分模型 K=4)。
- 每个 RSTB 模块的 STL 层数：L=6。
- Window size： M=8 (在图片压缩任务中 M=7)。
- Attention 的 head 数：6。
- channel 数：180 (轻量级超分模型为60)。