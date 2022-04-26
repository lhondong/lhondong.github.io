# Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement

基于深度学习的层级式视频压缩方法

## 摘要

提出层级式视频压缩 (Hierarchical Learned Video Compression,HLVC)，包括双向深度预测 (Bi-Directional Deep Compression, BDDC) 和单运动深度压缩 (Single Motion Deep Compression,SMDC)，BDDC 主要用于压缩第二层视频帧，SMDC 采用单个运动向量预测多个帧，可以节省运动信息的码率。在解码端，利用加权递归质量增强 (Weighted Recurrent Quality Enhancement,WRQE) 网络，此网络将压缩帧和比特流作为输入。

将传统视频编码中的双向预测（bi-directional prediction）和层级式（hierarchical layer）编码方法引入深度学习的视频压缩框架，并在解码端针对层级式视频压缩的特点设计质量增强网络，有效提升视频压缩效率。本文方法优于现有的深度学习视频压缩方法，并且在 PSNR 和 MS-SSIM 的率失真性能上均超过 x265 LDP very fast 模式，在 MS-SSIM 率失真性能上超过 x265 Hierarchical B medium 和 x265 LDP medium 模式，在 PSNR 率失真性能上与 x265 LDP medium 模式相近。

框架结构、分层结构与 Video Compression through Image Interpolation 很相似，都是将一个 GOP 中的帧分为三层。

这种分层结构的优势：

1. 高质量的视频帧可以给其它帧提供更多的参考信息；
2. 由于相邻帧之间的高相关性，在解码器端，可以通过利用高质量帧中的有利信息来增强低质量帧。

第 1、2、3 层（layers 1, 2 and 3）中的帧分别以高、次高、低质量压缩。层级式的压缩质量（hierarchical quality）有以下两个优点：

1. 在编码端，高质量的帧可以为其他帧的压缩提供高质量的参考，从而提高其他帧的压缩效率；
2. 在解码端，由于视频帧之间的相关性，可以利用邻近的高质量帧中的有利信息增强低质量帧的质量。质量增强不带来额外码率，因此等同于提高编码效率。

第 3 和第 8 帧属于压缩质量和码率最低的第 3 层，其质量可利用第 1、2 层中的高质量帧（例如第 0 和第 5 帧）被递归质量增强网络显著增强，达到与第 2 层中的第 5 帧相当的质量，但消耗的码率远低于第 2 层，从而实现高效率的视频压缩。

<div align=center><img src="/assets/HLVC-2022-04-22-10-37-13.png" alt="HLVC-2022-04-22-10-37-13" style="zoom:50%;" /></div>

本文 HLVC 的结构框图如下图所示。

<div align=center><img src="/assets/HLVC-2022-04-22-10-38-35.png" alt="HLVC-2022-04-22-10-38-35" style="zoom:50%;" /></div>

### 第 1 层

第 1 层为 I 帧，使用图像编码方法压缩。

### 第 2 层 BDDC

本文提出双向深度压缩网络（Bi-Directional Deep Compression，BDDC），即使用前后两个第 1 层中的帧作为参考帧。

<div align=center><img src="/assets/HLVC-2022-04-22-10-46-54.png" alt="HLVC-2022-04-22-10-46-54" style="zoom:50%;" /></div>

#### Motion Compression (MC)

首先也要进行光流估计（采用 SpyNet 网络），然后两个光流一起送到光流压缩编码网络中，编码后的特征采用算术编码进行编码，再进行 round 量化。根据我们的经验，算术编码可以用基于深度学习的熵编码替换，效果可能会有提升。然后送到运动后处理网络，这其实就是 DVC 框架中运动补偿网络。最后进行残差压缩。

$$
\hat{q}_{m}=\operatorname{round}\left(E_{m}\left(\left[f_{5 \rightarrow 0}, f_{5 \rightarrow 10}\right]\right)\right) 
$$

其中 $\hat{f}_{5\rightarrow 0}$ and $\hat{f}_{5\rightarrow 10}$ 定义为 compressed motions。$\hat{q}_m$ 是算术编码。

$$
{\left[\hat{f}_{5 \rightarrow 0}, \hat{f}_{5 \rightarrow 10}\right]=D_{m}\left(\hat{q}_{m}\right)}
$$

#### Motion Postprocessing (MP)

$W_b$ as the backward warping operation

$$
x_{0 \rightarrow 5}^{C}=W_{b}\left(x_{0}^{C}, \hat{f}_{5 \rightarrow 0}\right), x_{10 \rightarrow 5}^{C}=W_{b}\left(x_{10}^{C}, \hat{f}_{5 \rightarrow 10}\right)
$$

$$
\tilde{x}_{5}=MP\left(\left[x_{0 \rightarrow 5}^{C}, x_{10 \rightarrow 5}^{C}, \hat{f}_{5 \rightarrow 0}, \hat{f}_{5 \rightarrow 10}\right]\right)
$$

$\tilde{x}_5$ denotes the compensated frame.

## Residual Compression (RC)

$$
\hat{q}_{r}=\operatorname{round}\left(E_{r}\left(x_{5}-\tilde{x}_{5}\right)\right)
$$

$$
x_{5}^{C}=D_{r}\left(\hat{q}_{r}\right)+\tilde{x}_{5}
$$

整体还是 DVC 的一整套方案，唯一不同的就是运动压缩和运动补偿网络是前后两帧和运动矢量一起喂给网络。MP 和 RC 网络还是 balle 的含有 GDN 的网络。

### 第 3 层 SMDC

本文提出单运动深度压缩网络（Single Motion Deep Compression，SMDC），该网络利用视频帧间运动的相关性，使用一个运动矢量图来预测多帧间的运动，从而达到减少编码运动矢量图消耗的码率。最后，本文设计加权递归质量增强网络（Weighted Recurrent Quality Enhancement，WRQE），其中的递归单元被质量特征加权来合理利用不用质量的多帧信息。BDDC、SMDC 和 WRQE 网络的详细结构，详见 [论文和补充材料](https://arxiv.org/abs/2003.01966)。

<div align=center><img src="/assets/HLVC-2022-04-22-10-52-31.png" alt="HLVC-2022-04-22-10-52-31" style="zoom:50%;" /></div>

如上图，$x_0\rightarrow x_2$ 的光流需要网络估计，$x_1$ 的相关光流就是在运动均匀假设的情况下进行估算：

$$
f_{\mathrm{inv}}(a+\Delta a(a, b), b+\Delta b(a, b))=-f(a, b)
$$

$$\hat{f}_{1 \rightarrow 0}=\operatorname{Inverse}(\underbrace{0.5 \times \underbrace{\text { Inverse }\left(\hat{f}_{2 \rightarrow 0}\right)}_{\hat{f}_{0 \rightarrow 2}}}_{\hat{f}_{0 \rightarrow 1}})
$$

$$
\hat{f}_{1 \rightarrow 2}=\operatorname{Inverse}(\underbrace{0.5 \times \hat{f}_{2 \rightarrow 0}}_{\hat{f}_{2 \rightarrow 1}})
$$

### 增强网络（加权递归增强）

<div align=center><img src="/assets/HLVC-2022-04-22-10-54-26.png" alt="HLVC-2022-04-22-10-54-26" style="zoom:50%;" /></div>

基于 QG-ConvLSTM 网络设计的网络结构。

### 训练策略

损失采用率失真函数，BDDC 和 SMDC 的损失如下：

$$
L = \lambda D +R
$$

$$
L_{\mathrm{BD}}=\lambda_{\mathrm{BD}} \cdot \underbrace{D\left(x_{5}, x_{5}^{C}\right)}_{\text {Distortion }}+\underbrace{R\left(\hat{q}_{m}\right)+R\left(\hat{q}_{r}\right)}_{\text {Total bit-rate }}
$$

$$
L_{\mathrm{SM}}=\lambda_{\mathrm{SM}} \cdot \underbrace{\left(D\left(x_{1}, x_{1}^{C}\right)+D\left(x_{2}, x_{2}^{C}\right)\right)}_{\text {Total distortion }} +\underbrace{R\left(\hat{q}_{m}\right)+R\left(\hat{q}_{r 1}\right)+R\left(\hat{q}_{r 2}\right)}_{\text {Total bit-rate }}
$$

对于 BDDC 和 SMDC 损失的距离函数，采用 MSE 训练，对于 MS-SSIM 需要优化利用 (1-MS-SSIM) 优化。

增强网络损失：

$$
L_{\mathrm{QE}}=\frac{1}{N} \sum_{i=1}^{N} D\left(x_{i}, x_{i}^{D}\right)
$$

## 实验

率失真曲线如下图所示，以 x265 LDP very fast 为基准的 Bjontegaard Delta Bit-Rate (BDBR) 数值如表所示。从中可见，本文提出的 HLVC 方法在 PSNR 和 MS-SSIM 的率失真性能上均优于现有的深度学习视频压缩方法，且好于 x265 LDP very fast 基准。HLVC 方法和 x265 LDP very fast 基准在各个测试视频上的码率和 PSNR、MS-SSIM 数值，请见 [课题主页](https://github.com/RenYang-home/HLVC)。

<div align=center><img src="/assets/HLVC-2022-04-22-10-40-56.png" alt="HLVC-2022-04-22-10-40-56" style="zoom:50%;" /></div>

<div align=center><img src="/assets/HLVC-2022-04-22-10-41-12.png" alt="HLVC-2022-04-22-10-41-12" style="zoom:50%;" /></div>

## 总结

感觉本文依然是没有大的突破，而且根据文献的描述，这个框架并不是很简单，第二层的复杂度和 DVC 相当，再加上第三层和后增强网络，这个框架还是很复杂的，尤其增强网络是 LSTM。另外，第三层网络利用一个光流预测多帧，直观上看是节省了一些光流，这个实验我们在自己的工作中也有尝试过，其实大多数情况下，光流所占的比特还是很小的，然而这种方法会使利用估算光流编码的帧的指标有所下降，这个方法对光流估算要求还是挺高。

本文利用高质量帧辅助编解码低质量帧，且进行后增强，也许会对编码过程中的错误传播抑制有用。