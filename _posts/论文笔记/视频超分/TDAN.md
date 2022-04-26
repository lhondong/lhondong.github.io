# TDAN: Temporally Deformable Alignment Network for Video Super-Resolution

CVPR 2020

首次将可变形卷积 (Deformable Convolution)v1 引入视频超分，用于解决 temporal alignment （时域对齐） 问题。

这篇文章 18 年底在 arxiv 上出现，大器晚成于 CVPR 2020。

## 背景

### 单图像超分 (Single Image Super-Resolution, SISR)

SRGAN 中的 SRResNet 网络对低分辨率图像做超分时，采取的是 PixelShuffle 操作，源于 Checkerboard artifact free sub-pixel convolution 论文提出的 sub-pixel convolution 操作：

<div align=center><img src="/assets/TDAN-2022-04-24-17-43-26.png" alt="TDAN-2022-04-24-17-43-26" style="zoom:50%;" /></div>

### 视频超分 (Video Super-Resolution, VSR)

TDAN 之前，不讨论**计算光流图+融合超分重构**这些两阶段工作外，重要的一个工作有 Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation 论文提出的 DUF network：

<div align=center><img src="/assets/DUF-2022-04-24-17-44-56.png" alt="DUF-2022-04-24-17-44-56" style="zoom:50%;" /></div>

<div align=center><img src="/assets/DUF-2022-04-24-17-46-12.png" alt="DUF-2022-04-24-17-46-12" style="zoom:50%;" /></div>

DUF network 将相邻帧的动作信息编码到生成的 Dynamic Upsampling Filters 中，然后作用于当前帧，优缺点有：

1. 共享网络部分，两个网络的作用完全不一样，为了降低计算量。但是用 **3D 卷积**而不是 2D 卷积来学习 spatio-tempral feature。（后续 TDAN、EDVR 用的都是 2D 卷积）；
2. 上采样超分利用了**相邻帧编码得到的动作信息**；
3. 残差生成网络做帧间了信息融合，但没有对相邻帧做 temporal alignment。

可以看出 DUF network 虽然当时是 SOTA，但相邻帧融合没有做对齐（TDAN 要提出的针对点）。

同时可看 VSR 的整体操作和 SISR 很像：SR 图像 = 多帧融合的残差 + LR 图上采样。

$$
I_{t}^{SR}=f_{\text {residual }}\left(\left[\left\{I_{i}^{LR}\right\}_{i \in[t-N, t+N]}\right]\right)+f_{\text {upscale }}\left(\left[\left\{I_{i}^{LR}\right\}_{i \in[t-N, t+N]}\right]\right)
$$

后续可以在这两个操作上单独改进做文章。

## TDAN 针对的问题及改进

视频相比单图像多了相邻帧的信息，同时要解决一个重要问题：temporal alignment 问题，也就是帧间对齐。当前帧和相邻帧的相同特征可能出现在不同的像素位置上，如果能够先对齐，然后融合特征，最后超分可以得到更加精确的图像（alignment → reconstruction 结构）。

Temporal Alignment 问题，在之前的算法中用估计的光流信息，严重依赖于光流估计算法性能：速度和准确率，效果也比较差。而 DUF network 也没有解决这个问题。

TDAN 提出了 temporal deformable alignment network 模块，利用可变形卷积 v1 组成的网络，自适应的给当前帧 (reference frame) 和相邻帧 ((supporting frames) 做对齐：动态估计像素/特征空间上的 offsets。最后，利用重构网络融合对齐后的视频帧，实现 one-stage 视频超分。

建议先了解可变形卷积：可变形卷积 v1，RefineDet，AlignDet，可变形卷积 v2。

## TDAN 算法细节

<div align=center><img src="/assets/TDAN-2022-04-25-00-01-33.png" alt="TDAN-2022-04-25-00-01-33" style="zoom:50%;" /></div>

如上图所示，分为两大部分：TDAN，超分重构网络。

$$
I_{i}^{LR'} =f_{TDAN}\left(I_{t}^{LR}, I_{i}^{LR}\right), i \in[t-N, t+N], i \neq t 
$$

$$
I_{t}^{HR} =f_{SR}\left(\left[I_{t-N}^{LR'}, \ldots, I_{t-1}^{LR'}, I_{t}^{LR}, I_{t+1}^{LR'}, \ldots, I_{t+N}^{LR'}\right]\right)
$$

其中， $I_{i}^{LR'}$ 表示第 $i$ 帧低分辨率图像， $I_{t}^{LR}$ 表示参考帧，也就是要超分的帧；$I_{i}^{LR}\in R^{sH\times sW\times C}$ 表示第 $i$ 帧 $\times s$ 倍超分图像。

整个模型中的卷积层和 ESRGAN 中去 BN 层一样。前置特征抽取和后置的超分重建（包含了帧间特征融合）中规中矩，但用可变形卷积作为帧间对齐。

### TDAN: Temporally Deformable Alignment Network

包含了 3 部分：特征抽取 (Feature Extractor)、变形对齐 (Deformable Alignment)、对齐帧重构 (Aligned Frame Reconstruction)。

$$
F_{i}^{L R}=f_{e x t}\left(I_{i}^{L R}\right), i \in[t-N, t+N]
$$

$$
F_{i}^{L R'}=f_{d c}\left(F_{i}^{L R}, \Theta_{i}\right), \Theta_{i}=f_{\theta}\left(\left[F_{t}^{L R}, F_{i}^{L R}\right]\right)
$$

$$
I_{i}^{L R'}=f_{r e c}\left(F_{i}^{L R'}\right)
$$

其中，特征抽取卷积网络 $f_{e x t}$ 针对所有输入图像执行卷积操作抽取特征，$[*]$ 表示将特征拼接，经卷积网络 $f_{\theta}$ 计算得到可变形卷积 offsets $\Theta_{i}$, 可变形卷积详细操作为针对每个特征图的每个特征点 $p_{0}$ 及对应的 offset $\Delta p_{n}$, 计算输出：

$$
F_{i}^{L R'}\left(p_{0}\right)=\sum_{p_{n} \in R} w\left(p_{n}\right) * F_{i}^{L R}\left(p_{0}+p_{n}+\Delta p_{n}\right)
$$

随后将对齐特征重构为对齐帧 $I_{i}^{L R'} \in R^{H \times W \times C}$ 。

这部分包含对齐重构损失：

$$
L_{align}=\frac{1}{2 N} \sum_{i=t-N, \neq t}^{t+N}\left\|I_{i}^{L R'}-I_{t}^{L R}\right\|_{1}
$$

本来应该是所有的相邻帧都向参考帧对齐，但是由于没有对齐后的 GT, 所以就让参考帧作为 GT 对 齐帧。本文发现隐性对齐很难训练，所以加入这个对齐重构损失显性监督训练。

### SR Reconstruction Network

包含了 3 部分：帧间融合 (Temporal Fusion)、非线性映射 (Nonlinear Mapping)、高分辨率帧重建 (HR Frame Reconstruction)。

其实就是个类似 SRResNet 之类的结构：卷积 + B *卷积块 + （卷积+PixelShuffle) * 2。
超分重构的损失函数为：

$$
L_{s r}=\Vert I_{t}^{H R'}-I_{t}^{H R}\Vert_{1}
$$

故整体网络的损失函数为：

$$
L=L_{\text {align}}+L_{s r}
$$

## 实验结果及分析

超过 SISR（RCAN）、one-stage VSR（DUF）、基于光流的 VSR（TOFlow）。

RCAN 超分结果优于 TOFlow 和 TDAN，本文认为因为参考帧包含的信息就很丰富，根本不需要相邻帧的信息，所以这些 very deep SISR network 表现优异。而视频超分网络（TDAN）很难训一个 very deep network。

同时，也尝试过直接拼接变形卷积出来的特征 $F_{i}^{LR'}, i \in[t-N, t+N],\neq t$ 和参考帧特征 $F_{t}^{LR}$，作为 SR 重构网络输入，而不用中间显式的对齐。但发现这样的效果非常差。所以对齐损失项 $L_{align}$ 很重要
