# Scale-Space Flow for End-to-End Optimized Video Compression

[CVPR 2020]

Google Research, Perception Team

Eirikur Agustsson, David Minnen, Nick Johnston, Johannes Ballé,

## 摘要

尽管在面向图像压缩的端到端深度网络优化方面取得了长足的进步，视频编码仍然是一项具有挑战性的任务。最近提出的学习视频压缩的方法使用光流和双线性变形的运动补偿和显示竞争率失真的性能相对于手工编码，如 H.264和 HEVC。然而，这些基于学习的方法依赖于复杂的结构和训练方案，包括使用预先训练的光流网络，子网络的顺序训练，自适应速率控制，以及在训练过程中缓冲中间重构到磁盘。

本文提出了一套全新的端到端视频编解码框架。针对现有基于学习的视频编解码需要光流、双线性warping和运动补偿，而且有相对复杂的架构和训练策略(需要预训练光流、训练各个子网络、训练过程中重建帧需要缓冲区)，本文提出一种广义warping操作，可以处理比如去遮挡、快速运动等复杂问题，而且模型和训练流程大大简化。

提出尺度空间流，光流的直观概括，增加了尺度参数，使网络更好的模型不确定性。实验表明，使用尺度空间流进行运动补偿的低延迟视频压缩模型(没有 B 帧)可以胜过类似状态的学习视频压缩模型，同时使用一个简单得多的过程进行训练。

## 一、简介

### 1.1 Motivation

过去的深度学习视频压缩主要分为三种：

1. 3D Autoencoder
2. 插帧方法
3. 基于光流的运动补偿

针对现有的基于学习的包含光流估计+运动补偿的框架总结四个问题：

1. 光流预测需要解决孔径问题（光流之所以是个病态问题的原因），这个问题比压缩问题更复杂；
2. 编解码框架中加入光流网络，给整个编解码框架增加了约束和复杂度；
3. 好的光流模型如果想要达到state-of-the-art表现，需要标注数据且训练复杂化。（根据DVC的训练过程，在联合训练整个网络时，不需要单独的光流标注数据，所以作者总结的这个现有基于学习的视频编解框架的缺点个人认为有点牵强。）
4. 稠密光流没有“no use”的概念，每个像素都要进行warped，导致无遮挡情况下会有较大无意义的残差。

与先前的基于流的运动补偿方法相比，本文的系统简单得多，因为不需要单独估计光流或使用预训练的网络。也不需要使用高级的训练或编码策略，例如缓冲重建[24]或空间自适应速率控制。

消融研究表明，相比双线性扭曲，提出的尺度空间扭曲显着改善率失真性能，在某些比特率下增益超过1dB。

### 1.2 Contributions

1. 提出尺度空间光流和warping，一种对光流+双线性warping的直观概述；
2. 简单的编解码框架和训练过程，可以轻易地进行端到端的训练；
3. 实验结果显示达超过了基于训练的视频编解码的state-of-the-art结果，而且消融实验也表明了方法的有效性。

## 二、相关工作

### 2.1 Scale-space for flow estimation

尺度空间技术在光流估计方面有着悠久的历史，包括经典技术(如[6,18,13])以及在深度光流估计网络中使用的多尺度金字塔技术[28,14]。然而，这些工作仅仅利用尺度空间进行光流估计，而最终的结果仍然是一个标准的二维 displacement 场。相比之下，我们估计的三维尺度空间流直接集成到我们提出的尺度空间扭曲操作(见图1)，不管使用的是尺度空间还是多尺度金字塔来做估计。

### 2.2 Uncertainty estimates for optical flow

提出的尺度-空间流的尺度参数(见图1)可以解释为一个“不确定性参数”，因为在很难通过扭曲获得良好预测的区域使用大尺度值是自然的。

之前的有监督光流研究如何将不确定性整合到光流估计网络的预测中(见[20]概述) ，这些方法在有监督环境中运行: 即它们预测真实光流预测的不确定性。相比之下，本文的重点是泛化光流 + 扭曲操作，使扭曲结果形成一个良好的预测，而不考虑位移场和真实光流流之间的关系。

##  Method

### 3.1 尺度空间光流

重点就在于构造光流时引入了scale filed。

一般的warping：

$$
\begin{aligned}
&\mathbf{x}^{\prime}:=\text { Bilinear-Warp }(\mathbf{x}, \mathbf{f}) \quad \text { s.t. } \\
&\mathbf{x}^{\prime}[x, y]=\mathbf{x}\left[x+\mathbf{f}_{x}[x, y], y+\mathbf{f}_{y}[x, y]\right]
\end{aligned}
$$

包含尺度空间的 warping：

$$
\begin{aligned}
&\mathbf{x}^{\prime}:=\text { Scale-Space-Warp }(\mathbf{x}, \mathbf{g}) \quad \text { s.t. } \\
&\mathbf{x}^{\prime}[x, y]=\mathbf{X}\left[x+\mathbf{g}_{x}[x, y], y+\mathbf{g}_{y}[x, y], \mathbf{g}_{z}[x, y]\right]
\end{aligned}
$$

尺度空间 warping，主要就是构造了一个尺度固定的 volume X:

$X = [\mathbf{x}, \mathbf{x}∗G(\sigma_0), \mathbf{x}∗G(2\sigma_0), · · · , \mathbf{x}∗G(2^{M-1}\sigma_0)]$，其中，$\mathbf{x} ∗ G(\sigma)$ 表示 $\mathbf{x}$ 与尺度为 $\sigma$ 的高斯核进行卷积。

X 是 x 具有逐渐模糊尺度的图片组成的堆，维度为（M x N x (M+1)），作者采用 M=5。

继续定义尺度空间流为一个

![](assets/gif-20220326233500934.gif)=0 时，尺度空间的 warping 就退化为双线性 warping：

![](assets/20200614165746711.png)

当![](assets/gif-20220326233500675.gif)和![](assets/gif-20220326233504069.gif)为零，![](assets/\sigma _{0}) }.gif)时，尺度空间的 warping 就近似于高斯模糊：

![](assets/20200614170734767.png)

X 可以通过简单的三线性插值获取，而坐标 z 这样计算：

![](assets/20200614171333852.png)

其中，0 < σ <![](assets/gif.gif)σ0

2、压缩模型

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70.png)

主帧压缩、残差压缩、尺度空间光流计算均采用 balle 那一套框架。

3、量化和熵编码

没有创新的地方，可见参考文献 [10] 和[32]。

4、损失函数

![](assets/20200614172451298.png)

N 帧的率失真相加作为损失。这个在 [https://blog.csdn.net/cs_softwore/article/details/106301972](https://blog.csdn.net/cs_softwore/article/details/106301972) 论文解析中曾经提到过，这个方法有利于减少错误传播。

### 四、实验

1、网络结构。

上文已经提到了，就是 balle 的含有 GDN 的图像压缩压缩框架，三个网络都是。

2、训练相关

训练数据和一些训练参数可以参见文献，没有特殊地方。这里要提一下训练策略，先用 256 x 256 的图片训练，再用分辨率较大的 384 x 384 图片进行微调，这和我们复现 DVC 时的训练策略一致。

N(Number of unrolled frames) 采用 9 或 12 取得较好结果，在 Nvidia V100 GPU 上训练了 30 天，基于学习的视频编解码训练确实消耗时间。

3、实验结果

（1）一些定性结果：

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70-20220326233457554.png)

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70-20220326233458640.png)

（2）一些定量结果

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70-20220326233456890.png)

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70-20220326233458056.png)

也可以看出尺度空间的 warping 比普通插值 warping 具有更佳的表现。

（3）视频级的分析

![](assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzX3NvZnR3b3Jl,size_16,color_FFFFFF,t_70-20220326233457579.png)

### 五、结论与总结

提出尺度空间光流和尺度空间 warping，而且简化了编解码流程，与 DVC 相比，是完全不同的框架。为基于深度学习的端到到视频编解码框架提供了一个新的思路。

但从这个文献结果来看，依然是只比人工设计的传统的混合编解码框架 H265 略好，像本文这么简化的框架能达到如此效果已经很令人惊喜，但是相比 H266 算法（比 H265 提升 30%+）还是要差很多，基于学习的视频编解码还是有很长的路要走。

$$
\text{Scale-Space-Warp} \left(\mathbf{x},\left(\mathbf{g}_{x}, \mathbf{g}_{y}, \mathbf{0}\right)\right)=\operatorname{Bilinear-Warp}\left(\mathbf{x},\left(\mathbf{g}_{x}, \mathbf{g}_{y}\right)\right)
$$

$$
\text{Scale-Space-Warp} \left(\mathbf{x},\left(0,0,1+\log _{2}\left(\sigma / \sigma_{0}\right)\right) \approx \mathbf{x} * G(\sigma)\right. \\
\text{where equality holds if } \log _{2}\left(\sigma / \sigma_{0}\right) \in\{0, \cdot, M-1\}.
$$

$$
\sum_{i=0}^{N-1} d\left(\mathbf{x}_{i}, \hat{\mathbf{x}}_{i}\right)+\lambda\left[H\left(\mathbf{z}_{0}\right)+\sum_{i=1}^{N-1} H\left(\mathbf{v}_{i}\right)+H\left(\mathbf{w}_{i}\right)\right]
$$