# Scale-Space Flow for End-to-End Optimized Video Compression

[CVPR 2020]

Google Research, Perception Team

Eirikur Agustsson, David Minnen, Nick Johnston, Johannes Ballé,

## 一、摘要

本文提出了一套全新的端到端视频编解码框架。针对现有基于学习的视频编解码需要光流、双线性warping和运动补偿，而且有相对复杂的架构和训练策略(需要预训练光流、训练各个子网络、训练过程中重建帧需要缓冲区)，本文提出一种广义warping操作，可以处理比如去遮挡、快速运动等复杂问题，而且模型和训练流程大大简化。

针对现有的基于学习的包含光流估计+运动补偿的框架总结四个问题：

1. 光流预测需要解决孔径问题（光流之所以是个病态问题的原因），这个问题比压缩问题更复杂；
2. 编解码框架中加入光流网络，给整个编解码框架增加了约束和复杂度；
3. 好的光流模型如果想要达到state-of-the-art表现，需要标注数据且训练复杂化。根据DVC的训练过程，在联合训练整个网络时，不需要单独的光流标注数据，所以作者总结的这个现有基于学习的视频编解框架的缺点个人认为有点牵强。
4. 稠密光流没有“no use”的概念，每个像素都要进行warped，导致无遮挡情况下会有较大残差。

针对上面四个现有框架缺点，作者提出改进措施，本文的贡献如下：

1. 提出尺度空间光流和warping，一种对光流+双线性warping的直观概述；
2. 简单的编解码框架和训练过程；
3. 实验结果显示达超过了基于训练的视频编解码的state-of-the-art结果，而且消融实验也表明了方法的有效性。

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

![](assets/20200614164241928.png)

尺度空间 warping，主要就是构造了一个尺度固定的 volume X:

X = [x, x∗G(σ0), x∗G(2σ0), · · · , x∗G(![](assets/gif.gif)σ0)]，其中，x ∗ G(σ) 表示 x 与尺度为σ的高斯核进行卷积。

X 是 x 具有不用模糊尺度的图片组成的堆栈，维度为（M x N x (M+1)），作者采用 M=5。

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