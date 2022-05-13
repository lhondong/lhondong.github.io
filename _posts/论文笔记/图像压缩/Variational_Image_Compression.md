---
title: "Variational Image Compression With A Scale Hyperprior"
subtitle: "变分自编码器图像压缩"
layout: post
author: "L Hondong"
header-img: "img/post-bg-45.jpg"
mathjax: true
tags:
  - 笔记
---

# Variational Image Compression With A Scale Hyperprior

[代码地址](https://github.com/tensorflow/compression)

Balle Google

## 摘要

引入了一个超先验网络，用来捕获**特征图之间的空域冗余问题**。这种优先的边信息的使用在传统编码中已经通用，但是在自编码器中还未得到探索。且该模型与之前的模型不同，超先验网络结构是与自编码器一起进行端到端的训练优化的。

边信息捕捉隐层表示的隐藏信息以辅助熵模型的参数生成，从而改善熵模型估计与隐层实际分布不匹配问题。将边信息引入比特流，这使得 decoder 也可以共享熵模型。解压时 decoder 先解压边信息，构建熵模型，之后基于正确的熵模型解压隐层信息。

## 一、简介

### 1.1 Motivation

在以往的图像压缩方法中，用于压缩潜在表示的熵模型通常表示为联合分布，甚至完全参数化的分布 $p_{\hat{y}}(\hat{y})$。注意，我们需要区分潜表示的的实际边缘分布 $m(\hat{y})$，和熵模型 $p_{\hat{y}}(\hat{y})$。

虽然通常假设熵模型具有参数化形式，并且参数适合数据，但是边缘是未知的分布，当熵模型估计潜表示出现值越精准，则其越接近理论熵值。编码器/解码器对使用它们的共享熵模型可以实现的最小平均码长由两个分布之间的香农交叉熵给出： 

$$
R = \mathbb{E}_{\hat{y}\sim m}[-\log_2 p_{\hat{y}}(\hat{y}))]
$$

其中 $m$ 为隐表示（latent representation）实际分布，$p_{\hat{y}}$ 为熵模型估计分布，熵模型是一个发送人与接收人共享的先验概率模型，用来估计真实隐层表示分布。上面的式子说明当熵模型估计 $p_{\hat{y}}$ 与实际分布 $m$ 完全相同时，编码长度最小，熵值达到最小。这告诉我们，一方面，当熵模型使用全参数化（fully factorized）概率分布的时候，如果实际分布中存在统计依赖，熵模型估计分布天然不能拟合实际分布；$y=\text{encoder}(x)$ 另一方面，因为是一个确定性的过程，如果想要在使用全参数化概率分布的情况下效果提升，则需要 encoder 对输入的自然图像尽量多的去除统计依赖。

在常规压缩方法中，该辅助信息的结构是手工设计的，而本文中介绍的模型本质上是学习熵模型的潜表示，就像基础压缩模型学习图像的表示一样。该模型是端到端的优化模型，所以通过**学习平衡信息量和预期的熵模型改进**，可以使总的预期代码长度最小。这是通过 VAE，概率生成模型和近似推理模型增强正式表达的方式来完成的。

Ballé 之前曾指出，一些基于自动编码器的压缩方法在形式上等同于 VAE，其中如上所述的熵模型对应于潜表示中的先验模型。在这里使用这种形式主义来表明，可以将边信息视为熵模型参数的先验信息，从而使它们成为潜在表示的先决条件。

具体来说，本文扩展了 Ballé（2017 年）等人提出的模型，有先验网络，并加入一个超先验的功能，即表示潜表示的空间相邻元素在其比例尺上倾向于一起变化。

### 2.2 数据流程

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-01-10-48-09.png" alt="Variational_Image_Compression-2022-05-01-10-48-09" style="zoom:50%;" /></div>

输入图片经过主编码器 $g_a$ 获得了输出 $y$，即通过编码器得到的潜在特征表示，经过量化器 $Q$ 后得到输出 $\hat{y}$​，超先验网络对 $\hat{y}$​ 进行尺度（$\sigma$ 捕获，对潜在表示**每一个点**进行**均值为 0，方差为 $\sigma$ 的高斯建模**，不同于以往的方法，以往的方式通过对整体潜在特征进行建模，即一个熵模型在推理阶段应用在所有的特征值熵，而超先验架构为每个特征点都进行了熵模型建模，通过对特征值的熵模型获取特征点的出现情况以用于码率估计和熵编码。

由于算术/范围编码器在编解码阶段**都需要**解码点的出现概率或者 CDF（累计概率分布）表。故而需要将这部分信息传输到解码端，以用于正确的熵解码。故而超先验网络对这部分信息先压缩成 $z$, 通过对 $z$ 进行量化熵编码传输至超先验解码端，通过超先验解码端解码学习潜在表示 $y$ 的建模参数。通过超先验网络获取得到潜在表示 $y$ 的建模分布后，通过对其建模并且对量化后的 $\hat{y}$​ 进行熵编码得到压缩后的码流文件，而通过熵解码得到 $\hat{y}$​，传统的编解码框架中往往设定反量化模块，而在解码网络中，包含了反量化的功能。故而在反量化模块在论文中并未部署，再将熵解码结果 $\hat{y}$​ 输入到主解码端，得到最终的重建图片 $\hat{x}$。对于整个网络的参数优化问题，依旧采用整体的率失真函数：$L=\lambda \cdot D +R$ 其中 $D$ 代表重建图像与原图的失真度， $R=R_{\hat{y}}+R_{\hat{z}}$ 代表整体框架的压缩码率。

上述量化会引入误差，这种误差在有损压缩的情况下是可以容忍的，同时该模型也遵循 Balle 在 2017 年论文中的量化方式，采用添加均匀噪声的形式近似量化操作，传统方式通过调整量化步长的形式进行率失真控制，而端到端的优化技术通过对码率和失真的权衡进行控制，如使用上述公式中的 λ \lambda λ 进行率失真折衷。

假设熵编码技术有效运行，则可以再次将其写为交叉熵：$R=E_{x\sim p_x}[-log_2p_{\hat{y}}(Q(g_a(x;\phi_g)))]$ 其中 $Q$ 表示量化函数，而 $p_{\hat{y}}$ 表示熵模型。在这种情况下，潜在表示的边际分布来自（未知）图像分布 $p_x$​，$Q$ 量化形式，以及分析变换的属性都会影响最终实际的编码码率值。

### 2.3 变分自编码器

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-01-10-54-18.png" alt="Variational_Image_Compression-2022-05-01-10-54-18" style="zoom:50%;" /></div>

优化问题可以表示为变分自动编码器，也就是说，图像的自编码的编码器和解码器类似于图像的概率生成模型与推理模型（图 2）。熵模型对应 VAE 隐层表示的先验 $p_{\tilde{y}}$。边信息可以看做是熵模型参数的先验，先验的先验这里称之为超先验。

在变分推论中，VAE 使用一个带参变分密度 $q(\tilde{y}\vert x)$ 来拟合真实后验概率 $p_{\tilde{y}\vert x}(\tilde{y}\vert x)$，通过最小化优化目标，隐变量真实分布与模拟分布的 KL 散度来达到拟合的效果（期望后验概率趋向于正太分布，并且保持变分自编码器的生成模式不退化为常规自编码器），这相当于最小化图像压缩中的率失真（rate-distortion）性能。

$$
\mathbb{E}_{x\sim p_x} D_{KL}[q\Vert p_{\tilde{y}\vert x}] = \mathbb{E}_{x\sim p_x}\mathbb{E}_{\tilde{y}\sim q}[\underbrace{\log q(\tilde{y}\vert x)}_0\underbrace{-\log p_{x\vert \tilde{y}}(x\vert \tilde{y})}_{weighted \ distortion}\underbrace{-\log p_{\tilde{y}}(\tilde{y})}_{rate}]+\text{const}
$$

通过将参数密度函数与变换编码框架进行匹配，可以了解到，KL 散度的最小化等效于针对率失真性能优化压缩模型。第一项近似为零，第二项和第三项分别对应于加权失真和比特率。

首先，“推理” 的机制是计算图像的分析变换并添加均匀噪声（作为量化的替代），因此：

$$
q(\tilde{y}|x,\phi_g)=\prod \limits_{i}U(\tilde{y_i}|y_i -\frac{1}{2},y_i+\frac{1}{2}), \text{ with } y=g_a(x;\phi_g)
$$

其中 $U$ 表示以 $y_i$ 为中心、宽度为 1 的均匀分布。由于均匀分布的宽度是恒定的（等于 1），$q$ 概率为 1，$\log q == 0$，因此 KL 散度中的第一项在技术上估计为零，并且可以从损失函数中删除（存疑）。

其中第二项对应了编码框架中的失真度，对于上述公式假定：

$$p_{x|\tilde{y}}(x|\tilde{y},\theta_g)=N(x|\tilde{x},(2\lambda)^{-1}1), \text{ with } \tilde{x}=g_s(\tilde{y};\theta_g)
$$

表示在参数神经网络中，$\tilde{x}$ 已经由 $\tilde{y}$ 从生成器中运行得到，并且通过率失真进行了 $\lambda$ 的权衡，即 KL 散度最小时，第二项去除负号使得对数似然模型最大化： $\log p_{x|\tilde{y}}(x|\tilde{x})$，等价于 $\tilde{y} $​ 通过解码器的重构图与原始图片的失真最小化，直接理解可以解释为：根据解码得到的像素点 $\tilde{x}$ 推测 $x$ 的后验概率，当两者像素值越接近的时候，其得到的后验概率越大，则 $-\log p_{x|\tilde{y}}(x|\tilde{x})$ 的计算值（即自信息）越小。因此等价于图像编码中的失真项。

第二项就是 $x$ 和 $\tilde{x}$ 的平方差，以 $\lambda$ 为权重。也就是说，如果以 $\tilde{y}$ 为条件的 $x$ 的分布满足如上条件的多维高斯分布，那么第二项可以看做图像压缩中的类 MSE 的失真项 distortion，最小化目标函数等同于缩小重构图像的失真。

第三项，$\mathbb{E}_{\tilde{y}\sim q}[-\log p_{\tilde{y}}(\tilde{y})]$，很容易看出与边缘分布 $m(\tilde{y})=\mathbb{E}_{x\sim p_x}q(\tilde{y}|x)$（经过合成变换、量化操作之后的隐层分布）和先验（熵模型分布）的交叉熵 [1] 相同，当边缘分布与先验相同时最小，即最小化目标函数等同于使熵模型分布更拟合边缘分布。反映了编码的代价，可以看做图像压缩中的率 rate。这里将先验建模成为一个无参数、可全分解的密度函数如下：

$$
p_{\tilde{y}|\psi}(\tilde{y}|\psi) = \prod_i(p_{y_i|\psi^{(i)}}(\psi^{(i)})*\boldsymbol{U}(-\frac{1}{2},\frac{1}{2}))(\tilde{y}_i)
$$

其中，$\psi^{(i)}$ 代表每一个单变量分布 $p_{y_i|\psi^{(i)}}$ 的所有参数。在图像压缩中，熵模型的分布只由参数决定。$*$ 代表卷积，目的是让先验 $p_{\tilde{y}}(\tilde{y})$ 能够更好地匹配边缘分布 $m(\tilde{y})$。称上面这种式子为可分解先验模型。$-\log p_{\tilde{y}}(\tilde{y})$ 在信息论中即表示 $\tilde{y}$ 的自信息，其对应了编码框架中的码率估计模型或者说由熵编码后待编码点的码率大小。  

把压缩问题中的分析模型 $g_a$ 看做 VAE 中的推理模型，把合成模型 $g_s$ 看做生成模型。在 VAE 的推理模型，目的是要估计真实的后验概率 $p_{\tilde{y}\vert x}(\tilde{y}\vert x)$，这通常不可行。

输入图片经过主编码器 $g_a$ 得到输出 $y$，即通过编码得到隐空间的特征表示，然后经过量化器 $Q$ 后得到输出 $\hat{y}$。另一方面，隐变量 $y$ 作为超先验网络 $h_a$ 的输入，对进行尺度 $\sigma$ 捕获，对潜在表示每一个点进行均值为 0，方差为 $\sigma$ 的高斯建模，不同于以往的方法对整体潜在特征进行建模，即一个熵模型在推理阶段应用在所有的特征值熵，而超先验网络为每个特征点都进行了熵模型建模，通过对特征值的熵模型获取特征点的出现情况以用于码率估计和熵编码。

由于算术/范围编码器在编解码阶段都需要解码点的出现概率或者 CDF（累计概率分布）表。故而需要将这部分信息传输到解码端，以用于正确的熵解码。故而超先验网络对这部分信息先压缩成 $z$, 通过对 $z$ 进行量化熵编码传输至超先验解码端，通过超先验解码端解码学习潜在表示 $y$ 的建模参数。通过超先验网络获取得到潜在表示 $y$ 的建模分布后，通过对其建模并且对量化后的 $\hat{y}$ 进行熵编码得到压缩后的码流文件，而通过熵解码得到 $\hat{y}$，传统的编解码框架中往往设定反量化模块，而在解码网络中，包含了反量化的功能。故而在反量化模块在论文中并未部署，再将熵解码结果 $\hat{y}$ 输入到主解码端，得到最终的重建图片 $\hat{x}$。

对于整个网络的参数优化问题，依旧采用整体的率失真函数：

$$
L = \lambda\cdot D + R 
$$

其中 $D$ 代表重建图像与原图的失真度， $R=R_{\hat{y}}+R_{\hat{z}}$ 代表整体框架的压缩码率。

上述量化会引入误差，这种误差在有损压缩的情况下是可以容忍的，同时该模型也遵循 Balle 在 2017 年论文中的量化方式，采用添加均匀噪声的形式近似量化操作，传统方式通过调整量化步长的形式进行率失真控制，而端到端的优化技术通过对码率和失真的权衡进行控制，如使用上述公式中的 $\lambda$ 进行率失真折衷。

假设熵编码技术有效运行，则可以再次将其写为交叉熵：

$$
R=E_{x\sim p_x}[-\log p_{\hat{y}}(Q(g_a(x;\phi_g)))]
$$

其中 $Q$ 表示量化函数，而 $p_{\hat{y}}$ 表示熵模型。在这种情况下，潜在表示的边际分布来自（未知）图像分布 $p_x$，$Q$ 量化形式，以及分析变换的属性都会影响最终实际的编码码率值。

### 2.4 思路介绍

为什么要引入这个 hierarchical priors 模块？原因在于，图像压缩其实是学习图像的信号分布过程。对于待压缩图像 $x$，我们不知道它的实际分布，且它的分布是存在统计依赖（概率耦合）的，因此需要剔除这部分冗余，实现最优压缩。

图中，2、3 子图均可隐约看出图像的大致轮廓，即一些高对比度，边缘纹理等特征 ($y, \sigma$)。图 4 则完全混淆了图像的“可视”特征，可认为是达到了比较好的信息压缩效果 ($y/\sigma$)。由此可以发现，实现理想的最优压缩，我们需要学习边信息，添加至熵编码模块中。它既可以实现更优的压缩，也可辅助图像解码恢复。

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-01-11-01-37.png" alt="Variational_Image_Compression-2022-05-01-11-01-37" style="zoom:50%;" /></div>

最左边是原图，中左是得到的潜在表示（选取熵率最大的那一张） $y$，中右表示通过超先验网络捕获到的尺度信息即（$\sigma$）的特征图，最右边则是归一化之后的，通过 $y/ \sigma$ 得到的，可以看出除以 $\sigma$ 之后，消除了空间结构冗余。

由分析模型得出的隐层表示在尺度上存在空间耦合性，如高对比度区域响应集中的高、存在边缘。只靠全分解的熵模型不能够捕获这些耦合情况，超先验用来捕获这些空间耦合性。

根据上图可以看出，对于编码器的结果 $y$ 存在结构冗余，房子的结构上依旧保留了下来，并且可以看出，捕获的 $\sigma$ 在结构上与未消除的结构冗余是一致的，对一组目标变量之间的依存关系进行建模的标准方法是引入隐含变量，条件是假定目标变量是独立的（Bishop，1999）。我们引入了一组额外的随机变量 $\sigma$ 来捕获空间相关性，假定通过 $\sigma$ 消除后的变量为独立标准正太分布模型。

超先验的方法，通过引入隐藏变量 $\tilde{z}$ 来建模空间相关性，扩展模型，将每个隐层变量 $\tilde{y}_i$ 建模成满足均值为 0 标准差为 $\tilde{\sigma}_i$ 的高斯分布，其中 $\tilde{\sigma}$ 由隐藏变量 $\tilde{z}$ 经过变换  $h_a$ 得来（同样卷积上一个标准均匀分布）： 

$$
p_{\tilde{y}|\tilde{z}}(\tilde{y}|\tilde{z},\theta_h) = \prod_i(\boldsymbol{N}(0, \tilde{\sigma}^2)*\boldsymbol{U}(-\frac{1}{2},\frac{1}{2}))(\tilde{y}_i), \quad \tilde{\sigma} = h_s(\tilde{z};\theta_h)
$$

对应的意思是 $\prod_i(N(0,\tilde{\sigma}_i^2)*U(-\frac{1}{2},\frac{1}{2}))$ 为 $\tilde{y_i}$ 的概率密度函数，其中 $p_{\tilde{y}|\tilde{z}}(\tilde{y}|\tilde{z},\theta_h)$ 表示某个 $y_i$​ 在被先验认知为正太分布，即确认了 $\tilde{z}$ 和超参数 $\theta$ 的情况下，得到的正太分布的方差值，在训练阶段的量化为均匀噪声，通过卷积形式松弛概率密度（概率密度函数图形更加平缓）。个人以为公式最后的 $(\tilde{y}_i)$ 是不是可以删除？

拓展推理模型，$y$ 之上加一个变换 $h_a$，得到一个联合可分解变分后验概率（single joint factorized variational posterior）如下：

$$
q(\tilde{y},\tilde{z}|x,\phi_g,\phi_h) = \prod_i \boldsymbol{U}(\tilde{y}_i|y_i-\frac{1}{2},y_i+\frac{1}{2})\cdot\prod_j \boldsymbol{U}(\tilde{z}_j|z_j-\frac{1}{2},z_j+\frac{1}{2}), \\ with \ y = g_a(x;\phi_g),z = h_a(y;\phi_h)
$$

由于没有 $\tilde{z}$ 的先验信息，所以使用之前建模 $\tilde{y}$ 的全分解密度模型建模 $\tilde{z}$（A.6.1）:

$$
p_{\tilde{z}|\psi}(\tilde{z}|\psi)=\prod_i(p_{z_i|\psi^{i}}(\psi^{(i)})*U(-\frac{1}{2},\frac{1}{2})(\tilde{z_i})
$$

其中 $\prod_i(p_{z_i|\psi^{i}}(\psi^{(i)})$ 这部分为 Factored Entropy model 通过离线学习得到的概率密度函数，通过对所有的点离线建模，统一认为所有的点服从上述的概率密度函数，同理 $U(-\frac{1}{2},\frac{1}{2})$ 的卷积运算表示量化过程的均匀噪声的添加，对概率密度具有松弛作用，不利于准确的码率估计，但是量化操作对于熵编码降低码率是必要的。**最后的 $(\tilde{z}_i)$ 也能删除吧？**

总损失函数如下所示，其中第三项和第四项分别代表编码、tilde{y}和、tilde{z}的交叉熵，第四项代表边信息： 

$$
\mathbb{E}_{x\sim p_x} D_{KL}[q\ ||\ p_{\tilde{y},\tilde{z}|x}] = \mathbb{E}_{x\sim p_x} \mathbb{E}_{\tilde{y},\tilde{z}\sim q}[{\log\ q(\tilde{y},\tilde{z}|x)} \underbrace{- \log\ p_{x|\tilde{y}}(x|\tilde{y})}_{distortion} \underbrace{- log\ p_{\tilde{y}|\tilde{z}}(\tilde{y}|\tilde{z}) - log\ p_{\tilde{z}}(\tilde{z})}_{rate} ] + const
$$

直观来看，$\tilde{z}$ 由 $\tilde{y}$ 经过分析变换得到，其规模进一步缩小，然后又通过合成变换扩大规模，得到熵模型的参数，其中可能会有两个位置的参数来源与同一个 $\tilde{z}_i$，这就达到了对两个元素之间耦合性建模的目的。

全文思路建立在编码器提取潜在特征的特征存在一定的结构冗余，这部分结构冗余即一些像素点存在相关性，本文的思路即如何消除这部分的相关性呢？

- 首先通过超先验网络对潜在特征点进行结构层次的信息捕获，通过超先验感知潜在点的结构信息，原始的方式是基于统计的方式，不是内容自适应，无法识别结构信息。
- 对潜在特征点进行自适应建模，由于算数编码的性质，在编码某一像素点，其概率值估计地越精准，则编码效率越高，而捕获这种结构实质的作用是通过 σ \sigma σ 参数告诉熵编码器，这个位置根据捕获到地结构，其概率在原来统计熵模型的情况下是没办法捕获这种结构通过对源模型进行高斯分布，通过方差参数捕获到的结构对概率进行调整。
- 这种熵模型是可以改动的，因为实际中真正的概率不一定符合高斯模型，目前最新的是高斯混合模型。其中

## 三、核心代码

### 3.1 整体流程

```
y = analysis_transform(x)  //编码器
z = hyper_analysis_transform(abs(y)) //超先验编码获得 z
z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)  //对 z 进行量化，以及码率估计
sigma = hyper_synthesis_transform(z_tilde) //解码 z，得到对应的 sigma 参数，即方差
scale_table = np.exp(np.linspace(
    np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)) //构建分组边界
conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table) //实例化高斯分布的熵率模型
y_tilde, y_likelihoods = conditional_bottleneck(y, training=True) //量化 y，以及估计 y 的码率
x_tilde = synthesis_transform(y_tilde) //根据量化后的值，重构图像
```

不赘述四个编码器的代码，就是卷积层以及 GDN 层的叠加，核心在于介绍 Gaussian Conditional 这个模块，该模块训练过程返回两个数值，量化以及每个待编码值的出现概率的估计。

### 3.2

```python
outputs = self._quantize(inputs, "noise" if training else "dequantize") //对输入进行量化
likelihood = self._likelihood(outputs) //对量化后的数值进行熵率估计。
  
'''训练过程进行添加（-0.5，0.5）的均匀噪声，推理过程则四舍五入。'''
def _quantize(self, inputs, mode):
    # Add noise or quantize (and optionally dequantize in one step).
    half = tf.constant(.5, dtype=self.dtype)
    _, _, _, input_slices = self._get_input_dims()

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half)
      return tf.math.add_n([inputs, noise])

    medians = self._medians[input_slices]
    outputs = tf.math.floor(inputs + (half - medians))

    if mode == "dequantize":
      outputs = tf.cast(outputs, self.dtype)
      return outputs + medians
    else:
      assert mode == "symbols", mode
      outputs = tf.cast(outputs, tf.int32)
      return outputs

def _likelihood(self, inputs):
    lower = self._logits_cumulative(inputs - half, stop_gradient=False) //获取分布函数的下界
    upper = self._logits_cumulative(inputs + half, stop_gradient=False) //获取分布函数的上界
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    sign = tf.stop_gradient(sign)
    likelihood = abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower)) //上界减去下界后即可得到估计的概率值。
    return likelihood
```

在假设 $y$ 服从均值为 0，方差为 $\sigma$ 的正太分布后，可以根据这一点对 logits_cumulative 进行重构，可以直接根据已知方差参数的正太概率密度函数进行上下界的分布函数求解得到出现的概率情况。替换上述 logits−cumulative 为下面的 standardized−cumulative。

```python
def _likelihood(self, inputs):
    values = inputs
    # This assumes that the standardized cumulative has the property
    # 1 - c(x) = c(-x), which means we can compute differences equivalently in
    # the left or right tail of the cumulative. The point is to only compute
    # differences in the left tail. This increases numerical stability: c(x) is
    # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
    # done with much higher precision than subtracting two numbers close to 1.
    values = abs(values)
    upper = self._standardized_cumulative((.5 - values) / self.scale)
    lower = self._standardized_cumulative((-.5 - values) / self.scale)
    likelihood = upper - lower
    return likelihood

  def _standardized_cumulative(self, inputs):
    half = tf.constant(.5, dtype=self.dtype)
    const = tf.constant(-(2 ** -0.5), dtype=self.dtype)
    # Using the complementary error function maximizes numerical precision.
    return half * tf.math.erfc(const * inputs)
```

## 四、实验

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-01-11-07-52.png" alt="Variational_Image_Compression-2022-05-01-11-07-52" style="zoom:50%;" /></div>

性能上相较于 BPG（HEVC 帧内编码）差了一点，但是相比于原始的 2017 年的论文的优化上，在 R-D 性能上，同码率下提高了一点几个 db, 可以说奠定了大部分图像压缩研究的基础。

## 补充

### A.6.1 单变量无参密度模型

借助累计分布函数 $c:\mathbb{R} \rightarrow [0,1]$ 定义密度模型 $p:\mathbb{R} \rightarrow \mathbb{R}^+$。其中累计分布函数满足：

$$
c(-\infty)=0;\ c(\infty)=1;\ p(x)=\frac{\partial{c(x)} }{\partial{x}} \ge 0
$$

累计分布函数应满足单调性，所以需要密度函数非负。假设累计分布函数可以分解为若干个函数：

$$
c = f_K\circ f_{K-1} \cdots f_1,\ with f_k:\mathbb{R}^{d_k} \rightarrow \mathbb{R}^{r_k}
$$

$$
p = f_K^{'} \cdot f_{K-1}^{'} \cdots f_1^{'}
$$

$f_K^{'}$ 是雅克比矩阵，矩阵形状为 $(r_k, d_k)$，为了保证 $p(x)$ 是一个单变量函数，即 $p$ 的形状为 (1,1)，需要 $r_K = d_1 = 1$（矩阵相乘从左向右）

为满足 $0\le p(x) \le 1$，首先需要雅克比矩阵非负，则选择 $f_K$ 如下：

$$
f_k(x) = g_k(H^{(k)}x + b^{(k)}), \quad 1 \le k \le K
$$

$$
f_K(x) = sigmoid(H^{(K)}x + b^{(K)})
\\ with\ nonlinearities: \quad g_k(x) = x+a^{(k)} \odot tanh(x)
$$

$\odot$ 代表逐元素相乘，$H^{(k)}$ 代表权重矩阵，$a^{(k)},b^{(k)}$ 代表偏置向量。将上面的式子求导如下：

$$
f_k^{'}(x) = diag g_k^{'}(H^{(k)}x + b^{(k)}) \cdot H^{(k)},\quad 1\le k \le K
$$

$$
g_k^{'}(x) = 1+a^{(k)} \odot tanh^{'}(x)
$$

$$
f_K^{'}(x) = sigmoid^{'}(H^{(K)}x + b^{(K)}) \cdot H^{(K)}
$$

为了限制导数非负，需要限制 $H^{(k)}$ 所有元素非负，$a^{(k)}$ 所有元素以 -1 为下界，通过重参数化操作实现（其中带 hat 的是真实参数）：

下面是使用该密度函数拟合一种高斯混合分布的情况（熵模型最终目的是要拟合真实分布）：

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-03-17-12-40.png" alt="Variational_Image_Compression-2022-05-03-17-12-40" style="zoom:50%;" /></div>

该模型 pytorch 版对应于 entropy_model 中的 bottleneck，可以看出累计分布函数形如一个多层感知机 MLP。在实现代码使用多种方法解决精度表示的数值问题。

其中累计分布函数部分代码如下：

```python
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)
 
            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias
 
            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits
```

### A.6.2 加上均匀噪声来建模先验

在正文中，用到了与均匀噪声卷积后的密度函数来建模先验（熵模型）$p_{\tilde{y}|\tilde{z}}$ 和超先验 $p_{\tilde{z}}$，以使先验更加灵活的拟合变分后验 $q$（实际分布）。假设变分后验和先验只有一维，此时，$g_a$ 总是为相应维度生成一个常数值：

$$
y=g_a(x)=c,\quad independent\ of\ x
$$

由于量化操作添加了均匀噪声，此时变分后验应该要精确匹配边缘分布：

$$
m(\tilde{y}) = q({\tilde{y}|x}) = \boldsymbol{U}(\tilde{y}|c-\frac{1}{2}, c+\frac{1}{2})
$$

交叉熵为：$\mathbb{E}_{\tilde{y}\sim m}[-log_2\ p_{\tilde{y}}]$，该交叉熵应该为 0，为了使交叉熵估计为 0，此时先验应该足够灵活地估计后验的形状——单位宽度均匀密度。

均匀分布不仅是高斯密度也是 A.6.1 中单变量无参密度函数的边界案例。为了解决这个问题，给先验卷上一个均匀分布：

$$
\begin{aligned}
p_{\tilde{y}}(\tilde{y}) &= (p*\boldsymbol{U}(-\frac{1}{2}, \frac{1}{2}))(\tilde{y}) \\ &= \int_{-\infty}^{\infty} p(y)\boldsymbol{U}(\tilde{y}-y|-\frac{1}{2}, \frac{1}{2})dy \\ &=\int_{\tilde{y}-\frac{1}{2}}^{\tilde{y}+\frac{1}{2}}p(y)dy \\ &=c(\tilde{y}+\frac{1}{2}) - c(\tilde{y}-\frac{1}{2})
\end{aligned}
$$

其中 $c$ 是累计分布函数。于是，先验的概率密度可以使用累计分布函数的差来表示。此时不论 p 是什么样的，当它的尺度参数 [2] （简单来说，尺度越大分布越分散，尺度越小分布越集中）趋向于 0 的时候，$p_{\tilde{y}}$ 趋向于一个单位均匀密度。如下所示，卷积上一个均匀分布使得先验概率对均匀分布的陡峭边界更加拟合：

<div align=center><img src="/assets/Variational_Image_Compression-2022-05-03-17-15-39.png" alt="Variational_Image_Compression-2022-05-03-17-15-39" style="zoom:50%;" /></div>