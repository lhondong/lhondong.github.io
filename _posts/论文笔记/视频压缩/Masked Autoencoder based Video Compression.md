# Masked Autoencoder based Video Compression

- [ ] Decoder 设计
- [ ] pre-train model
- [ ] 端到端 fine-tune
- [ ] PSNR，SSIM，LPIPS 横向对比

## 摘要

掩码 masked image model 已经取得了 outperform 性能。

Transformer 证明在图像领域也是非常好的性能，包括 Video Restoration。

传统的视频编解码框架存在问题，因此使用深度学习来进行视频压缩工作。

现有的端到端视频压缩主要是通过光流，但是现在的光流压缩效果不好。

将 Masked Autoencoder 用于视频压缩技术，它的压缩效果比之前的深度学习视频压缩框架都要好。在达到相同恢复质量的情况下，我们的工作相比于 HEVC 压缩效率好多少，相比于之前的工作好多少。

## Introduction

近年来，随着网络带宽的提高、智能终端和互联网音视频应用数量的迅猛增长，视频流量已经成为网络传输的主要流量，并且仍在移动网络中持续大幅增长。

持续增长的视频流量不仅对网络传输带来了巨大挑战，也带来了海量存储的要求。为了能够传输和存储海量的视频数据，视频压缩编码作为音视频应用的基础技术在快速发展，以 HEVC/H.265 为代表的新一代视频编码技术正在逐渐完善。

传统的编解码器依赖于经典的预测帧变换架构。基于学习的视频压缩算法以其强大的表征能力，不仅实现了更好的压缩效果，同时也比传统编解码压缩算法恢复质量更优。另外，传统的视频压缩工作需要手工针对帧内预测与帧间预测设计很多复杂模块，而端到端的深度学习模型能够自行学习，从而减少帧间冗余，实现更好的压缩效果。

Traditional video coding standards such as MPEG, AVC/H.264 [49], HEVC/H.265 [43], and VP9 [38] have achieved impressive performance on video compression tasks. However, as their primary applications are human perception driven, those hand-crafted codecs are likely suboptimal for machine-related tasks such as deep learning based video analytic.

rate-distortion trade-off

During recent years, a growing trend of employing deep neural networks (DNNs) for image compression tasks has been witnessed. Prior works [46, 7, 36] have provided theo- retical basis for application of deep autoencoders (AEs) on image codecs that attempt to optimize the rate-distortion trade-off, and they have showed the feasibility of latent representation as a format of compressed signal

深度学习图像压缩进展

### Contributions

1. 第一个将 ViT 用于端到端视频压缩。
2. 第一个将 Masked Image Model 用于视频压缩。
3. 设计了一个特别好的解码器，恢复效果非常好。
4. An end-to-end learned video compression codec based on GAN-prior generative image compression

A desired compression rate is controlled by the size of latent dimension in the image compression stage as well as the number of quantization levels used in residual encoding

设计不同的压缩率，不同掩码比例来控制

## Related Work

1. Deep Learning based Video Compression
2. Vision Transformer
3. Mask Autoencoder

#### 视频压缩主要方法

- 通过压缩单张视频图像以去除视频图像中的空间冗余

编码时，将待压缩的视频图像帧输入网络，然后通过卷积层逐步减少图像的特征图数目和空间尺度，将图像从像素空间映射到新的特征空间。然后利用量化、熵编码等方式去除特征空间内的统计冗余，通过生成对抗网络生成足够清晰的视频图像，最后输出码流得到解码结果。 

- 通过视频插帧等方法以去除视频帧间的时间冗余

基于多尺度卷积网络和对抗训练的视频插帧方法，用于去除视频中相邻图像帧之间的时间冗余。通过 GAN 的生成器得到插帧结果，然后利用 GAN 判别器判别插帧结果的准确性。采用多尺度结构更能捕捉物体的运动信息，而对抗训练能使插帧结果更符合人类的视觉系统。

#### 神经网络视频压缩方法

- 混合式神经网络视频编码 （即将神经网络代替传统编码模块嵌入到视频编码框架）
  - 预测编码
    - 帧内预测
    - 帧间预测
  - 熵编码
- 端到端视频编码（通过神经网络实现完整编码框架）
  - P 帧视频压缩
  - B 帧视频压缩

#### DVC

[CVPR 2019] DVC: An End-to-end Deep Video Compression Framework

传统的视频压缩方法采用预测编码结构，对相应的运动信息和残差信息进行编码。

端到端的神经网络视频压缩系统利用传统视频压缩方法的经典结构和神经网络强大的非线性表示能力，同时优化了视频压缩框架的各个组成部分：利用基于学习的光流估计来获取运动信息并重构当前帧。然后采用两种自动编码方式的神经网络对相应的运动信息和残差信息进行压缩。
所有模块通过一个共同的损失函数进行学习，通过在减少码率和提高解码视频质量之间的权衡来相互协作。实验结果表明，该方法在峰值信噪比方面优于目前广泛使用的视频编码标准 H.264，在 MS-SSIM 方面与最新标准 H.265 相当。

#### FVC

[CVPR 2021] A New Framework towards Deep Video Compression in Feature Space

提出了一个特征空间视频编码网络 (FVC)，在特征空间中执行所有主要操作（即运动估计、运动压缩、运动补偿和残余压缩）。

在所提出的可变形补偿模块中，首先在特征空间中通过运动估计来产生运动信息，并使用自动编码器式网络进行压缩。然后利用可变形卷积进行运动补偿，并生成预测特征。然后压缩当前帧的特征和可变形补偿模块的预测特征之间的残差特征。

为了更好地进行帧重建，还在多帧特征融合模块中使用非局部注意机制，融合了之前多个重构帧的参考特征。综合的实验结果表明，该框架在 HEVC、UVG、VTL 和 MCL-JCV 等四个基准数据集上达到了最先进的性能。

#### M-LVC

[CVPR 2020] M-LVC: Multiple Frames Prediction for Learned Video Compression

以往的方法都局限于以前一帧作为参考，本文引入前面的多个帧作为参考。

在多个参考帧和多个 MV 的情况下，本文设计的网络可以对当前帧产生更精确的预测，产生更少的残差。多参考帧也有助于生成 MV 预测，从而降低 MV 的编码成本。使用两个深度自动编码器分别压缩残差和 MV。为了补偿自动编码器的压缩误差，同时利用多个参考帧，进一步设计了一个 MV 优化网络和一个残差优化网络。

### CVPR 图像视频压缩比赛：CLIC

CLIC 挑战赛是计算机视觉会议上首次明确关注图像视频压缩的比赛。计算机视觉会议上讨论的许多技术都与有损压缩有关。例如，超分辨率和伪影消除可以被视为有损压缩问题的特例，其中的编码器是固定的，只训练解码器。同时，修复、着色、光流、生成对抗网络和其他概率模型也被用作有损压缩的一部分。因此，CVPR 中的大部分工作可能对有损压缩感兴趣。

机器学习与深度学习的最新进展使人们对将神经网络应用于压缩问题的兴趣与日俱增。例如，在 CVPR 2017 上，一个口头报告展示了使用循环卷积网络进行压缩。在最近的 CVPR 中，提出了多种有损和无损压缩方法。为了促进图片与视频压缩这一领域的进一步发展，CLIC 挑战赛不仅鼓励更多的压缩解决方案，而且还建立了基线并提出了评估基准和方案。

CLIC 有损图像和视频压缩的挑战赛，专门针对传统上被忽视的方法，重点放在神经网络。这种方法通常包括一个编码器，用来获取图像/视频并产生比像素更容易压缩的表征（例如，它可以是卷积堆栈，产生整数特征图），然后使用整概率模型来生成压缩比特流。压缩比特流构成了要存储或传输的文件。在解压过程，通过解码器（与编码器有一个共享的概率模型）生成原始图像/视频的重建。

图像与视频压缩算法是一项有趣的壮举，特别是在统一现实基准的基础上，它的结果与其他类似的算法相比更好，就具有了更大的意思。

#### MV-residual prediction

Wu, X., Zhang, Z., et al. . End-to-end optimized video compression with MV-residual prediction. CVPR Workshops.

端到端的 P 帧压缩框架，设计了一个基于运动矢量和残差预测的网络 MV-residual。模型将两个连续帧作为输入，提取运动矢量表示和残差信息的融合特征。用超先验自动编码器建立了线性表示的先验概率模型，并与 MV 残差网络同时训练。

另外，该压缩框架将空间位移卷积应用于视频帧预测，其中通过在源图像中的位移位置应用卷积核来学习每个像素的运动核以生成预测像素。

最后，采用新的速率分配和后处理策略来产生最终的压缩码率。在验证集上的实验结果表明，所提出的优化框架的质量 MS-SSIM 最好。

#### 过拟合

Gang He, Chang Wu, Lei Li, et al. . A Video Compression Framework Using an Overfitted Restoration Neural Network. CVPR Workshops.（西交大，时间最短）

现有的许多基于深度学习的视频压缩方法都是利用深度神经网络（DNNs）通过学习解码视频和原始视频之间的映射关系来增强解码后的视频。这种方法最大的挑战是训练一个适用于不同视频序列的模型（一个映射），我们将模型与视频压缩一起训练，并对每个序列甚至每帧使用一个模型。

主要思想是利用过拟合恢复神经网络（ORNN）建立一个视频压缩框架（VCOR）。针对一组连续的帧训练一个轻量级的 ORNN，使其对这组帧进行过度拟合，从而获得很强的恢复能力。之后，ORNN 的参数作为编码比特流的一部分被发送到解码器。在去编码端，ORNN 可以对重构帧执行同样的强恢复操作。

创新性地将深度学习“过拟合”方式应用于视频压缩，并且设计出鲁棒的码率控制算法大幅提升压缩效率，在竞赛中主观评价指标 MS-SSIM 的结果在所有参赛队伍中处于前列（与第一名 TUCODEC_SSIM 仅差 0.00025）。

此外，该方案在解码器的轻量化及解码速度上取得了重大突破，在与前三名 MS-SSIM 相近的情况下，该方案的解码器大小在所有队伍中最小，且该方案的解码速度均快于前三名。

## Method

### Encoder

借鉴了 MAE 中的编码器，使用 Swin Transformer。

### Decoder

为了更好的恢复效果，SwinIR 超分，Video Restortion Transformer 视频恢复。

设计更好的 Decoder。

### Loss Fuction

$$
\mathcal L = 码流 + \lambda 质量
$$

## Experiments

### Datasets

Kinetics dataset [10] and the UGC dataset

- HEVC 数据集，16 个视频， Class B, C, D, E 不同的分辨率，从 416 × 240 到 1920 × 1080
  - Class A:从超清视频序列"Traffic" (4096x2048p 30 fps), "PeopleOnStreet" (3840x2160p30 fps).中截取的2560x1600的序列
  - Class B:1920x1080p 24 fps: "ParkScene","Kimono"，1920x1080p 50-60 fps: "Cactus", "BasketballDrive","BQTerrace
  - Class C:832x480p 30-60 fps (WVGA):"BasketballDrill", "BQMall", "PartyScene","RaceHorses"
  - Class D:416x240p 30-60 fps (WQVGA):"BasketballPass", "BQSquare", "BlowingBubbles","RaceHorses"
  - Class E:1280x720p 60fps video conferencing scenes:"Vidyo1", "Vidyo3" and "Vidyo4"
- UVG 数据集包括七个高帧率的，1920 × 1080 分辨率的视频，相邻帧之间的差异很小 
- MCL-JCV 数据集包括 30 个 1080p 视频序列，用于视频质量评估
- VTL 数据集使用前 300 帧的高分辨率视频序列，分辨率为 352 × 288
- JCT-VC

### Setting

优化器 Adam，学习率 $\alpha$。

### Conclusion
