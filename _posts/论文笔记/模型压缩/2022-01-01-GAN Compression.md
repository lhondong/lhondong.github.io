---
title: "GAN Compression"
subtitle: "MIT 韩松老师组 GAN 压缩工作"
layout: post
author: "L Hondong"
header-img: "img/post-bg-7.jpg"
mathjax: true
tags:
  - 模型压缩
  - GAN
---

# GAN Compression

GAN Compression: Efficient Architectures for Interactive Conditional GANs

CVPR 2020

MIT HAN SONG, Adobe Research, Shanghai Jiao Tong University

## 摘要

强大的生成能力是以高强度计算为代价的，**cGANs 所需算力比图像识别领域的 CNNs 高出 1-2 个数量级**。比如，MobileNetv3 处理每张图片只需要 0.4G MACs，而 GauGAN 需要 281G MACs，相差数百倍。下图展示了几种算法的模型参数和算力的比较：

<div align=center><img src="/images/GAN Compression-2022-01-12-10-30-41.png" alt="GAN Compression-2022-01-12-10-30-41" style="zoom:30%;" /></div>

对于 cGANs 相关的图像生成应用来说，**需要在低延迟的设备上运行才能保证人机交互时的流畅性**。然而，类似于手机和平板的设备往往受限于内存和电池，这对于应用程序的普及是非常不利的。 

针对于 cGANs 模型压缩存在的问题，提出了有效的解决方案。

cGAN 模型压缩的难点：

- **GANs 训练难度大，尤其是在数据不成对的情况下；**
- GANs 的生成器和模式识别中的 CNNs 差别较大，难以借鉴其架构设计；

解决方法：

- 将原始教师生成器的中间表示的知识，迁移到对应压缩的学生生成器中；
- **针对 unpaired 训练方式，使用教师生成器的输出构造伪造样本对（pseudo pairs）；**
- 借助神经架构搜索技术（neutral architecture search，NAS），自动的寻找需要更低的计算成本和更少的参数数量的高效结构设计；

设计了一个包含所有可能 channel number 的 once-for-all 网络，将模型训练与架构搜索分离，以此来降低训练成本。通过权值共享的方式生成多个子网络，因此可以在不需要重新训练的情况下对每个子网络的性能进行评估。

GAN compression 方法可以应用于各种条件 GAN 模型，不受模型结构的限制。

## 二、相关工作

### 2.1 Conditional GANs

### 2.2 Knowledge distillation

## 三、方法

### 3.1 网络架构

左半部分是一个预训练的 Teacher 生成器 G'，通过 Distill 获得轻量级的“once-for-all” Student 生成器 G。

**右上部分是通过训练获得的多个子生成器 sub-generators，通过右下部分的选择评估获取最终的最优子生成器**。

<div align=center><img src="/images/GAN Compression-2022-01-12-10-32-27.png" alt="GAN Compression-2022-01-12-10-32-27" style="zoom:50%;" /></div>

#### 目标函数

Unifying unpaired and paired learning，统一成对的学习和不成对的学习。

使用 X 表示源域（source domain），Y 表示目标域（target domain），**cGANs 目的是通过网络训练，学习到映射函数 G，使得 G 能够将 X 映射到 Y**。

对于 cGANs 来说，可以采用不成对（unpaired）方式或成对（paired）方式进行训练。为了保证压缩算法的通用性，必须对训练方式进行统一。

使用教师生成器 G'的输出作为 ground-truth，可以和压缩的学生生成器 G 构成 paired 学习目标：

$$
\mathcal L_{recon}= \begin{cases} \mathbb E_{x,y}\Vert G(x)-y\Vert_1 & \text{if paired cGANs} \\\mathbb E_{x}\Vert G(x)-G'(x)\Vert_1 & \text{if unpaired cGANs}
  \end{cases}
$$

通过上述构建方式，将不成对学习转为成对学习，可以有效的提高算法的稳定性，并确保压缩算法的通用性。

#### Inheriting the teacher discriminator

为了保证模型稳定性，并尽可能的指导生成器 G 的训练，**可采用 teacher 的判别器 D'对 student 的判别器进行初始化，并与压缩生成器一起微调判别器**。随机初始化的判别器会导致严重的训练不稳定性和图像质量下降，使用教师预先训练的权重则好的多。

对抗性训练的目标函数：

$$
\mathcal L_{cGAN}=\mathbb E_{x,y}[\log D(x,y)]+\mathbb E_{x}[\log(1-D(x,G(x)))]
$$

#### Intermediate feature distillation

中间特征蒸馏，由于 cGANs 输出是确定性的图像，而不是概率分布。**所以不能使用输出层的逻辑分布，将知识从 teacher 模型迁移到 student 模型**。特别是对于成对训练，教师模型生成的输出图像与 GT 目标图像相比基本上不包含额外信息。实验表明，成对训练简单地模仿教师模型的输出没有任何好处。

为了解决上述问题，**可采用 teacher 模型的中间层表示进行匹配，将中间层包含的丰富信息迁移到 student**，使得学生模型在输出之外获取更多信息，蒸馏的目标函数如下：

$$
\mathcal L_{distill}=\sum\limits_{t=1}^{T}\Vert G_t(x)-f_t(G'_t(x))\Vert_2
$$

其中 $G_t(x)$ 表示 student 的中间层特征激活，$G'_t(x)$ 表示 teacher 的中间层特征激活，T 表示总的层数，t 表示当前层位置。$f_t$ 是一个 1 × 1 learnable 卷积将通道从学生模型映射到教师模型，因为他们通道数不同。

#### 总体目标

$$
\mathcal L=\mathcal L_{cGAN}+\lambda_{recon}\mathcal L_{recon}+\lambda_{distill}\mathcal L_{distill}
$$

### 3.2 Efficient Generator Design Space

选择一个高效的学生生成器网络结构对于知识蒸馏至关重要，单纯地缩小教师模型的通道数无法生成紧凑的学生模型：当计算量减少 4 倍以上时，性能开始显著下降。原因是现有的生成器架构采用了图像识别模型，这对于图像合成来说不是最佳的。

下面展示如何从现有的 cGAN 生成器中获得更好的架构设计空间，并在该空间内执行神经架构搜索（NAS）。

#### 卷积分解和卷积层的敏感性

Convolution decomposition and layer sensitivity

现有的生成器通常采用普通卷积（分类和分割 CNN），最近高效的 CNN 设计采用卷积分解 (depthwise + pointwise)，有更好的性能计算量权衡。

早期的实验表明，简单地将分解应用于所有卷积层（如在分类器中）将显著降低图像质量。虽然模型层可能会健壮一些，但分解某些层会立即影响性能。例如，在 ResNet generator（downsampling, resblock, upsampling），resBlock 层消耗了大部分模型参数和计算成本，而几乎不受分解的影响。相反，上采样层的参数少得多，但对模型压缩相当敏感，适度压缩可能导致 FID 大幅退化。因此，我们只分解 resBlock 层。（在第 4.3 节中对各层的敏感性进行了全面研究）

#### 使用 NAS 自动减少通道

现有的生成器在所有层中使用手工制作的（并且大部分是统一的）通道数量，会有冗余，不利于优化。为了进一步提高压缩比，我们使用自动通道剪枝来选择生成器中的通道宽度，以消除冗余，以平方形式减少计算。

关于通道数量进行细粒度选择，每个卷积层通道数量可以从 8 的倍数中选择，平衡了 MAC 和硬件并行性。

目标是使用 NAS 选出最佳通道：
$$
(c_1^*,c_x^*,\cdots,c_K^*)=arg\min_{c_1,c_2,\cdots,c_K} \mathcal L,s.t.MAC\leq F_t
$$

### 3.3 训练与搜索分离

Decouple Training and Search，遵循最近在 one-shot NAS 方法，首先训练一个“once-for-all”的网络，然后从 once-for-all 切割出对应的通道，并抽取出权重，获取子网络的输出值，计算梯度，然后更新抽取出来的权重。

Sub-networks 与 “once-for-all” network 共享权重，不同通道数的 sub-network 一起训练，在验证集上直接评估每个候选子网络的性能来找到最佳子网络。不同的子网络使用教师网络的前 $c_K$ 个通道，因为前面的通道更新的最频繁，所以他们也更重要。

由于“once-for-all”网络通过权重共享进行了全面训练，因此无需进行微调。

## 四、实验

### 4.1 测试模型

- CycleGAN：一个 unpaired 图像到图像转换模型，使用基于 ResNet 的生成器将图像从源域转换到目标域。
- Pix2pix：基于 cGAN 的成对图像到图像转换模型。我们用基于 ResNet 的生成器替换了原始的 U-Net 生成器，因为基于 ResNet 的生成器以较少的计算成本获得了更好的结果。
- GauGAN：SoTA paired 图像到图像转换模型，使用语义标签图生成高质量的图像。

### 4.2 数据集

horse——zebra；

edges——shoes；

cityscapes；

map——arial photo；

### 4.3 质量指标

- Fr ́ echet Inception Distance (FID)：计算从真实图像提取的特征向量分布与使用 InceptionV3 网络生成的特征向量分布之间的距离。

- Semantic Segmentation Metrics

下表展示了提出算法的压缩性能，**模型参数数量压缩的倍数在 1/5 到 1/33 之间，算力压缩在 1/4 到 1/22 之间**，将 CycleGAN 生成器的计算量减少 21.2 倍，远远优于基准算法的表现：

<div align=center><img src="/images/GAN Compression-2022-01-12-10-36-06.png" alt="GAN Compression-2022-01-12-10-36-06" style="zoom:50%;" /></div>
