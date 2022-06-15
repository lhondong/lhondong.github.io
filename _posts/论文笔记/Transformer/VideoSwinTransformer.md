# Video Swin Transformer

## 一、概述

Vision Transformer 是 Transformer 应用到图像领域的一个里程碑，它将 CNN 完全剔除，只使用了 Transformer 来完成网络的搭建，并且在图像分类任务中取得了 SoTA 的效果。

Swin Transformer 则更进一步，引入了一些 inductive biases，将 CNN 的结构和 Transformer 结合在了一起，使得 Transformer 在图像全领域都取得了 SoTA 的效果。Swin Transformer 中也有用到 CNN，但是并不是把 CNN 当做 CNN 来用的，只是用 CNN 的模块来写代码比较方便。所以，也可以认为是完全没有使用 CNN。

本文是 Swin Transformer 在视频领域的应用，也就是 Video Swin Transformer，只是多了一个时间的维度，做 attention 和构建 Window 的时候略有区别。

## 二、模型介绍

### 2.1 整体架构

#### 2.1.1 backbone

Video Swin Transformer 的 backbone 的整体架构和 Swin Transformer 大同小异，多了一个时间维度 $T$，在做 Patch Partition 的时候会有个时间维度的 patch size。  

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-26-18.png" alt="VideoSwinTransformer-2022-04-22-16-26-18" style="zoom:50%;" /></div>

输入为一个尺寸为 $T \times H \times W \times 3$ 的视频，通常还会有个 batch size，这里省略掉了。 $T$ 一般设置为 32，表示从视频的所有帧中采样得到 32 帧，采样的方法可以自行选择，不同任务可能会有不同的采样方法，一般为等间隔采样。这里其实也就天然限制了模型不能处理和训练数据时长相差太多的视频。通常视频分类任务的视频会在 10s 左右，太长的视频也很难分到一个类别里。

输入经过 Patch Partition 之后会变成一个 $\frac{T}{2} \times \frac{H}{4} \times \frac{W}{4} \times 96$ 的向量。这是因为 patch size 在这里为 $(2,4,4)$，分别是时间，高度和宽度三个维度的尺寸，其中 96 是因为 2×4×4×3=96，也就是一个 patch 内的所有像素点的 RGB 三个通道的值。Patch Partition 会在 2.2 中详述。

Patch Partiion 之后会紧跟一个 Linear Embedding，这两个模块在代码中是写在一起的，可以参见 [PatchEmbed3D](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py#L416)，就是直接用一个 3D 的卷积，用这个卷积充当全连接。如果 Embedding 的 dim 为 96，那么经过 Embedding 之后的尺寸还是 2×4×4×3=96。

之后分别会经过多个 Video Swin Transformer block 和 patch merging。Video Swin Transformer 是利用 attention 同一个 Window 内的特征进行特征融合的模块；patch merging 则是用来改变特征的 shape，可以当作 CNN 模型当中的 pooling，不过规则不同，而且 patch merging 还会改变特征的 dim，也就是 C 改变。整个过程模仿了 CNN 模块中的下采样过程，这也是为了让模型可以针对不同尺度生成特征。浅层可以看到小物体，深层则着重关注大物体。

Video Swin Transformer block 的结构如下图 2-2 所示。

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-29-01.png" alt="VideoSwinTransformer-2022-04-22-16-29-01" style="zoom:50%;" /></div>

图中左右是两个不同的 blocks，需要连在一起搭配使用。在图 1 中的 Video Swin Tranformer block 下方有 ×2 或是 ×6 这样的符号，表示有几个 blocks，这必定是个偶数，比如 ×2 就表示这样 1 组 blocks，×6 就表示这样 3 组 blocks 相连。

不难看出，有两种 blocks，每个 block 都是先过一个 LN 层 ([LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html))，再过一个 MSA(multi-head self-attention)，再过一个 LN，最后过一个 MLP，其中有两处使用了残差模块。残差块主要是为了缓解梯度消失。

两种 blocks 的区别在于前者的 MSA 是 Window MSA，后者是 Shifted-Window MSA。前者是为了 window 内的信息交流（局部），后者是为了 window 间的信息交流（全局）。

#### 2.1.2 head

backbone 的作用是提取视频的特征，真正来做分类的还是接在 backbone 后面的 head，这个部分就很简单了，就是一层全连接，代码中使用的是 [I3DHead](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/heads/i3d_head.py#L9)。顺便还带了 [AdaptiveAvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html)，这是用来将输入变成适合全连接的 shape 的。这部分比较简单。

### 2.2 模块详述

#### 2.2.1 Patch Partition

图 3 是一段视频中的 8 帧，每帧都被分成了 8×8=64 个网格，假设每个网格的像素为 4×4，那么当 patch size 为 (1,4,4) 时，每个小网格就是一个 patch；当 patch size 为 (2,4,4) 时，每相邻两帧的同一个位置的网格组成一个 patch。这里和 Vision tranformer 中的划分方式相同，只不过多了时间的概念。

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-32-35.png" alt="VideoSwinTransformer-2022-04-22-16-32-35" style="zoom:50%;" /></div>

#### 2.2.2 3D Patch Merging

3D Patch Merging 这一块直接看代码会比较好理解，它和 Swin Transformer 中的 2D patch merging 一模一样，3D Patch Merging 虽然多了时间维度，但是并没有对时间维度做 merging 的操作，也就是输出的时间维度不变。

```python
x0 = x[:, :, 0::2, 0::2, :]  # B T H/2 W/2 C
x1 = x[:, :, 1::2, 0::2, :]  # B T H/2 W/2 C
x2 = x[:, :, 0::2, 1::2, :]  # B T H/2 W/2 C
x3 = x[:, :, 1::2, 1::2, :]  # B T H/2 W/2 C
x = torch.cat([x0, x1, x2, x3], -1)  # B T H/2 W/2 4*C
```

看代码再结合图就更好理解了。图中每个颜色都是一个 patch。  

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-34-16.png" alt="VideoSwinTransformer-2022-04-22-16-34-16" style="zoom:50%;" /></div>

#### 2.2.3 W-MSA

W-MSA(window based MSA) 相比于 MSA 多了一个 Window 的概念，相比于 Vision Transformer 引入 Window 的目的是减小计算复杂度，使得复杂度和输入图片的尺寸成线性关系。这里不推导复杂度的计算，3D 和 2D 的复杂度计算方法是一致的。  

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-32-35.png" alt="VideoSwinTransformer-2022-04-22-16-32-35" style="zoom:50%;" /></div>

窗口的划分方式如图所示，每个窗口的大小由 window size 决定。图中的 window size 为 (4,4,4) 就表示在时间，高度和宽度的 window 尺寸都是 4 个 patch，划分后的结果如图 3 中间所示。之后的 attention 每个 window 单独做，window 之间不互相干扰。

#### 2.2.4 SW-MSA

由于 W-MSA 的 attention 是局部的，作者就提出了 SW-MSA(shifted window based MSA)。

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-22-16-32-35.png" alt="VideoSwinTransformer-2022-04-22-16-32-35" style="zoom:50%;" /></div>

SW-MSA 如图 3 最右所示，图中 shift size 为 (2,2,2)，一般 shift size 都是 window size 的一半，也就是 $(\frac{P}{2}, \frac{M}{2}, \frac{M}{2})$。shift 之后，window 会往前，往右，往下分别移动对应的 size，目的是让 patch 可以和不同 window 的 patch 做特征的融合，这样多过几层之后，也就相当于做了全局的特征融合。

不过这里有一个问题，shift 之后，window 的数量从原来的 2×2×2=8 变成了 3×3×3=27。这带来的弊端就是计算时窗口不统一会比较麻烦。为了解决这个问题，本文引入了 mask，并将窗口的位置进行了移动，使得进行 shift 和不进行 shift 的 MSA 计算方式相同，只不过 mask 不同。

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-23-17-03-21.png" alt="VideoSwinTransformer-2022-04-23-17-03-21" style="zoom:50%;" /></div>

我们的目的是把图 3 中右侧的 27 个 windows 变成和中间那样的 8 个 window。给每个 window 都标了序号，标序号的方式是从前往后，从上往下，从左往右。shift window 的方法就是把左上角的移到右下角，把前面的移到后面。这样一来，比如 [27,25,21,19,9,7,3,1] 就组成了 1 个 window，[18,16,12,10] 就组成了 1 个 window，依此类推，一共有 8 个 windows。平移的方式可以和上述的不同，只要保证可以把 27 个 windows 变成和 8 个 windows 的计算方式一样即可。

这样在每个 window 做 self-attention 的时候，需要加一层 mask，可以说是引入了 Inductive bias。因为在组合而成的 window 内，各个小 window 我们不希望他们交换信息，因为这不是图像原有的位置，比如 17 和 11 经过 shift 之后，会在同一个 window 内做 attention，但是 11 是从上面移下来的，只是为了计算的统一，并不是物理意义上的同一个 window。有了 mask 就不一样了，**mask 的目的是告诉 17 号窗口内的每一个 patch，只和 17 号窗口内的 patches 做 attention，不和 11 号窗口内的做 attention，依此类推其他**。

mask 的生成方法可以参见 [源码](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py#L317)，主要思路是给每个 patch 一个 window 的编号，编号相同的 patch 之间 mask 为 0，否则为 - 100。

```python
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
```

如果 window 的大小为图中的 (P,M,M) 的话，attention mask 就是一个 (P×M×M，P×M×M) 的矩阵，这是一个对称矩阵，第 i 行第 j 列就表示 window 中的第 i 个 patch 和第 j 个 patch 的 window 编号是否是相同的，相同则为 0，不同则为 -100。对角线上的元素必为 0。

有人认为浅层的网络需要 SW-MSA，深层的就不需要了，因为浅层已经讲全局的信息都交流了，深层不需要进一步交流了。这种说法的确有一定的道理，但也要看网络的深度和 shift 的尺寸。

#### 2.2.5 Relative Position Bias

在上述的所有内容中，都没有涉及到位置的概念，也就是模型并不知道每个 patch 在图片中和其他 patches 的位置关系是怎么样的，最有也就是知道某几个 patch 是在同一个 window 内的，但 window 内的具体位置也是不知道的，因此就有了 Relative Position Bias。它是加在 attention 的部分的，下式中的 B 就是 Relative Position Bias。

$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}} + B)V
$$

很多 Swin Tranformer 的文章都会讲这个 B 是如何得到的，但是却没有讲为什么要这样生成 B。其实只要知道了设计这个 B 的目的，就可以不用管是如何生成的了，甚至自己设计一种生成的方法都行。

B 是为了表示一个 windows 内，每个 patch 的相对位置，给每个相对位置一个特殊的 embedding 值。其实也正是因为这个 B 的存在，SW-MSA 才必须要有 mask，因为 SW-MSA 内的 patches 可能来自于多个 windows，相对位置不能按照这个方法给，如果 B 可以表示全图的相对位置，那就不用这个 mask 了。

这个 B 和 mask 的 shape 是一致的，也是 (P×M×M，P×M×M) 的矩阵，第 i 行第 j 列就表示 window 中的第 j 个 patch 相对于第 i 个 patch 的位置。

如果 window size 为 (P,M,M) 的话，那么相对位置状态就会有 (2P−1)×(2M−1)×(2M−1) 种状态，把 (2,2,2) 的 window 的 27 种相对位置状态全都在图上写出来了。

<div align=center><img src="/assets/VideoSwinTransformer-2022-04-23-17-10-45.png" alt="VideoSwinTransformer-2022-04-23-17-10-45" style="zoom:50%;" /></div>

有了状态之后，就只需要在 B 这个矩阵中将相对位置的状态对号入座即可。这就是相对位置坐标相减，然后加个偏置，再乘个系数。

但最终使用的不是状态，而是状态对应的 embedding 值，这就需要有一个 table 来根据状态查找 embedding，这个 embedding 是模型训练出来的。

## 三、模型效果

作者在三个数据集上进行了测试，分别是 [Kinetics-400](https://deepmind.com/research/open-source/kinetics)，[Kinetics-600](https://deepmind.com/research/open-source/kinetics) 和 [Something-Something v2](https://developer.qualcomm.com/software/ai-datasets/something-something)。每个数据集上都有着 state-of-art 的表现。