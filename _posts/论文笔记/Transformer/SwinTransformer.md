
# Swin Transformer

- Swin Transformer 使用了类似卷积神经网络中的层次化构建方法（Hierarchical feature maps），比如特征图尺寸中对图像下采样 4 倍的，8 倍的以及 16 倍，这样的 backbone 有助于在此基础上构建目标检测，实例分割等任务。而在之前的 Vision Transformer 中是一开始就直接下采样 16 倍，后面的特征图也是维持这个下采样率不变。
- 在 Swin Transformer 中使用了 Windows Multi-Head Self-Attention(W-MSA) 的概念，比如在下图的 4 倍下采样和 8 倍下采样中，将特征图划分成了多个不相交的区域（Window），并且 Multi-Head Self-Attention 只在每个窗口（Window）内进行。相对于 Vision Transformer 中直接对整个（Global）特征图进行 Multi-Head Self-Attention，这样做的目的是能够减少计算量的，尤其是在浅层特征图很大的时候。
- W-MSA 虽然减少了计算量但也会隔绝不同窗口之间的信息传递，所以作者又提出了 Shifted Windows Multi-Head Self-Attention(SW-MSA) 的概念，通过此方法能够让信息在相邻的窗口中进行传递。

<div align=center><img src="/assets/SwinTransformer-2022-05-27-09-42-36.png" alt="SwinTransformer-2022-05-27-09-42-36" style="zoom:50%;" /></div>

## 网络框架

### 整体流程

- 图片预处理：分块和降维 (Patch Partition)
- 线性变换 (Linear Embedding)
- Swin Transformer Block
- Stage 2/3/4

<div align=center><img src="/assets/SwinTransformer-2022-05-27-10-04-26.png" alt="SwinTransformer-2022-05-27-10-04-26" style="zoom:50%;" /></div>

- 首先将图片输入到 Patch Partition 模块中进行分块，即每 4x4 相邻的像素为一个 Patch，然后在 channel 方向展平（flatten）。假设输入的是 RGB 三通道图片，那么每个 patch 就有 4x4x3=48 个像素，通过 Patch Partition 后图像 shape 由 [H, W, 3] 变成了 [H/4, W/4, 48]
- 通过 Linear Embeding 层对每个像素的 channel 数据做线性变换，由 48 变成 C，即图像 shape 再由 [H/4, W/4, 48] 变成了 [H/4, W/4, C]。
- 其实在源码中 Patch Partition 和 Linear Embeding 就是直接通过一个卷积层实现的，和之前 Vision Transformer 中讲的 Embedding 层结构一模一样。
- 然后就是通过四个 Stage 构建不同大小的特征图，除了 Stage1 中先通过一个 Linear Embeding 层外，剩下三个 stage 都是先通过一个 Patch Merging 层进行下采样。
- 然后重复堆叠 Swin Transformer Block。注意这里的 Block 其实有两种结构，如图 (b) 中所示，这两种结构的不同之处仅在于一个使用了 W-MSA 结构，一个使用了 SW-MSA 结构。而且这两个结构是成对使用的，先使用一个 W-MSA 结构再使用一个 SW-MSA 结构。所以堆叠 Swin Transformer Block 的次数都是偶数（因为成对使用）。
- 最后对于分类网络，后面还会接上一个 Layer Norm 层、全局池化层以及全连接层得到最终输出。图中没有画，但源码中是这样做的。

### 维度变化

- 输入 224×224×3
- Patch Partition -> 56×56×48
- Linear Embedding -> 56×56×96 (Swin-Tiny 中 C=96)
- Stage1 -> 56×56×96 -> Stage2 -> 28×28×192 -> Stage3 -> 14×14×384 -> Stage4 -> 7×7×768


### Patch Merging

在每个 Stage 中首先要通过一个 Patch Merging 层进行下采样（Stage1 除外）。如下图所示，假设输入 Patch Merging 的是一个 4x4 大小的单通道特征图（feature map），Patch Merging 会将每个 2x2 的相邻像素划分为一个 patch，然后将每个 patch 中相同位置（同一颜色）像素给拼在一起就得到了 4 个 feature map。接着将这四个 feature map 在深度方向进行 concat 拼接，相当于将空间的维度转换为更多的通道数：H×W×C 变为 H/2 × W/2 × 4C。

接下来通过一个 LayerNorm 层后，最后通过一个 1×1 卷积将 feature map 的通道深度由 4C 变成 2C。也就是说，通过 Patch Merging 层后，feature map 的高和宽会减半，深度会翻倍。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-14-52.png" alt="SwinTransformer-2022-05-28-16-14-52" style="zoom:50%;" /></div>

### W-MSA

引入 Windows Multi-head Self-Attention（W-MSA）模块是为了减少计算量。

普通的 Multi-head Self-Attention（MSA）模块，对于 feature map 中的每个patch 在 Self-Attention 计算过程中需要和所有的像素去计算。

但在使用 Windows Multi-head Self-Attention（W-MSA）模块时，首先将 feature map 按照 MxM（M=7）大小划分成一个个 Windows。这样 56×56 的特征图就会有 8×8 个 Windows，每个 Window 里有 7×7 个 patches。然后单独对每个 Windows 内部进行 Self-Attention。

两者的计算量具体差多少呢？原论文中有给出下面两个公式，这里忽略了 Softmax 的计算复杂度：  

$$
\Omega (MSA)=4hwC^2 + 2{(hw)}^2C
$$

$$
\Omega (W-MSA)=4hwC^2 + 2M^2hwC
$$

- h 代表 feature map 的高度
- w 代表 feature map 的宽度
- C 代表 feature map 的深度
- M 代表每个窗口（Windows）的大小

#### MSA 模块计算量

首先 Self-Attention 的公式：

$$
Attention(Q, K, V)={\rm SoftMax}(\frac{QK^T}{\sqrt d})V 
$$

对于 feature map 中的每个像素（或称作 token，patch），都要通过 $W_q, W_k, W_v$ ​生成对应的 query(q)，key(k) 以及 value(v)。这里假设 q, k, v 的向量长度与 feature map 的深度 C 保持一致。那么对应所有像素生成 Q 的过程如下式：  

$$
I^{hw \times C} \cdot W^{C \times C}_q=Q^{hw \times C}
$$

- $I^{hw \times C}$ 为将所有像素（token）拼接在一起得到的矩阵（一共有 hw 个像素，每个像素的深度为 C）
- $W^{C \times C}_q$ ​为生成 query 的变换矩阵
- $Q^{hw \times C}$ 为所有像素通过 $W^{C \times C}_q$ 得到的 query 拼接后的矩阵

根据矩阵运算的计算量公式可以得到生成 Q 的计算量为 $hw \times C \times C$，生成 K 和 V 同理都是 $hwC^2$，那么总共是 $3hwC^2$。接下来 $Q$ 和 $K^T$ 相乘，对应计算量为 $(hw)^2C$：  

$$
Q^{hw \times C} \cdot {K^T}^{C \times hw}= A^{hw \times hw}
$$

接下来忽略除以 $\sqrt d$ ​以及 softmax 的计算量，假设得到 $\Lambda ^{hw \times hw}$，最后还要乘以 V，对应的计算量为 $(hw)^2C$:  

$$
\Lambda ^{hw \times hw} \cdot V^{hw \times C}=B^{hw \times C}
$$

那么对应单头的 Self-Attention 模块，总共需要 $3hwC^2 + (hw)^2C + (hw)^2C=3hwC^2 + 2(hw)^2C$。

而在实际使用过程中使用 Multi-head Self-Attention 模块，多头注意力模块相比单头注意力模块的计算量仅多了最后一个融合矩阵 $W_O$ 的计算量 $hwC^2$。  

$$
B^{hw \times C} \cdot W_O^{C \times C} = O^{hw \times C}
$$

所以总共加起来是： $4hwC^2 + 2(hw)^2C$

#### W-MSA 模块计算量

对于 W-MSA 模块首先要将 feature map 划分到一个个窗口（Windows）中，假设每个窗口的宽高都是 M，那么总共会得到 $\frac {h} {M} \times \frac {w} {M} $ 个窗口，然后对每个窗口内使用多头注意力模块。

刚刚计算高为 h，宽为 w，深度为 C 的 feature map 的计算量为 $4hwC^2 + 2(hw)^2C$，这里每个窗口的高为 M 宽为 M，带入公式得 $4(MC)^2 + 2(M)^4C$，又因为有 $\frac {h} {M} \times \frac {w} {M}$ 个窗口，则：  

$$
\frac {h} {M} \times \frac {w} {M} \times (4(MC)^2 + 2(M)^4C)=4hwC^2 + 2M^2 hwC
$$

故使用 W-MSA 模块的计算量为： $4hwC^2 + 2M^2 hwC$

假设 feature map 的 h、w 都为 112，M=7，C=128，采用 W-MSA 模块相比 MSA 模块能够节省约 40124743680 FLOPs：  

$$
2(hw)^2C-2M^2 hwC=2 \times 112^4 \times 128 - 2 \times 7^2 \times 112^2 \times 128=40124743680
$$

### SW-MSA

前面有说，采用 W-MSA 模块时，只会在每个窗口内进行自注意力计算，所以窗口与窗口之间是无法进行信息传递的。为了解决这个问题，引入了 Shifted Windows Multi-Head Self-Attention（SW-MSA）模块，即移动的 W-MSA。

如下图所示，左侧使用的是刚刚讲的 W-MSA（假设是第 L 层），那么根据之前介绍的 W-MSA 和 SW-MSA 是成对使用的，那么第 L+1 层使用的就是右侧的 SW-MSA。根据左右两幅图对比能够发现窗口（Windows）发生了偏移（可以理解成窗口从左上角分别向右侧和下方各偏移了 $\left \lfloor \frac {M} {2} \right \rfloor$ 个像素）。看下偏移后的窗口，比如对于第一行第 2 列的 2x4 的窗口，它能够使第 L 层的第一排的两个窗口信息进行交流。再比如，第二行第二列的 4x4 的窗口，他能够使第 L 层的四个窗口信息进行交流，其他的同理。那么这就解决了不同窗口之间无法进行信息交流的问题。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-14-57-59.png" alt="SwinTransformer-2022-05-28-14-57-59" style="zoom:50%;" /></div>

根据上图，可以发现通过将窗口进行偏移后，由原来的 4 个窗口变成 9 个窗口了。后面又要对每个窗口内部进行 MSA，这样做计算复杂度又增加了。为了解决这个问题，作者又提出了 Efficient batch computation for shifted configuration ，一种更加高效的计算方法，通过 cycle shift 的方法，合并小的 windows，将 A,B,C 这3个小的 windows 进行循环移位，使成大的 window。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-14-59-38.png" alt="SwinTransformer-2022-05-28-14-59-38" style="zoom:50%;" /></div>

经过了 cycle shift 的方法，一个 window 可能会包括来自不同 window 的内容。比如图4右下角的 window，来自4个不同的 sub-window。因此，要采用 masked MSA 机制将 self-attention 的计算限制在每个子窗口内。最后通过 reverse cycle shift 的方法将每个 window 的 self-attention 结果返回。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-15-42-10.png" alt="SwinTransformer-2022-05-28-15-42-10" style="zoom:50%;" /></div>

这样再按照之前的 window 划分，就能够得到 window 5 的attention 的结果了。但是这样操作会使得 window 6 和 4 的 attention 混在一起，window 1,3,7 和 9 的 attention 混在一起。所以需要采用 masked MSA 机制将 self-attention 的计算限制在每个子窗口内。也就是在做正常的 self-attention (在 window_size 上做)之后进行一次 mask 操作，把不需要的 attention 值给置为 0。

比如 4 号和 6 号 window 组成的新的 window，使用 mask 将其来自4 号和 6 号的 patch 算出来的 attention 置 0：

<div align=center><img src="/assets/SwinTransformer-2022-05-28-15-51-12.png" alt="SwinTransformer-2022-05-28-15-51-12" style="zoom:100%;" /></div>

整体的掩码操作如图：

<div align=center><img src="/assets/SwinTransformer-2022-05-28-15-53-08.png" alt="SwinTransformer-2022-05-28-15-53-08" style="zoom:50%;" /></div>

### Relative Position Bias

关于相对位置偏置，在 Imagenet 数据集上如果不使用任何位置偏置，top-1为 80.1，但使用了相对位置偏置（rel. pos.）后 top-1 为 83.3，提升还是很明显的。根据论文中提供的公式可知是在 Q 和 K 进行匹配并除以 $\sqrt d$ ​后加上了相对位置偏置 B。  

$$
Attention(Q, K, V)=SoftMax(\frac {QK^T} {\sqrt d} + B)V
$$

由于论文中并没有讲这个相对位置偏置，所以根据阅读源码做了简单的总结。如下图，假设输入的 feature map 高宽都为 2，那么首先我们可以构建出每个像素的绝对位置（左下方的矩阵），对于每个像素的绝对位置是使用行号和列号表示的。比如蓝色的像素对应的是第 0 行第 0 列所以绝对位置索引是 (0,0)，接下来再看看相对位置索引。

首先看蓝色的像素，在蓝色像素使用 q 与所有像素 k 进行匹配过程中，是以蓝色像素为参考点。然后用蓝色像素的绝对位置索引与其他位置索引进行相减，就得到其他位置相对蓝色像素的**相对位置索引**。例如黄色像素的绝对位置索引是 (0,1)，则它相对蓝色像素的相对位置索引为 (0,0)−(0,1)=(0,−1)。

那么同理可以得到其他位置相对蓝色像素的相对位置索引矩阵。同样，也能得到相对黄色，红色以及绿色像素的相对位置索引矩阵。接下来将每个相对位置索引矩阵按行展平，并拼接在一起可以得到下面的 4x4 矩阵 。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-16-53.png" alt="SwinTransformer-2022-05-28-16-16-53" style="zoom:50%;" /></div>

请注意，这里描述的一直是**相对位置索引**，并不是**相对位置偏置参数**。因为后面会根据相对位置索引去取对应的参数。

比如说黄色像素是在蓝色像素的右边，所以相对蓝色像素的相对位置索引为 (0,−1)。绿色像素是在红色像素的右边，所以相对红色像素的相对位置索引为 (0,−1)。可以发现这两者的相对位置索引都是 (0,−1)，所以他们使用的**相对位置偏置参数**都是一样的。

其实讲到这基本已经讲完了，但在源码中作者为了方便把二维索引给转成了一维索引，但是简单地直接把行列索引相加不行，比如上面的相对位置索引中有 (0,−1) 和 (−1,0) 在二维的相对位置索引中明显是代表不同的位置，但如果简单相加都等于 -1 就出问题了。

首先在原始的相对位置索引上加上 M-1(M 为窗口的大小，在本示例中 M=2)，加上之后索引中就不会有负数了。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-19-26.png" alt="SwinTransformer-2022-05-28-16-19-26" style="zoom:50%;" /></div>

接着将所有的行标都乘上 2M-1。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-20-27.png" alt="SwinTransformer-2022-05-28-16-20-27" style="zoom:50%;" /></div>

最后将行标和列标进行相加。这样即保证了相对位置关系，而且不会出现上述 0+(−1)=(−1)+0 的问题了。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-21-16.png" alt="SwinTransformer-2022-05-28-16-21-16" style="zoom:50%;" /></div>

之前计算的是**相对位置索引**，并不是**相对位置偏置参数**。真正使用到的可训练参数 $\hat{B}$ 是保存在 relative position bias table 表里的，这个表的长度是等于 $(2M-1) \times (2M-1)$ 的。那么上述公式中的相对位置偏置参数 B 是根据上面的相对位置索引表根据查 relative position bias table 表得到的，如下图所示。

<div align=center><img src="/assets/SwinTransformer-2022-05-28-16-22-44.png" alt="SwinTransformer-2022-05-28-16-22-44" style="zoom:50%;" /></div>