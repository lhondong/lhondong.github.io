
# An image is worth 16×16 words: Transformers for image recognition at scale  

ICLR 2021

Google Research, Google Brain Team

[代码](https://github.com/google-research/vision_transformer)

<div align=center><img src="/assets/ViT.gif" alt="fit" style="zoom:50%;" /></div>

ViT(Vision Transformer) 直接将直接应用在图像，经过微调：将图像拆成 16x16 patch，然后将 patch 的 the sequence of linear embedding 作为 Transformers 的输入。

1. 数据处理部分：先对图片作分块，再将每个图片块展平成一维向量
2. 数据嵌入部分：
   - Patch Embedding：对每个向量都做一个线性变换 
   - Positional Encoding：通过一个可学习向量加入序列的位置信息
3. 编码部分：class_token：额外的一个用于分类的可学习向量，与输入进行拼接
4. 分类部分：mlp_head：使用 LayerNorm 和两层全连接层实现的，采用的是 GELU 激活函数

但是实验表明，在中等尺寸数据集训练后，分类正确率相比于 ResNet 上往往降低几个百分点，这是由于 Transformer 缺乏 CNN 的固有的 inductive bias 如 translation equivariance and locality，因而在数据不充分情况时不能很好泛化。而在数据尺寸足够的情况下训练 transfprmer，是能够应对这种 inductive bias，实现对流行模型的性能逼近甚至超越。

### CNN 的归纳偏置

Inductive biases 归纳偏置，可以理解为先验知识。CNN 的 Inductive biases 是 locality 和 平移等变性 translation equaivariance（平移不变性 spatial invariance）。

- Locality: CNN 用滑动窗口在图片上做卷积。假设是图片相邻的区域有相似的特征。i.e., 桌椅在一起的概率大，距离近的物品 相关性越强。
- Translation Equaivariance：$f(g(x)) = g(f(x))$  ，卷积 $f$ 和平移 $g$ 函数的顺序不影响结果。

CNN 的卷积核像一个 template 模板，同样的物体无论移动到哪里，遇到了相同的卷积核，它的输出一致。

CNN 有 locality 和 translation equivariance 归纳偏置，即 CNN 有很多先验信息，所以只需要较少的数据就可以学好一个模型。

Transformer 没有这些先验信息，只能从图片数据里，自己学习对**视觉世界**的感知。因此 Transformers 在小数据上的预测正确率比 CNN 低，当采用混合结构时（即将 CNN 的输出特征作为输入序列时，尽在小数据上实现性能提升），这与我们预期有差，期望 CNN 的引入能够提升所有尺寸训练样本下的性能。就是凭借一些规律得出的偏好：如 CNN 天然的对图像处理的较好，天然的具有平移不变性等。

### 怎么验证 Transformer 无 inductive bias 的假设？

在 1400 万 (ImageNet-21K) - 3000 万 (JFT-300) 得到图片数据集上预训练 trumps inductive bias, ViT + 足够训练数据，CV SOTA。

###  如何理解 CNN 整体结构不变？

ResNet 50 有 4 个 stages (res2, res3, res4, res5), stage 不变，Attention 取代 每一个 stage 每一个 block 里的这个操作。

### 如何理解 Patch？

patch 是将 3 维图像 reshape 为 2 维之后进行切分，使用的 position embedding 是 1 维，将 patch 作为一个小整体，然后对 patch 在整个图像中的位置进行编码，还是按照分割后的位置信息。

### Transformer 应用在 CV 的难点

- 计算像素的 self-attention，序列长，维度爆炸
- Transformer 的计算复杂度是 序列长度 n 的 平方 $O(n^2)$。224 分辨率的图片，有 50176 个像素点，（2d 图片 flatten）序列长度是 BERT 的近 100 倍。

## ViT 原理分析

这个工作本着尽可能少修改的原则，将原版的 Transformer 开箱即用地迁移到分类任务上面。并且作者认为没有必要总是依赖于 CNN，只用 Transformer 也能够在分类任务中表现很好，尤其是在使用大规模训练集的时候。同时，在大规模数据集上预训练好的模型，在迁移到中等数据集或小数据集的分类任务上以后，也能取得比 CNN 更优的性能。

首先把 $x\in H \times W \times C$ 的图像，变成一个 $x_p \in N \times (P^2 \cdot C)$ 的 sequence of flattened 2D patches。它可以看做是一系列的展平的 2D 块的序列，这个序列中一共有 $N =HW/P^2$ 个展平的 2D 块，每个块的维度是 $(P^2\times C)$。其中 $P$ 是块大小，$C$ 是 channel 数。

**注意作者做这步变化的意图**：根据之前的讲解，Transformer 希望输入一个二维的矩阵 $(N,D)$ ，其中 $N$ 是 sequence 的长度，$D$ 是 sequence 的每个向量的维度，常用 256。所以这里也要设法把 $H\times W \times C$ 的三维图片转化成 $(N,D)$ 的二维输入。

所以有：$H \times W \times C \to N \times (P^2 \cdot C)$, where $N=HW/P^2$ 。

其中，$N$ 是 Transformer 输入的 sequence 的长度。

代码是：

```python
x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
```

具体是采用了 einops 库实现，具体可以参考这篇博客，科技猛兽：PyTorch 70.einops：[优雅地操作张量维度](https://zhuanlan.zhihu.com/p/342675997)

现在得到的向量维度是：$x_p \in N \times (P^2 \times C)$，要转化成 $(N,D)$ 的二维输入，我们还需要做一步叫做 Patch Embedding 的步骤。

### Patch Embedding

方法是对每个向量都做一个线性变换（即全连接层），压缩后的维度为 $D$ ，这里我们称其为 Patch Embedding。

$$
z_0 = [x_{class};  x_p^1E; x_p^2E; .... ; x_p^nE]+ E_{pos}
$$

这个全连接层就是上式中的 $E$，它的输入维度大小是 $(P^2 \cdot C)$，输出维度大小是 $D$。

```python
# 将 3072 变成 dim，假设是 1024
self.patch_to_embedding = nn.Linear(patch_dim, dim)
x = self.patch_to_embedding(x)
```

注意这里 $x_{class}$，假设切成 9 个块，但是最终到 Transfomer 输入是 10 个向量，这是人为增加的一个向量。

### 为什么要加这个向量？

如果没有这个向量，假设 $N=9$ 个向量输入 Transformer Encoder，输出 9 个编码向量，然后呢？对于分类任务而言，我应该取哪个输出向量进行后续分类呢？  

不知道。干脆就再来一个向量 $x_{class}(vector ,dim =D)$，这个向量是可学习的嵌入向量，它和那 9 个向量一并输入 Transfomer Encoder，输出 1+9 个编码向量。然后就用第 0 个编码向量，即 $x_{class}$ 的输出进行分类预测即可。

这么做的原因可以理解为：ViT 其实只用到了 Transformer 的 Encoder，而并没有用到 Decoder，而 $x_{class}$ 的作用有点类似于解码器中的 Query 的作用，相对应的 Key,Value 就是其他 9 个编码向量的输出。$x_{class}$ 是一个可学习的嵌入向量，它的意义说通俗一点为：寻找其他 9 个输入向量对应的 image 的类别。

代码为：

```python
# dim=1024
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

# forward 前向代码
# 变成 (b,64,1024)
cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
# 跟前面的分块进行 concat
# 额外追加 token，变成 b,65,1024
x = torch.cat((cls_tokens, x), dim=1)
```

### Positional Encoding

按照 Transformer 的位置编码的习惯，这个工作也使用了位置编码。**引入了一个 Positional encoding $E_{pos}$ ​来加入序列的位置信息**，同样在这里也引入了 pos_embedding，是用一个可训练的变量。

$$
z_0 = [x_{class}; x_p^1E; x_p^2E; .... ; x_p^nE]+E_{pos}
$$

没有采用原版 Transformer 的 sincos 编码，而是直接设置为可学习的 Positional Encoding，效果差不多。对训练好的 pos_embedding 进行可视化，如下图所示。

<div align=center><img src="/assets/ViT-2022-04-24-10-59-03.png" alt="ViT-2022-04-24-10-59-03" style="zoom:100%;" /></div>

发现**位置越接近，往往具有更相似的位置编码**。此外，出现了行列结构：**同一行/列中的 patch 具有相似的位置编码。**

```python
# num_patches=64，dim=1024,+1 是因为多了一个 cls 开启解码标志
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
```

### Transformer Encoder 的前向过程

$$
z_0 = [x_{class}; x_p^1E; x_p^2E; .... ; x_p^nE]+ E_{pos}, \qquad E\in \mathbb{R}^{P^2 \times C\times D}, E_{pos} \in \mathbb{R}^{(N+1)\times D}$$

$$
{z}'_\ell=MSA(LN(z_{\ell-1}))+z_{\ell-1}, \qquad \ell=1...L  \qquad \qquad
$$

$$
z_{\ell} = MLP(LN({z}'_\ell))+{z}'_{ell}, \qquad  \ell=1...L 
$$

$$
y = LN(z^0_{\ell})
$$

- 其中，第 1 个式子为上面讲到的 Patch Embedding 和 Positional Encoding 的过程。
- 第 2 个式子为 Transformer Encoder 的 Multi−head, Self−Attention, AddandNorm 的过程，重复 L 次。
- 第 3 个式子为 Transformer Encoder 的 FeedForward, AddandNorm 的过程，重复 L 次。采用的是没有任何改动的 Transformer。
- 最后是一个 Classfication−Head，整个的结构只有这些，如下图所示，为了方便理解，把变量的维度变化过程标注在了图中。

<div align=center><img src="/assets/ViT-2022-04-24-11-04-21.png" alt="ViT-2022-04-24-11-04-21" style="zoom:50%;" /></div>

```python
x  = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
```

### 训练方法

先在大数据集上预训练，再迁移到小数据集上面。做法是把 ViT 的 prediction−head 去掉，换成一个 $D \times K$ 的 FeedForwardLayer。其中 $K$ 为对应数据集的类别数。

当输入的图片是更大的 shape 时，patch size $P$ 保持不变，则 $N=HW/P^2$ 会增大。

ViT 可以处理任意 $N$ 的输入，但是 Positional Encoding 是按照预训练的输入图片的尺寸设计的，所以输入图片变大之后，Positional Encoding 需要根据它们在原始图像中的位置做 2D 插值。

### 整个流程

- 输入图像 224×224×3，每个 patch 大小为 16×16，则打成 patches 后，维度变为 $(\frac{224}{16}\times\frac{224}{16})\times(16\times16\times 3)=196\times 768$，即 196 个 patches，每个维度是 768；
- 线性投射层 E （本质就是全连接层）的维度为 $768\times 768$，前面的 768 来自图像输入，后一个 768 则是论文中维度 D，可以有 768, 1024, 1280 三个版本。
- 输入经过线性投射层之后，$X_{196\times 768} \cdot D_{768\times 768}=Embedding_{196\times 768}$；
- 拼接一个 cls tokens，变成 197×768；
- 再加上 position embedding，还是 197×768；
- 进入 Transformer Encoder，layer norm 不改变维度，多头自注意力的头如果是 12，则 kqv 的维度为 $197\times \frac{768}{12} = 197\times 64$，三个结果拼接又变回 197×768；
- 经过 Transformer Encoder 中的 MLP 是会先将维度放大 4 倍，变为 196×3072，然后再缩小投射回去变为 197×768。

<div align=center><img src="/assets/ViT-2022-05-28-09-53-25.png" alt="ViT-2022-05-28-09-53-25" style="zoom:50%;" /></div>

### Hybrid 模型详解

在论文 4.1 章节的 Model Variants 中详细讲了 Hybrid 混合模型，就是将传统 CNN 特征提取和 Transformer 进行结合。

下图绘制的是以 ResNet50 作为特征提取器的混合模型，但这里的 Resnet 与之前讲的 Resnet 有些不同。首先这里的 R50 的卷积层采用的 StdConv2d 不是传统的 Conv2d，然后将所有的 BatchNorm 层替换成 GroupNorm 层。在原 Resnet50 网络中，stage1 重复堆叠 3 次，stage2 重复堆叠 4 次，stage3 重复堆叠 6 次，stage4 重复堆叠 3 次，但在这里的 R50 中，把 stage4 中的 3 个 Block 移至 stage3 中，所以 stage3 中共重复堆叠 9 次。

通过 R50 Backbone 进行特征提取后，得到的特征矩阵 shape 是 [14, 14, 1024]，接着再输入 Patch Embedding 层，注意 Patch Embedding 中卷积层 Conv2d 的 kernel_size 和 stride 都变成了 1，只是用来调整 channel。后面的部分和前面 ViT 中讲的完全一样，就不在赘述。

<div align=center><img src="/assets/ViT-2022-05-28-09-54-48.png" alt="ViT-2022-05-28-09-54-48" style="zoom:50%;" /></div>

### Experiments

预训练模型使用到的数据集有：

- ILSVRC-2012 ImageNet dataset：1000 classes
- ImageNet-21k：21k classes
- JFT：18k High Resolution Images

将预训练迁移到的数据集有：

- CIFAR-10/100
- Oxford-IIIT Pets
- Oxford Flowers-102
- VTAB

设计了 3 种不同答小的 ViT 模型，分别是：

|DModel|Layers|Hidden size|MLP size|Heads|Params|
|---|---|---|---|---|---|
|ViT-Base|12|768|3072|12|86M
|ViT-Large|24|1024|4096|16|307M
|ViT-Huge|32|1280|5120|16|632M|

ViT-L/16 代表 ViT-Large + 16 patch size

### 评价指标 Metrics

结果都是下游数据集上经过 finetune 之后的 Accuracy，记录的是在各自数据集上 finetune 后的性能。

<div align=center><img src="/assets/ViT-2022-04-24-11-18-09.png" alt="ViT-2022-04-24-11-18-09" style="zoom:50%;" /></div>

#### 实验 1：性能对比

实验结果如下图所示，整体模型还是挺大的，而经过大数据集的预训练后，性能也超过了当前 CNN 的一些 SOTA 结果。对比的 CNN 模型主要是：

- 2020 年 ECCV 的 Big Transfer (BiT) 模型，它使用大的 ResNet 进行有监督转移学习。
- 2020 年 CVPR 的 Noisy Student 模型，这是一个在 ImageNet 和 JFT300M 上使用半监督学习进行训练的大型高效网络，去掉了标签。

All models were trained on TPUv3 hardware。

<div align=center><img src="/assets/ViT-2022-04-24-11-18-53.png" alt="ViT-2022-04-24-11-18-53" style="zoom:50%;" /></div>

在 JFT-300M 上预先训练的较小的 ViT-L/16 模型在所有任务上都优于 BiT-L（在同一数据集上预先训练的），同时训练所需的计算资源要少得多。更大的模型 ViT-H/14 进一步提高了性能，特别是在更具挑战性 ImageNet, CIFAR-100 和 VTAB 数据集上。与现有技术相比，该模型预训练所需的计算量仍然要少得多。

下图为 VTAB 数据集在 Natural, Specialized, 和 Structured 子任务与 CNN 模型相比的性能，ViT 模型仍然可以取得最优。

<div align=center><img src="/assets/ViT-2022-04-24-11-20-37.png" alt="ViT-2022-04-24-11-20-37" style="zoom:50%;" /></div>

#### 实验 2：ViT 对预训练数据的要求

ViT 对于预训练数据的规模要求到底有多苛刻？作者分别在下面这几个数据集上进行预训练：ImageNet, ImageNet-21k, 和 JFT-300M。

结果如下图所示：

<div align=center><img src="/assets/ViT-2022-04-24-11-21-16.png" alt="ViT-2022-04-24-11-21-16" style="zoom:50%;" /></div>

发现当在最小数据集 ImageNet 上进行预训练时，尽管进行了大量的正则化等操作，但 ViT-Large 的性能不如 ViT-Base 模型。使用稍微大一点的 ImageNet-21k 预训练，它们的表现也差不多。

只有到了 JFT 300M，才能看到更大的 ViT 模型全部优势。图 3 还显示了不同大小的 BiT 模型跨越的性能区域。BiT CNNs 在 ImageNet 上的表现优于 ViT（尽管进行了正则化优化），但在更大的数据集上，ViT 超过了所有的模型，取得了 SOTA。

作者还进行了一个实验：在 9M、30M 和 90M 的随机子集以及完整的 JFT300M 数据集上训练模型，结果如下图所示。ViT 在较小数据集上的计算成本比 ResNet 高，ViT-B/32 比 ResNet50 稍快；它在 9M 子集上表现更差，但在 90M + 子集上表现更好。ResNet152x2 和 ViT-L/16 也是如此。这个结果强化了一种直觉，即：

**残差对于较小的数据集是有用的，但是对于较大的数据集，像 attention 一样学习相关性就足够了，甚至是更好的选择。**

<div align=center><img src="/assets/ViT-2022-04-24-11-23-14.png" alt="ViT-2022-04-24-11-23-14" style="zoom:50%;" /></div>

#### 实验 3：ViT 的注意力机制 Attention

作者还给了注意力观察得到的图片块，Self-attention 使得 ViT 能够整合整个图像中的信息，甚至是最底层的信息。作者欲探究网络在多大程度上利用了这种能力。

具体来说，根据**注意力权重**计算图像**空间中整合信息的平均距离**，如下图所示。

<div align=center><img src="/assets/ViT-2022-04-24-11-23-53.png" alt="ViT-2022-04-24-11-23-53" style="zoom:100%;" /></div>

注意这里只使用了 attention，而没有使用 CNN，所以这里的 attention distance 相当于 CNN 的 receptive field 的大小。 

作者发现：在最底层，有些 head 也已经注意到了图像的大部分，说明模型已经可以 globally 地整合信息了，它们负责 global 信息的整合。其他的 head 只注意到图像的一小部分，说明它们负责 local 信息的整合。Attention Distance 随深度的增加而增加。

整合局部信息的 attention head 在混合模型 （有 CNN 存在） 时，效果并不好，说明它可能与 CNN 的底层卷积有着类似的功能。

作者给出了 attention 的可视化，注意到了适合分类的位置：

<div align=center><img src="/assets/ViT-2022-04-24-11-25-08.png" alt="ViT-2022-04-24-11-25-08" style="zoom:50%;" /></div>