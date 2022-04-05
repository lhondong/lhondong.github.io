
## 分组卷积 Grouped Convolution

Grouped convolution 分组卷积，最早在 AlexNet 中出现，由于当时的硬件资源有限，训练 AlexNet 时卷积操作不能全部放在同一个 GPU 处理，因此作者把 feature maps 分给多个 GPU 分别进行处理，最后把多个 GPU 的结果进行融合。

首先，传统的 2D 卷积通过应用 128 个 filters（每个 filter 的大小为 $3\times 3 \times 3$），大小为 $7\times 7 \times 3$ 的输入层被转换为大小为 $5\times 5 \times 128$ 的输出层。

针对通用情况，可概括为：通过应用 $c_2$ 个卷积核（每个卷积核的大小为 $h_1 \times w_1 \times c_1$），可将大小为 $H \times W \times c_1$ 的输入层转换为大小为 $H \times W \times c_2$ 的输出层，参数量是 $h_{1} \times w_{1} \times c_{1} \times c_{2}$。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-12-54-47.png" alt="ShuffleNet-2022-04-01-12-54-47" style="zoom:70%;" /></div>

在分组卷积中，filters 被拆分为不同的组，每一个组都负责具有一定深度的传统 2D 卷积的工作。

首先需要将原输入按照通道数量先分成 $g$ 个组。对于每个组来说，输入的尺寸是 $H \times W \times (c_{1}/g)$，相对应的每个组的输出尺寸应该是 $H \times W \times (c_{2}/g)$。那么每个组需要使用 $c2/g$ 个尺寸为 $h_{1} \times w_{1} \times c_{1}/g$ 的卷积核，那么每组参数量便是 $h_{1} \times w_{1} \times (c_{1}/g) \times(c_{2}/g)$，这样的操作需要进行 $g$ 次，于是，总的参数量便是 $h_{1} \times w_{1} \times (c_{1}/g) \times(c_{2}/g) \times g =h_{1} \times w_{1} \times c_{1} \times c_{2}/g$，也就是原来的正常卷积的 $1/g$。每一组卷积的计算结果最后 concat 起来也就能达到 $c_2$ 的通道数。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-13-02-58.png" alt="ShuffleNet-2022-04-01-13-02-58" style="zoom:100%;" /></div>

本质上来看，深度可分离卷积中的深度卷积就是一种特殊的分组卷积，只是说深度可分离卷积的分组的组数正好和输入图像的通道数一致。

#### 优点

第一个优点是有效的训练。由于卷积被划分为多个路径，因此每个路径可以由不同的 GPU 分别处理，此过程允许以并行方式在多个 GPU 上进行模型训练。与使用一个 GPU 进行所有训练相比，通过多 GPU 进行的模型并行化，可以将更多图像传到网络中。模型并行化被认为比数据并行化更好的方式，最终将数据集分成多个批次，然后我们对每个批次进行训练。但是，当批次大小变得太小时，与 batch 梯度下降相比，我们实际上是随机的，这将导致收敛变慢，有时甚至变差。

第二个优点是模型更有效，即模型参数随着 filters 组数的增加而减小。

第三个优点分组卷积可以提供比标准 2D 卷积更好的模型，一个很棒的 [博客](https://blog.yani.ai/filter-group-tutorial/) 对此进行了解释。

在 CIFAR10 上训练的 Network-in-Network 模型中，相邻层 filters 之间的相关矩阵。高度相关的 filters 较亮，而较低相关的 filters 则较暗。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-13-17-15.png" alt="ShuffleNet-2022-04-01-13-17-15" style="zoom:20%;" /></div>

当使用 1、2、4、8 和 16 个 filters 组训练时，在 CIFAR10 上训练的 Network-in-Network 模型中相邻层 filters 之间的相关性。

<div align=center><img src="/assets/cifar-nin-groupanimation.gif" alt="cifar-nin-groupanimation" style="zoom:20%;" /></div>

上面的图像是当使用 1、2、4、8 和 16 个 filters 组训练模型时，相邻层 filters 之间的相关性。提出了一个推理：“filters 组的作用是通过以对角线结构的稀疏性来学习 channel 维度……在具有 filters 组的网络中，以更结构化的方式来学习具有高相关性的 filters。实际上，不必学习的 filters 关系就在较长的参数上。在以这种显着的方式减少网络中参数的数量时，过拟合并不容易，因此类似正则化的效果使优化器可以学习到更准确，更有效的深度网络。”此外，每个 filter 组都会学习数据的唯一表示形式。正如 AlexNet 的作者所注意到的那样，filters 组似乎将学习到的 filters 分为两个不同的组，即黑白 filter 和彩色 filter。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-13-21-21.png" alt="ShuffleNet-2022-04-01-13-21-21" style="zoom:100%;" /></div>

### 随机分组卷积（Shuffled Grouped Convolution）

随机分组卷积是在 Magvii Inc（Face ++）的 [ShuffleNet](https://arxiv.org/abs/1707.01083) 中引入的。ShuffleNet 是一种计算效率高的卷积体系结构，专为计算能力非常有限（例如 10–150 MFLOP）的移动设备而设计。

随机分组卷积背后的思想与分组卷积（例如在 [MobileNet](https://arxiv.org/abs/1704.04861) 和 [ResNeXt](https://arxiv.org/abs/1611.05431) 中使用）和深度可分离卷积（在 [Xception](https://arxiv.org/abs/1610.02357) 中使用）背后的思想是相关的。

总的来说，随机分组卷积包括分组卷积和 channel shuffle。

在关于分组卷积的部分中，我们知道 filters 被分为不同的组，每个组负责具有一定深度的标准 2D 卷积，总操作数大大减少。对于下图中的示例，我们有 3 个 filters 组，第一个 filters 组与输入层中的红色部分卷积。类似地，第二和第三个 filters 组与输入中的绿色和蓝色部分卷积。每个 filter 组中的 kernels 深度仅为输入层中总通道数的 1/3。在此示例中，在第一次分组卷积 GConv1 之后，输入层被映射到中间特征图。然后，此特征图通过第二个分组卷积 GConv2 映射到输出层。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-13-23-41.png" alt="ShuffleNet-2022-04-01-13-23-41" style="zoom:50%;" /></div>

分组卷积在计算上是有效的，但是问题在于每个 filters 组仅处理从先前层中的固定部分向下传递的信息。例如上图中的示例，第一个 filters 组（红色）仅处理从前 1/3 个输入通道向下传递的信息。蓝色 filters 组（蓝色）仅处理从最后 1/3 个输入通道向下传递的信息。因此，每个 filters 组仅限于学习一些特定功能，此属性会阻止 channel 组之间的信息流，并在训练过程中削弱表示，为了克服这个问题，我们应用了 channel shuffle。

Channel shuffle 的想法是，我们希望混合来自不同组 filters 的信息。在下图中，在将第一个分组卷积 GConv1 与 3 个 filters 组应用后，我们得到了特征图。在将此特征图传到第二组卷积之前，我们首先将每组中的通道划分为几个子组，然后将这些子组混合在一起。

<div align=center><img src="/assets/ShuffleNet-2022-04-01-13-25-25.png" alt="ShuffleNet-2022-04-01-13-25-25" style="zoom:100%;" /></div>

经过这样的改组后，照常继续执行第二个分组卷积 GConv2。但是现在，由于 shuffled 层中的信息已经混合在一起，因此基本上将 GConv2 中的每个组与特征图层（或输入层）中的不同子组一起提供。结果允许信息在通道组之间流动，并加强了表示。

### 逐点分组卷积（Pointwise Grouped Convolution）

ShuffleNet 论文还介绍了逐点分组卷积。通常，对于诸如 MobileNet 或 ResNeXt 中的分组卷积，分组操作是在 $3\times3$ 卷积上执行的，而不是在 $1\times1$ 卷积上执行的。

ShuffleNet 论文认为 $1\times1$ 卷积在计算上也很昂贵，建议将分组卷积也应用于 $1\times1$ 卷积。顾名思义，逐点分组卷积执行 $1\times1$ 卷积的分组运算，该操作与分组卷积的操作相同，只不过有一处修改：对 $1\times1$ 的 filters 而不是 $N\times N$ 的 filters（$N> 1$）执行。

在 ShuffleNet 论文中，作者使用了三种类型的卷积：

1. 随机分组卷积（Shuffled Grouped Convolution）
2. 点分组卷积（Pointwise Grouped Convolution）
3. 深度可分离卷积

这样设计的体系结构在保持精度的同时大大降低了计算成本。例如，ShuffleNet 和 AlexNet 的分类错误在实际的移动设备上是差不多的。但是，计算成本已从 AlexNet 中的 720 MFLOP 大幅度降低到 ShuffleNet 中的 40 到 140 MFLOP。ShuffleNet 具有相对较低的计算成本和良好的性能，在用于移动设备的卷积神经网络领域中很受欢迎。
