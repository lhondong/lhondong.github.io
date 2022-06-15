
# ConvMAE: Masked Convolution Meets Masked Autoencoders 

主要提出多尺度的混合 Convolution-Transformer 模型可以助力 Masked Auto-Encoding (MAE) 的训练范式，帮助其学习到更好的表征。

## Self-supervised Learning  

在预训练阶段我们使用无标签的数据集 (unlabeled data)，因为有标签的数据集很贵。相反，无标签的数据集网上很多。在训练模型参数的时候，我们不追求把这个参数用带标签数据从初始化的一张白纸给一步训练到位，原因就是数据集太贵。于是 Self-Supervised Learning 就想先把参数从一张白纸训练到初步成型，再从初步成型训练到完全成型。

注意这是 2 个阶段。这个训练到初步成型的东西，我们把它叫做 Visual Representation。预训练模型的时候，就是模型参数从一张白纸到初步成型的这个过程，还是用无标签数据集。等把模型参数训练个八九不离十，这时候再根据你下游任务 (Downstream Tasks) 的不同去用带标签的数据集把参数训练到完全成型，那这时用的数据集量就不用太多了，因为参数经过了第 1 阶段就已经训练得差不多了。  

第一个阶段不涉及任何下游任务，就是拿着一堆无标签的数据去预训练，没有特定的任务，这叫做：in a task-agnostic way。第二个阶段涉及下游任务，就是拿着一堆带标签的数据去在下游任务上 Fine-tune，这叫做：in a task-specific way。  
以上就是 Self-Supervised Learning 的核心思想。

Self-Supervised Learning 方法可分为 3 类：Data Centric, Prediction （也叫 Generative) 和 Contrastive。

其中的主流就是基于 Generative 的方法和基于 Contrastive 的方法。基于 Generative 的方法主要关注的重建误差，比如对于 NLP 任务而言，一个句子中间盖住一个 token，让模型去预测，令得到的预测结果与真实的 token 之间的误差作为损失。基于 Contrastive 的方法不要求模型能够重建原始输入，而是希望模型能够在特征空间上对不同的输入进行分辨。

<div align=center><img src="/assets/ConvMAE-2022-05-24-18-15-22.png" alt="ConvMAE-2022-05-24-18-15-22" style="zoom:50%;" /></div>

<div align=center><img src="/assets/ConvMAE-2022-05-23-22-14-08.png" alt="ConvMAE-2022-05-23-22-14-08" style="zoom:50%;" /></div>

## ConvMAE 的动机

ConvMAE 这个方法所基于的论点是：目前已经有许多工作 （如 MoCo[1]，MAE[2]，BEiT[3]，DINO[4]) 验证了 MAE Self-Supervised Learning 的训练范式能够帮助释放 Vision Transformer 模型的潜力，并且在下有任务上取得非常好的性能。

MAE 作为这个范式的代表作，开发了一个非对称编码器 - 解码器架构，其中编码器只对可见的 patch 子集进行操作 （即没有被 mask 掉的 token)，另一个非对称的解码器可以从潜在表征和被 masked 掉的 token 重建原始图像。Decoder 的架构可以是十分轻量化的模型，且具体的架构对模型性能影响很大。作者进一步发现，Mask 掉大部分输入图像 （例如 75%) 会产生重要且有意义的自监督任务。同时 MAE 这种训练的范式不但能够在不需要超大规模数据集 (JFT-300M，ImageNet-22K) 的情况下，学习到判别性能很强 (Discriminative) 的表征，而且可以轻松的扩展 (Scalable) 到更大的模型上，并且通过实验发现随着模型增大，效果越来越好。

为了加速 ViT 训练并得到更好的性能，大量工作验证了局部的归纳偏置 (local inductive bias) （如 SMCA-DETR [5]，SAM-DETR[6]，DAB-DETR[7]，Uniformer[8]，CoAtNet[9]，ConViT[10]，Early Convolution[11]) 和可以进一步帮助提升 ViT 模型的性能。同时，这种性能的提升也可以通过多尺度的金字塔式架构 (multi-scale hierarchical representation) （如 Swin Transformer[12]，PVT[13]) 来实现。二者结合的有效性已经在大量的识别，检测，分割的监督学习任务中得到的验证。

所以一个自然而然的问题是：这种多尺度的金字塔式架构 + 局部的归纳偏置的模型，能不能经过 MAE 的训练方式之后，进一步挖掘和提升 MAE 的性能？ 

本文就是探索这个问题。ConvMAE 简而言之就是：多尺度的金字塔式架构 + 局部的归纳偏置的模型，使用 MAE 的 Self-supervised Learning 的训练方式。

与 MAE-Base 相比，ConvMAE-Base 将 ImageNet-1k 的微调精度提高到 85.0% (+1.4%)，将 Mask-RCNN COCO 检测任务的 AP box 提高到 53.2% (+2.9%)，将 UperNet 的 ADE20k 分割任务的 mIoU 提高到 51.7% (+3.6%)。

## ConvMAE Encoder 架构

MAE 是一种以自监督的方式，以 ViT 为模型架构进行预训练的框架。MAE 的方法很简单：Mask 掉输入图像的随机的 patches 并重建它们。

ConvMAE 相比于 MAE 框架做了一些微小却非常有效的改进，如前文所述它的特点是：多尺度的金字塔式架构 + 局部的归纳偏置的模型。

如下图所示是 ConvMAE 框架，它也有一个 Encoder 和 Decoder。Encoder 是 convolution-transformer 混合架构，Decoder 是纯 Transformer 架构。

<div align=center><img src="/assets/ConvMAE-2022-05-23-22-16-59.png" alt="ConvMAE-2022-05-23-22-16-59" style="zoom:50%;" /></div>

先看左上角灰色的 Encoder 部分。它包括了 3 个 stage，设 H 和 W 是输入图片的尺寸，每个 stage 输出的特征分别是 $\frac{H}{4}\times\frac{W}{4}, \frac{H}{8}\times\frac{W}{8}, \frac{H}{16}\times\frac{W}{16}$。前两个 stage 是卷积模块，使用 Masked Convolutional Block 对特征进行操作，其结构如下图右下角所示 （其中的 Depthwise Convolution 使用 5×5 大小卷积核）。在每个阶段之间，进行一次 stride 为 2 的卷积以进行下采样操作。最后一个 stage 都是 Transformer 模块，拉大感受野，并融合所有 patch 的特征。另外作者发现绝对位置编码性能是最优的。  

## ConvMAE mask 策略

MAE 对输入图片的 patch 采用随机 mask 策略，然而，同样的策略不能直接应用于 ConvMAE 的编码器。因为 ConvMAE 的特征是不同 stage 是逐渐下采样的，如果在 的特征这里进行了随机的 mask，就会导致 stage3 阶段的每个 tokens 都有一部分的可见信息。因此 ConvMAE 的做法是 mask 掉 stage3 的输出 （比如 75%) 之后，把这些 mask 分别上采样 2 倍和 4 倍得到前两个阶段的 mask。这些被 mask 掉的 token 在编码阶段被丢弃，并且希望经过 Decoder 之后能够重建出来。通过这种方式，ConvMAE 只需要保留至少 25% 的 token 用于训练。

但是前两个阶段使用 5×5 的 Depthwise Convolution 的感受野可能大于一个 masked patch 的大小，因此作者为了确保预训练的质量，在前两个阶段采用了 masked convolution[14][15]，确保被 mask 掉的部分不会参与到编码的过程。

## ConvMAE Decoder 架构

原始 MAE 的 Decoder 以 Encoder 的输出以及 masked token 为输入，通过一系列的 Tranformer Block 得到最终的重建结果。

ConvMAE 的编码器获得了多尺度特征 E1, E2, E3，分别捕捉到了细粒度和粗粒度的图像信息，为了更好的进行训练，将 E1 和 E2 分别进行 stride=2 和 stride=4 的下采样之后与 E3 相加，进行多尺度特征的融合，得到的结果在通过 Linear Transformation 得到最终要输入给 Decoder 的 token。

$$
E_d = \text{Linear}(\text{StrideConv}(E_1,4) + \text{StrideConv}(E_2, 2) + E_3)
$$

式中，$\text{StrideConv}(\cdot,k)$ 代表 stride=k 的卷积。

训练使用的目标函数与 MAE 保持一致，都是 mask 的部分的重建结果与原图的 L2 Loss。

$$
\mathcal{L}=\frac{1}{T_M}\sum\limits_{t\in T_M}\vert I(t)-\hat I(t)\vert^2
$$

式中，$T_M$ 代表 masked tokens 的集合。

ConvMAE 经过预训练之后，Encoder 能够输出多尺度的特征（$\frac{H}{4}\times\frac{W}{4}, \frac{H}{8}\times\frac{W}{8}, \frac{H}{16}\times\frac{W}{16}$)，它们可以被用于后续的检测分割任务。

<div align=center><img src="/assets/ConvMAE-2022-05-24-09-17-50.png" alt="ConvMAE-2022-05-24-09-17-50" style="zoom:50%;" /></div>

ConvMAE 用于检测任务的微调过程：先把 Encoder 的输出特征 E3 进行 max-pooling 操作得到 E4。对于检测任务，因为 ConvMAE 的 stage3 有 11 个全局 Self-attention 层，计算成本过高，所以把 stage3 里面第 1,4,7,11 个 Self-attention 换成了 7×7 Window size 的 Swin Attention 层。通过这样的做法减少了计算量和 GPU 占用。最终得到的 E1, E2, E3, E4 被送入 Mask R-CNN 或者 UperNet 进行目标检测或者语义分割任务。对于分割任务，Stage3 的架构不变。

## 实验结果

首先使用 ImageNet 训练 ConvMAE 框架，mask 掉 25% 的 input token 进行训练，Decoder 的具体架构是一个 8 层的 Transformer, hidden dimension 是 512，head 数是 12。一共预训练 1600 Epoch，使用 cosine 的学习率衰减策略以及 40 Epoch 的学习率 warm up。使用 AdamW 作为优化器，使用 1.5e-4 的初始学习率，0.05 的 weight decay, batch ize 设置为 1024。

预训练时使用 Random cropping 作为数据增强策略，预训练之后，使用 ImageNet-1K 进行监督学习 100 个 Epoch，依然使用 cosine 的学习率衰减策略。

通过 300 Epoch 的预训练，BEiT 可以达到 83.0% 的 Finetuning Accuracy 以及 37.6% 的Linear Probe Accuracy。与 BEiT 相比，ConvMAE 只使用了 25% 的图像和一个更加轻量化的 Decoder，可以达到 89.6% 的 Finetuning Accuracy 以及 69.4% 的 Linear Probe Accuracy。
