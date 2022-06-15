
# Masked Image Modelling

Reading list for research topics in Masked Image Modeling(MIM).

We list the most popular methods for MIM, if we missed something, please submit a request.
(Note: We show the date the first edition of the paper was submitted to arxiv, but the link to the paper may be up to date.)

## Self-supervied Vision Transformers as backbone models

Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2020-xx-xx(maybe 2019)|iGPT|ICML 2020|[Generative Pretraining from Pixels](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)|[iGPT](https://github.com/openai/image-gpt)
2020-10-22|ViT|ICLR 2021 (Oral)|[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)|[ViT](https://github.com/google-research/vision_transformer)
2021-04-08|SiT|Arxiv 2021|[SiT: Self-supervised vIsion Transformer](https://arxiv.org/pdf/2104.03602.pdf)|None
2021-06-10|MST|NeurIPS 2021|[MST: Masked Self-Supervised Transformer for Visual Representation](https://arxiv.org/pdf/2106.05656.pdf)|None
2021-06-14|BEiT|ICLR 2022 (Oral)|[BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)|[BEiT](https://github.com/microsoft/unilm/tree/master/beit)
2021-11-11|MAE|Arxiv 2021|[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)|[MAE](https://github.com/facebookresearch/mae)
2021-11-15|iBoT|ICLR 2022|[iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/pdf/2111.07832.pdf)|[iBoT](https://github.com/bytedance/ibot)
2021-11-18|SimMIM|Arxiv 2021|[SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/pdf/2111.09886.pdf)|[SimMIM](https://github.com/microsoft/SimMIM)
2021-11-24|PeCo|Arxiv 2021|[PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers](https://arxiv.org/pdf/2111.12710.pdf)|None
2021-11-30|MC-SSL0.0|Arxiv 2021|[MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning](https://arxiv.org/pdf/2111.15340.pdf)|None
2021-12-16|MaskFeat|Arxiv 2021|[Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/pdf/2112.09133.pdf)|None
2021-12-20|SplitMask|Arxiv 2021|[Are Large-scale Datasets Necessary for Self-Supervised Pre-training?](https://arxiv.org/pdf/2112.10740.pdf)|None
2022-01-31|ADIOS|Arxiv 2022|[Adversarial Masking for Self-Supervised Learning](https://arxiv.org/pdf/2201.13100.pdf)|None
2022-02-07|CAE|Arxiv 2022|[Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/pdf/2202.03026.pdf)|None
2022-02-07|CIM|Arxiv 2022|[Corrupted Image Modeling for Self-Supervised Visual Pre-Training](https://arxiv.org/pdf/2202.03382.pdf)|None
2022-03-10|MVP|Arxiv 2022|[MVP: Multimodality-guided Visual Pre-training](https://arxiv.org/pdf/2203.05175.pdf)|None
2022-03-23|AttMask|Arxiv 2022|[What to Hide from Your Students: Attention-Guided Masked Image Modeling](https://arxiv.org/pdf/2203.12719.pdf)|None
2022-03-29|mc-BEiT|Arxiv 2022|[mc-BEiT: Multi-choice Discretization for Image BERT Pre-training](https://arxiv.org/pdf/2203.15371.pdf)|None
2022-04-18|Ge2-AE|Arxiv 2022|[The Devil is in the Frequency: Geminated Gestalt Autoencoder for Self-Supervised Visual Pre-Training](https://arxiv.org/pdf/2204.08227.pdf)|None

## 3D

Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2021-11-29|Point-BERT|Arxiv 2021|[Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling](https://arxiv.org/pdf/2111.14819.pdf)|[Point-BERT](https://github.com/lulutang0608/Point-BERT)
2022-03-28|Point-MAE|Arxiv 2022|[Masked Autoencoders for Point Cloud Self-supervised Learning](https://arxiv.org/pdf/2203.06604.pdf)|[Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

## Image generation
Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2022-02-08|MaskGIT|Arxiv 2022|[MaskGIT: Masked Generative Image Transformer](https://arxiv.org/pdf/2202.04200.pdf)|None

## Video
Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2021-12-02|BEVT|Arxiv 2021|[BEVT: BERT Pretraining of Video Transformers](https://arxiv.org/pdf/2112.01529.pdf)|[BEVT](https://github.com/xyzforever/BEVT)
2022-03-23|VideoMAE|Arxiv 2022|[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)|[VideoMAE](https://github.com/MCG-NJU/VideoMAE)

## Multi-modal

Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2022-04-04|MultiMAE|Arxiv 2022|[MultiMAE: Multi-modal Multi-task Masked Autoencoders](https://arxiv.org/pdf/2204.01678.pdf)|[MultiMAE](https://github.com/EPFL-VILAB/MultiMAE)

Masked image modelling can provide competitive results to the other approaches like contrastive learning. Performing computer vision tasks using masked images can be called masked image modelling.

使用掩码图像进行 CV 任务。

In machine learning, nowadays, we can see that the models and techniques of one domain can perform tasks of other domains. For example, models focused on natural language processing can also perform a few tasks related to computer vision. In this article, we will discuss such a technique that is transferable from NLP to computer vision. When applying it to the computer vision tasks, we can call it Masked Image Modelling. We will try to understand the working of this technique along with its important applications.

## What is Masked Image Modelling?

In machine learning, masked signal learning is a type of learning where the masked portion of the input is used to learn and predict the masked signal. We can find the use cases of this type of learning in NLP for self-supervised learning. In many works, we can see the use of masked signal modelling for learning from huge unannotated data. While talking about the computer vision task, this approach can also provide competitive results to the other approaches like contrastive learning. Performing computer vision tasks using masked images can be called masked image modelling.

Applying masked image modelling can have the following difficulties:

- Pixels near to each other are highly correlated.
- Signals under the images are raw and low level in comparison to the signal (tokens) under the NLP data.(CV 中底层语义）
- Signals in image data are continuous while text signals are discrete.

So applying this approach to image or computer vision-related data, requires the procedure to be accomplished very well so that correlation can be avoided. Prediction from the low-level signals can be used for high-level visual tasks and the approach can adapt the continuous signal behaviour. 

We can witness various works which have modelled image data to generalize these difficulties like:

- [Pre-Trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf): This work shows the adoption of continuous signal from image for classification tasks using the colour clustering techniques in addition. 
- [Swin Transformer V2](https://arxiv.org/abs/2111.09883): This work represents the technique for scaling a Swin transformer up to 3 billion parameters and making it capable of learning and performing computer vision tasks with images up to 1536 x 1536 resolution. They have applied the adaptation techniques for continuous signals from images using models.
- [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254): This work can be considered as using the BERT model in computer vision where we can witness a similar technique of tokenization using an additional network for image data and its block-wise image masking breaks the short-range connection between the pixels. 

In the above image, we can see the input patches of images, with a linear layer to perform regression on pixel values of the masked area under loss. The design and insights of a simple model can consist of the following:

- Masking applied to the images
- Model for raw pixel regression
- Lightweight prediction head

By applying simple masking to the images, we can make the process simple for a transformer. The regression task aligns well with the continuous nature of visual signals, and a lightweight prediction head should have the property of bringing a remarkable speedup in pre-training. Heavier heads have the capability of a stronger generation but can lead to a loss in the downstream fine-tuning tasks.

## The framework of Masked Image Modelling

We could understand that the motive of these procedures is to learn representation using the masked image modelling, in which the procedure should be capable of masking a portion of an image signal and predicting the original signals at the masked area. A framework to complete the motivation can have the following components:

- Masking strategy: This component should be designed for selecting the area to mask and to perform the masking on the selected area so that the masked image can be used as an input.
- Encoder architecture: This component should be able to extract latent feature representation for the masked image and use the extracted representation to predict the original signals at the masked area. If using transformers as encoders, then it is expected from the encoder that it should be capable of performing a variety of computer vision tasks. Some examples of transformers in computer vision are vanilla ViT and Swim transformers.
- Prediction head: This component should be capable of producing one form of original signals at the masked area of the image when applied to the latent feature representation learned by the encoder. 
- Prediction target: This component should be capable of defining the form of the prediction from original signals and loss type. Talking about the prediction from it can be either raw pixels or a transformation of the raw pixels. The form of loss type can be a cross-entropy classification or L1 and L2 regression loss. 

For image masking, we can use a variety of strategies of image masking like square shape masking, block-wise masking, random masking, etc. The below image is a representation of the different types of image masking.

## Works Related to Masked Image Modelling

Some of the important works related to masked image modelling are as follows:

- [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V1_ICML.pdf): In this work, we can see an example of a trained sequence transformer to predict pixels using an autoregressive approach. In this work, a GPT-2 model is used to learn strong image representations. This work has achieved 96.3% accuracy with a linear probe, outperforming a supervised Wide ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models.
- [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192): In this work, we can see the use of spatial context from the image as a source of visual signals which can be used for training a visual representation. While extracting random pairs of patches from images, they have trained a CNN to predict the place of the second patch correlated to the first. This work is a representation of unsupervised visual discovery by learning feature representations within images. Also, learning feature representations within images helps in capturing the visual similarity across images.
- [Selfie: Self-supervised Pre Training for Image Embedding](https://arxiv.org/abs/1906.02940): This work generalizes the concept of masked language modelling of BERT to image data. Using the masked image, the method learns to select the correct patch of the image, among other patches sampled from the same image. This work on ImageNet 224 x 224 with 60 examples per class (5%), improves the mean accuracy of ResNet-50 from 35.6% to 46.7%, an improvement of 11.1 points in absolute accuracy.
- [SimMIM: A Simple Framework for Masked Image modelling](https://arxiv.org/pdf/2111.09886.pdf): SimMIM is a very simple framework for masked image modelling. This framework is an example of applying a very light prediction head that can be compared to the linear layers. Using ViT-B, this approach achieves 83.8% top-1 fine-tuning accuracy on the ImageNet-1K dataset. By pre-training also on this dataset using SwinV2-H, it achieves 87.1% accuracy using only ImageNet-1K data.
- [Masked image modelling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling/): This is an example of masked image modelling given by Keras where we can find a simple and effective method to pre-train large vision models like ViT. This method gets inspiration from the pre-training algorithm of BERT. Using this example we can learn how to patch images and predict by learning from extracted patches of images. This can be considered as an implementation of a masked autoencoder for self-supervised pre-training with the CIFAR-10 data.

## Applications of Masked Image Modelling

Performing image masking helps transformers and autoencoders to learn easily using only required information from the images. Masking can speed up the transformer to perform classification tasks using images. Also, masking images is a process of creating an image piece from a larger image and also we can use it to modify a larger image. It is a process that is underneath many types of image processing like edge detection, motion detection, and noise reduction. Mainly, we can say that this technique can be used in self-supervised learning in computer vision. Masked images are easy to learn because of the low and important information in masked images. Due to high-level unannotated data creating confusion for the model, image masking can be considered as a process of converting high dimensional data to a lower dimension. 

### BEiT

Bidirectional Encoder representation from Image Transformers (BEiT)，提出了 Masked Image Modeling 自监督训练任务的概念，以此来对 ViT 进行训练。如算法概览图（下图）所示，BEiT 预训练中，每一张图片有两种视角：一是图像块 (image patches)，如每一小块图像为 16x16 像素；二是离散的视觉标记 (discrete visual tokens)。在预训练过程中，BEiT 先将原始图片标记化，并且对图像块进行随机掩码，并将掩码后的图片输入到编码器当中，主要的预训练目标就是基于未掩码图像块来恢复掩码图像块。

<div align=center><img src="/assets/MIM-2022-04-07-17-12-25.png" alt="MIM-2022-04-07-17-12-25" style="zoom:50%;" /></div>

首先来看图片的表示，图像块和视觉标记。

图像块和 ViT 原文所描述的并无二致，而对于重建目标，BEiT 并没有使用原始的像素，而是通过一个 “image tokenizer” 进行离散化，遵循的是 dVAE 的思路，在 BEiT 预训练之前，先构建 “tokenizer” 和 “decoder” 进行 dVAE 的训练，并构建视觉词汇表。在 BEiT 中是直接采用 Zero-shot text-to-image generation 文章开源的代码进行训练。

对于预训练的主干网络，则是标准的 ViT，每个图像块会被线性投射为对应的 embedding 向量，同时再加上标准的可学习的绝对位置编码。而与之不同的是，BEiT 采用了 Blockwise Masking 的方式，对大约 40% 的图像块进行了掩码操作，而预训练的目标便是期望能够正确预测掩码图像块的视觉标记，从而获得可以提取图像特征向量的编码器。

在下游的分类和分割任务上，BEiT 均超过了之前的自监督算法和有监督训练模式，达到 SOTA 水准。

BEiT 可以说是将 BERT 形式的预训练方式迁移到视觉领域的开山之作，并提出 MIM 预训练的任务概念，为自监督领域做出了重要的贡献。

### MAE

MAE 相比于 BEiT，简化了整体训练逻辑，利用随机掩码处理输入的图像块，以及直接重建掩码图像块来进行训练。MAE 基于两大主要设计：一是采用了非对称结构的编码-解码器，其中编码器只计算非掩码图像块，同时采用了轻量化的解码器设计；二是遮盖大部分的图像块，如掩码概率为 75%，可以获得更加具有意义的自监督训练任务。

<div align=center><img src="/assets/MAE-2022-03-02-18-56-33.png" alt="MAE-2022-03-02-18-56-33" style="zoom:50%;" /></div>

MAE 逻辑和其他的自编码器类似，通过编码器将原始信号映射到特定空间内的隐变量，再基于隐变量通过解码器重建原始信号，但是与传统的自编码器不同的是，MAE 采用非对称的结构和轻量级解码器。

首先看掩码部分，拆分图像块的方式和 ViT 一致，之后再随机遮盖图像的大部分，使其只留下部分可见内容，所以在训练过程中，模型不容易找到捷径解，例如插值等。

其次来看编码器和解码器，编码器即是标准的 ViT 模型，只不过只对非掩码图像块进行计算，从中提取特征，这种设计可以减少计算量和内存；而解码器则会对所有可见的图像块和掩码图像块进行计算，对于每个图像块会加上位置编码信息，以避免图像块的位置信息丢失。由于数据的输入只有掩码图像块以及编码器和解码器的非对称性，两个模块互相独立设计，所以可以大大减少训练时间。

对于重建目标，MAE 针对每个掩码图像块进行像素值预测，并计算 MSE 损失函数。

在下游任务上，作者提出之前的 linear probing 和端到端微调具有很强的不相关性，即使在过去几年内 linear probing 是最受欢迎的下游评价方式。并且对基于 MAE 预训练的和 MoCo v3 所训练的 ViT-L 进行了实验对比，MAE 在 linear probing 中的结果要差于 MoCo v3，但是从部分微调开始，其结果都要比 MoCo v3 要更好（实验结果如下，0 代表 linear probing ，24 则是全量微调）。

<div align=center><img src="/assets/MAE-2022-03-04-20-37-25.png" alt="MAE-2022-03-04-20-37-25" style="zoom:50%;" /></div>

linear probing 遗漏了一些强大但是非线性的特征，而这正是深度学习的优势。例如在 MAE 中便有更加强大的非线性的特征表示，而 linear probing 并不能很好的展示这一点，所以采用全量微调或者部分微调的 MAE 能取得更好的结果。

MAE 凭借简单的训练思路和 SOTA 的结果，在视觉领域是迅速走红，是一个非常漂亮的研究工作。

### SimMIM

SimMIM 提出了一种简单掩码学习框架，相比之前 SOTA，简化了一些特殊的设计，例如 Blockwise Masking、dVAE 的 tokenizer 或聚类等方法；而简单的设计，例如随机掩码、回归 RGB 像素值和采用线性预测头，也可以取得 SOTA 结果。

<div align=center><img src="/assets/MIM-2022-04-07-18-29-30.png" alt="MIM-2022-04-07-18-29-30" style="zoom:50%;" /></div>

SimMIM 认为掩码部分采用随机掩码，并且适当增大图像块的分辨率即可获得很好的结果，并且文章中提供了对比试验的结果，当图像块分辨率为 32x32 时可以获得最好的效果，并将其设置为 ViT 模型的默认设置。和 NLP 领域算法及 BEiT 类似，掩码 token 为可学习的向量，并且维度和其他图像块线性映射后的维度一致。

在本文中，编码器采用了 ViT 和 Swin Transformer。

对于预测头，文章中也分别对四种架构进行了实验，包括线性层、2 层 MLP、inverse Swin-T 和 inverse Swin-B。实验结果表明使用最简单的线性层，花费最少的训练时间，却可以获得最高的准确率，所以最终便采用了最简单的线性层作为预测头，对掩码图像块进行预测。

预测目标为最简单的 L1 损失函数，即可以获得 SOTA 级别的结果。

### MaskFeat

MaskFeat 算法在整体思路上依然是重建掩码图像块的思路，只不过它的重建目标从原始像素值变成了 HOG 特征描述器。通过作者的实验，在五种不同类型的特征描述中，HOG 可使网络获得最好的结果，且训练更加高效，算法总览图如下：

<div align=center><img src="/assets/MIM-2022-04-07-18-32-55.png" alt="MIM-2022-04-07-18-32-55" style="zoom:50%;" /></div>

MaskFeat 证明了可以直接在无标注的视频数据集上进行训练，并且具有非常优秀的迁移性能。因此，视频理解模型可以不再依靠大数据集的预训练，如 ImageNet-21K，这令视频理解类型的任务受益匪浅。

作者在文中列举了五种不同的目标特征，分别为像素值、HOG 特征、dVAE、预训练神经网络提取的特征和伪标签。基于这五种目标特征，进行了实验，图像目标特征的结果如下：

<div align=center><img src="/assets/MIM-2022-04-07-19-19-45.png" alt="MIM-2022-04-07-19-19-45" style="zoom:50%;" /></div>

可以看到，虽然基于预训练模型的特征可以取得稍好的结果，但是其预训练所额外消耗的时间导致整个训练流程过长，所以基于 HOG 的特征可以说是当前实验结果内的最优选择。

MaskFeat 不需要依赖 ImageNet-21K 这类超大型数据集，提高了预训练的效率；另外，选取不同的目标特征进行实验也为后续的视觉自监督提供了一个新的探索方向。