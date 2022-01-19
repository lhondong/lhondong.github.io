---
title: "Transformer&Self-Supervised"
subtitle: "Transformer&Self-Supervised相关知识"
layout: post
author: "L Hondong"
header-img: "img/post-bg-14.jpg"
tags:
  - Transformer
  - Self-Supervised
---

收藏自知乎[2021年深度学习哪些方向比较新颖，处于上升期或者朝阳阶段，没那么饱和，比较有研究潜力](https://www.zhihu.com/question/460500204/answer/1902640999)，尽可能将 Transformer 和 Self-Supervised 的相关论文读一下。

# Transformer&Self-Supervised

最近火热我比较看好的方向 **Transformer** 和 **Self-Supervised**，我这里举的例子倾向于计算机视觉方向。

最后再补充Zero-Shot和多模态两个方向。

# 1. Transformer

自从去年DETR和ViT出来之后，计算机视觉领域掀起了Transformer狂潮。目前可以做的主要有两个路径，**一个是魔改DETR和ViT，另一个是不同task迁移算法。**

魔改DETR和ViT的方法，无非是引入local和hierarchical，或者魔改算子。

不同task迁移算法主要是探究如何针对不同的task做适配设计。

## 魔改DETR

- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [TSP-FCOS/TSP-RCNN](https://arxiv.org/abs/2011.10881)
- [UP-DETR](https://arxiv.org/abs/2011.09094)
- [SMCA](https://arxiv.org/abs/2101.07448)
- [Meta-DETR](https://arxiv.org/abs/2103.11731)
- [DA-DETR](https://arxiv.org/abs/2103.17084)

## 魔改ViT

### 1. 魔改算子

- [LambdaResNets](https://openreview.net/pdf%3Fid%3DxTJEN-ggl1b)
- [DeiT](https://arxiv.org/abs/2012.12877)
- [VTs](https://arxiv.org/abs/2006.03677)
- [So-ViT](https://arxiv.org/abs/2104.10935)
- [LeViT](https://arxiv.org/abs/2104.01136)
- [CrossViT](https://arxiv.org/abs/2103.14899)
- [DeepViT](https://arxiv.org/abs/2103.11886)
- [TNT](https://arxiv.org/abs/2103.00112)
- [T2T-ViT](https://arxiv.org/abs/2101.11986)
- [BoTNet](https://arxiv.org/abs/2101.11605)
- [Visformer](https://arxiv.org/abs/2104.12533)

### 2. 引入local或者hierarchical

- [PVT](https://arxiv.org/abs/2102.12122v1)
- [FPT](https://arxiv.org/abs/2102.12122)
- [PiT](https://arxiv.org/abs/2103.16302)
- [LocalViT](https://arxiv.org/abs/2104.05707)
- [SwinT](https://arxiv.org/abs/2103.14030)
- [MViT](https://arxiv.org/abs/2104.11227)
- [Twins](https://arxiv.org/abs/2104.13840)

### 3. 引入卷积

- [CPVT](https://arxiv.org/pdf/2102.10882.pdf)
- [CvT](https://arxiv.org/abs/2103.15808)
- [ConViT](https://arxiv.org/abs/2103.10697)
- [CeiT](https://arxiv.org/abs/2103.11816)
- [CoaT](https://arxiv.org/abs/2104.06399)
- [ConTNet](https://arxiv.org/abs/2104.13497)

### 4. 不同task迁移算法的可以参考以下工作：

- ViT+Seg
  - [SETR](https://arxiv.org/abs/2012.15840)
  - [TransUNet](https://arxiv.org/abs/2102.04306)
  - [DPT](https://arxiv.org/abs/2103.13413)
  - [U-Transformer](https://arxiv.org/abs/2103.06104)
- ViT+Det
  - [ViT-FRCNN](https://arxiv.org/abs/2012.09958)
  - [ACT](https://arxiv.org/abs/2011.09315)
- ViT+SOT
  - [TransT](https://arxiv.org/abs/2103.15436)
  - [TMT](https://arxiv.org/abs/2103.11681)
- ViT+MOT
  - [TransTrack](https://arxiv.org/abs/2012.15460)
  - [TrackFormer](https://arxiv.org/abs/2101.02702)
  - [TransCenter](https://arxiv.org/abs/2103.15145)
- ViT+Video
  - [STTN](https://arxiv.org/abs/2007.10247)
  - [VisTR](https://arxiv.org/abs/2011.14503)
  - [VidTr](https://arxiv.org/abs/2104.11746)
  - [ViViT](https://arxiv.org/pdf/2103.15691.pdf)
  - [TimeSformer](https://arxiv.org/abs/2102.05095)
  - [VTN](https://arxiv.org/abs/2102.00719)]
- ViT+GAN
  - [TransGAN](https://arxiv.org/abs/2102.07074)
  - [AOT-GAN](https://arxiv.org/abs/2104.01431)
  - [GANsformer](https://arxiv.org/abs/2103.01209)
- ViT+3D
  - [Group-Free](https://arxiv.org/abs/2104.00678)
  - [Pointformer](https://arxiv.org/abs/2012.11409)
  - [PCT](https://arxiv.org/abs/2012.09688)
  - [PointTransformer](https://arxiv.org/abs/2012.09164)
  - [DTNet](https://arxiv.org/abs/2104.13044)
  - [MLMSPT](https://arxiv.org/abs/2104.13636)

**以上几个task是重灾区（重灾区的意思是听我一句劝，你把握不住）**

- ViT+Multimodal
  - [Fast and Slow](https://arxiv.org/abs/2103.16553)
  - [VATT](https://arxiv.org/abs/2104.11178)
- ViT+Pose
  - [TransPose](https://arxiv.org/abs/2012.14214)
  - [TFPose](https://arxiv.org/abs/2103.15320)
- ViT+SR
  - [TTSR](https://arxiv.org/abs/2006.04139)
- ViT+Crowd
  - [TransCrowd](https://arxiv.org/abs/2104.09116)
- ViT+NAS
  - [BossNAS](https://arxiv.org/abs/2103.12424)  
- ViT+ReID
  - [TransReID](https://arxiv.org/abs/2102.04378)
- ViT+Face
  - [FaceT](https://arxiv.org/abs/2104.11502)

**想一想算子怎么魔改，或者还有什么task没有做的。**

# 2. Self-Supervised

Self-Supervised自从何恺明做出MoCo以来再度火热，目前仍然是最为火热的方向之一。目前可以做的主要有三个路径，**一个是探索退化解的充要条件，一个是Self-Supervised+Transformer探索上限，还有一个是探索非对比学习的方法。**

## 2.1 探索退化解

**探索退化解的充要条件主要是探索无negative pair的时候，避免退化解的最优方案是什么。**

- [SimCLR](https://arxiv.org/abs/2002.05709v3)
- [BYOL](https://arxiv.org/abs/2006.07733)
- [SwAV](https://arxiv.org/pdf/2006.09882.pdf)
- [SimSiam](https://arxiv.org/abs/2011.10566)
- [Twins](https://arxiv.org/abs/2103.03230)

## 2.2 Self-Supervised+Transformer

**Self-Supervised+Transformer是MoCov3首次提出的，NLP领域强大的预训练模型(BERT和GPT-3)都是Transformer架构的，CV可以尝试去复制NLP的路径，探究Self-Supervised+Transformer的上限。**

- [MoCov1](https://arxiv.org/abs/1911.05722)
- [MoCov2](https://arxiv.org/abs/2003.04297v1)
- [MoCov3](https://arxiv.org/abs/2104.02057)
- [SiT](https://arxiv.org/abs/2104.03602)

## 2.3 探索非对比学习

**探索非对比学习的方法就是要设计合适的proxy task。**

### 基于上下文

- [Unsupervised Visual Representation Learning by Context Prediction](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf)
- [Unsupervised Representation Learning by Predicting Image Rotations](https://openreview.net/pdf?id=S1v4N2l0-)
- [Self-supervised Label Augmentation via Input Transformations](https://arxiv.org/pdf/1910.05872.pdf)

### 基于时序

- [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888)
- [Unsupervised Learning of Visual Representations using Videos](http://www.cs.cmu.edu/~xiaolonw/papers/unsupervised_video.pdf)

刚写了基于时序，何恺明和Ross Girshick就搞了个时序的：  
[A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](https://arxiv.org/abs/2104.14558)

[何恺明+Ross Girshick：深入探究无监督时空表征学习](https://zhuanlan.zhihu.com/p/369159211)

# 3. Zero-Shot

最近因为CLIP的出现，Zero-Shot可能会引起一波热潮，ViLD将CLIP成功应用于目标检测领域，相信未来会有越来越多的基于CLIP的Zero-Shot方法。

[ViLD：超越Supervised的Zero-Shot检测器](https://zhuanlan.zhihu.com/p/369464298)

# 4. 多模态

最近的ViLT结合了BERT和ViT来做多模态，并且通过增加标志位来巧妙的区分不同模态，感觉是一个非常好的做多模态的思路，相信未来会有更强大的多模态出现。

[ViLT：最简单的多模态Transformer](https://zhuanlan.zhihu.com/p/369733979)

**最后，适当灌水，有能力还是要做有影响力的工作。**

