# Transformer-based Transform Coding

ICLR 2022

Qualcomm AI Research

## 摘要

总结过去的神经网络视频编码方法主要是依赖于 prior models, quantization methods and nonlinear transforms.但是现在为了达到更好的 rate-distortion performance，大家用了更先进的网络，导致解码太慢。

本文基于 Swin-transformer 构建better rate-distortioncomputation trade-off的模型，效果比ConvNets的效果更好，requiring fewer parameters and shorter decoding time。

## 一、简介

### 1.1 Motivation

Transform coding 可以分为三个过程：transform, quantization, and entropy coding，其中：

- autoencoder networks are adopted as flexible nonlinear transforms
- deep generative models are used as powerful learnable entropy models
- various differentiable quantization schemes are proposed to aid end-to-end training

然而，基于上下文先前建模的率失真提升在解码复杂度方面往往要付出高昂的代价。

nonlinear transforms 一直被低估了，提出问题：

- can we achieve the same performance as that of expensive prior models by designing a more expressive transform together with simple prior models? 
- And if so, how much more complexity in the transform is required?

事实证明：不仅可以用简单的先验模型构建神经网络编码器，这些模型的性能优于昂贵的空间自回归先验模型，而且与其卷积模型相比，这样做的transform复杂度更小，实现严格的更好的失真复杂性权衡。

### 1.2 Contributions

1. extend SwinTransformer (Liu et al., 2021) to a decoder setting and build Swin-transformer based neural image codecs that attain better rate-distortion performance with lower complexity compared with existing solutions，使用 Swin-Transformer 构建图像编解码器，性能更好，复杂度更低
2. 通过增强SSF验证其有效性
3. 探讨卷积和Transformer的差异，并研究编码增益的潜在来源

## 二、相关工作

### 2.1 

### 2.2 

## 三、方法

### 3.1 

### 3.2 

## 四、实验

### 4.1 

### 4.2 

## 五、总结
