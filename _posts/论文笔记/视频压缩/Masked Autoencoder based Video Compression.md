# Masked Autoencoder based Video Compression

## 摘要

传统的视频编解码框架存在问题，因此使用深度学习来进行视频压缩工作。

现有的端到端视频压缩主要是通过光流，但是现在的光流压缩效果不好。

将Masked Autoencoder用于视频压缩技术，它的压缩效果比之前的深度学习视频压缩框架都要好。在达到相同恢复质量的情况下，我们的工作相比于HEVC压缩效率好多少，相比于之前的工作好多少。

### Contributions

1. 第一个将ViT用于端到端视频压缩。
2. 设计了一个特别好的解码器，恢复效果非常好。
3. An end-to-end learned video compression codec based on GAN-prior generative image compression

A desired compression rate is controlled by the size of latent dimension in the image compression stage as well as the number of quantization levels used in residual encoding

设计不同的压缩率

## Introduction

Traditional video coding standards such as MPEG, AVC/H.264 [49], HEVC/H.265 [43], and VP9 [38] have achieved impressive performance on video compression tasks. However, as their primary applications are human perception driven, those hand-crafted codecs are likely suboptimal for machine-related tasks such as deep learning based video analytic.

rate-distortion trade-off

During recent years, a growing trend of employing deep neural networks (DNNs) for image compression tasks has been witnessed. Prior works [46, 7, 36] have provided theo- retical basis for application of deep autoencoders (AEs) on image codecs that attempt to optimize the rate-distortion trade-off, and they have showed the feasibility of latent representation as a format of compressed signal

深度学习图像压缩进展

## Related Work

1. Deep Learning based Video Compression
2. Vision Transformer
3. Mask Autoencoder

## Method

### Encoder

借鉴了MAE中的编码器，使用Swin Transformer。



## Decoder

为了更好的恢复效果，SwinIR超分，Video Restortion Transformer视频恢复。

设计更好的Decoder。

### Loss Fuction

$$
\mathcal L = 码流 + \lambda 质量
$$

## Experiments

### Datasets

Kinetics dataset [10] and the UGC dataset

### Setting

优化器Adam，学习率$\alpha$。

### Conclusion

我们第一次将ViT用到了视频压缩领域。实现了非常好的效果。