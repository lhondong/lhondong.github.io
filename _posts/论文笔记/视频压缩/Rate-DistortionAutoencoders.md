---
title: ""
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-5.jpg"
mathjax: true
tags:
  - 笔记
---

# Video Compression With Rate-Distortion Autoencoders

## 摘要

使用深度生成模型来实现有损视频压缩。3D autoencoder with a discrete latent space，以及一个 autoregressive prior used for entropy coding。

简单又高效，比之前的运动补偿和插帧的性能更好。

尝试了不同的设计，frame-based or spatio-temporal autoencoders，以及各种种类的 autogressive prior。

提出了三种 extension：

1. semantic compression，模型被训练为分配更多的比特给感兴趣的对象
2. adaptive compression，模型适用于一个变化有限的领域，例如，自动驾驶汽车的视频，实现该领域的高效压缩
3. multimodal compression，非标准成像传感器，例如 quad cameras

## 一、简介

### 1.1 Motivation

本质上就是具有离散的潜在空间和确定性编码器的 VAE。

VAE 框架特别适合解决有损数据压缩的问题，因为它自然地提供了 rate-distortion 权衡，正好就是 VAE 的两项损失。

为了压缩的目的，一般不会使用 VAE 中的随机编码器（近似后验），因为任何噪声添加到编码中，会增加比特率，而不会改善失真。

尝试了 2D Autoencoder 和 3D Autoencoder，以及各种种类的 autoregressive prior，发现其中效果最好的是 ResNet autoencoder with 3D convolutions, and a temporally-conditioned gated PixelCNN as prior.

### 1.2 Contributions

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
