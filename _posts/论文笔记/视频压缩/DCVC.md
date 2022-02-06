---
title: "Deep Contextual Video Compression"
subtitle: "DCVC"
layout: post
author: "L Hondong"
header-img: "img/post-bg-46.jpg"
mathjax: true
tags:
  - 笔记
---

# DCVC

Deep Contextual Video Compression

[NIPS 2021]

## 摘要

过去的神经网络视频编码方法还是用的传统的思路，先生成预测帧，然后编码当前帧的残差。然而预测编码可能只是一个次优解，因为只使用了简单的减法去除冗余。

从预测编码到条件编码。

提出一个问题：

> How to define, use, and learn condition under a deep video compression framework.

设计了一种高效的条件编码框架，将时域（特征域 feature domain）上下文特征作为条件输入去帮助编解码器编码当前帧，从而充分挖掘条件编码的潜力。而这种设计也便于充分利用高维特征来帮助视频高频细节获得更好的重建质量。

与此同时，DCVC 是一个拓展性非常强的框架，其里面的上下文特征可以灵活设计。实验表明，在标准 1080p 视频上，所提出的 DCVC 相比 x265 (veryslow) 获得了 26.0% 的码率节省（PSNR 为指标）。在 DCVC 下，最新的方法相比 H.265-HM 有 14.4% 的码率节省（PSNR 为质量评价指标）。如果以 MS-SSIM 为质量评价指标，相比 H.266-VTM 则有 21.1% 的码率节省。

## 一、简介

### 1.1 Motivation

从 1988 年的 H.261 到 2020 年发布的 H.266，近 30 年来所有传统的视频编码标准都是基于残差编码的框架。在残差编码中，预测帧先会从之前已经解码的帧中生成出来，然后再计算当前帧与预测帧的残差。该残差会被编码变成码流，解码器将码流解码并获得重建后的残差，最后和预测帧相加获得解码帧。残差编码是一种简单高效的方式，但它的熵大于或等于条件编码的熵，并不是最优的方式。

因为给定预测帧 $\tilde{x}_t$，编码当前帧 $x_t$，

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

## 六、思考
