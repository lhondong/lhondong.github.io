---
title: "Deep Blind Video Super-resolution"
subtitle: "盲超分"
layout: post
author: "L Hondong"
header-img: "img/post-bg-5.jpg"
mathjax: true
tags:
  - 视频超分
  - 盲超分
---

# 盲超分 Deep Blind Video Super-resolution

ICCV 2021

南理工，商汤

## 摘要

现有超分算法假设退化过程中的模糊核是已知的，所以在重建过程中不去建模blur kernel。但是这样会导致over-smoothed问题。

提出了一个盲超分模型

1. 从LR帧中估计出模糊核；
2. 用模糊核，设计了一种盲超分算法，使用图像反卷积。

为了有效的利用相邻帧间信息，从LR中估计运动场，提取LR视频帧的特征，扭曲特征。

提出了sharp feature exploration method：

1. 首先，从恢复的中间latent帧中提取sharp特征，
2. 然后使用变换操作，基于提取的sharp特征和扭曲特征，生成更好的特征，用于HR帧重建。

端到端训练的模型，效果SoTA。

## 一、简介

### 1.1 Motivation

HR帧被各种未知的模糊核 contaminated 污染，盲超分就是在不知道模糊核的情况下进行超分。

超分是ill-posed 问题，传统的方法会有一个手工先验，让这个问题 well-posed，估计 latent HR帧，尽管效果还行，但是这些算法通常需要解决复杂的 energy 函数或涉及复杂的匹配过程，性能会被手工先验限制。

另外，他们会假设模糊核已知，如bicubic，然后就不去对模糊核建模，所以不能捕捉到视频帧的固有特点。

从图像超分到视频超分，主要解决 motion fields 和 alignment estimation methods。比如subpixel光流运动补偿，可变形对齐网络，空间对齐网络等。后来又提出了GAN。

重点：**简单地结合去模糊和VSR算法不能有效的解决盲超分问题**。如图所示。

<div align=center><img src="../assets/盲超分Deep Blind Video Super-resolution-2022-01-12-12-34-40.png" alt="盲超分Deep Blind Video Super-resolution-2022-01-12-12-34-40" style="zoom:50%;" /></div>

在图像超分方面，已经有工作使用估计出的模糊核，能很大程度上改善超分效果。然而不能直接用到视频超分，因为虽然同时估计 underlying 运动场和模糊核，但是性能会收到手工图像先验的限制，这些手工设计的图像先验会导致复杂的难以优化的问题。

所以提出盲超分，用Deep CNN 同时估计underlying模糊核，motion fields和latent HR videos，不仅能够避开手工图像先验，同时能准确的估计模糊核和运动场，可以更好的恢复视频。

从LR输入视频中显式估计模糊核，图像反卷积，生成结构清晰的中间帧。基于运动场估计融合从LR视频帧中提取出来的特征，并变换中间latent图像的sharp特征。端到端训练之后可以得到更惊喜的细节。

第一个variational方法深度CNN的盲视频SR算法。

### 1.2 Contributions

1. effective 盲VSR算法，同时估计 模糊核，运动场，和latent图像；
2. 图像反卷积方法，用于VSR的图像生成；
3. 提出sharp特征提取方法，从恢复的中间latent帧中提取sharp特征；
4. 端到端训练网络，性能SoTA

## 二、相关工作

### 2.1 Variational approach

由于ill-posed，所以过去的方法主要是关注**先验**，但是使用已知的模糊核，导致过度平滑的问题。

- 有方法同时估计运动场，模糊核和latent图像，用最大后验估计。
- Liu和Sun，用Bayesian框架做VSR。
- Ma，Expectation Maximization框架，联合VSR和模糊核估计。

但是这些方法都被手工设计的先验信息限制了。

### 2.2 Deep learning approach

## 三、方法 Revisiting Variational Methods

再论变分方法。

通过探索VSR的图像生成，设计了一个紧凑的深度CNN模型。

### 3.1 光流估计算法

<div align=center><img src="../assets/盲超分Deep Blind Video Super-resolution-2022-01-12-12-35-18.png" alt="盲超分Deep Blind Video Super-resolution-2022-01-12-12-35-18" style="zoom:50%;" /></div>

### 3.2 模糊核估计

使用手工设计的先验 $\phi(K)$ 往往会有优化困难的问题。所以使用CNN模型 $\mathcal N_k$ 进行模糊核估计。

使用CNN估计出来的模糊核视觉上更好，shaper。第6节有证明。

### 3.3 潜在帧恢复

不再专注于设计复杂的图像先验，而是首先恢复一个具有清晰细节的中间潜在HR帧，然后用深度CNN模型来探索这些恢复的清晰结构细节。

## 四、实验

<div align=center><img src="../assets/盲超分Deep Blind Video Super-resolution-2022-01-12-12-36-11.png" alt="盲超分Deep Blind Video Super-resolution-2022-01-12-12-36-11" style="zoom:50%;" /></div>

## 五、总结

