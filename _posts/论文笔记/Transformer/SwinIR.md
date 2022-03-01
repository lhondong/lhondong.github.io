---
title: "论文笔记模板"
subtitle: "论文笔记模板"
layout: post
author: "L Hondong"
header-img: "img/post-bg-3.jpg"
mathjax: true
tags:
  - Template
  - 笔记
---

# SwinIR: Image Restoration Using Swin Transformer

SwinIR 由三部分组成：浅层特征提取，深层特征提取和高质量图像重建。特别的，深层特征提取模块由一系列的残差 Swin Transformer 块组成，每个 RSTB 包含一些 Swin Transformer 层及一个残差连接。

- 浅层特征提取模块使用 CNN 提取浅层特征，这些特征最后会直接传递到重建模块以便保留低频信息；
- 深层特征提取模块主要由 RSTB 模块组成，每个 RSTB 块使用多个 Swin Transformer 层用于局部注意力和窗口间交互；
- 在模块的最后增加了一个卷积层用来做特征增强，并且使用了一个残差连接用来提供特征融合的 shortcut。

最终，浅层和深层特征融合进重建模块重建高质量图像。相比于流行的基于 CNN 的图像复原模型，基于 Transformer 的 SwinIR 有如下几个好处：

1. 图像内容和注意力权重之间基于内容的交互，可以被解释为空间变化卷积。
2. 移动窗口机制有助于长范围依赖的建模。
3. 更少的参数，更好的性能，相比于已存在的图像复原模型，SwinIR 使用更少的参数得到的更好的 PSNR。

## 摘要

## 一、简介

### 1.1 Motivation

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
