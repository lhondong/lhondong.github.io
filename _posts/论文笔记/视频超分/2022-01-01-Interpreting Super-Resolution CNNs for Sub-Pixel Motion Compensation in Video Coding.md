---
title: "Interpreting Super-Resolution CNNs for Sub-Pixel Motion Compensation in Video Coding"
subtitle: "可解释性"
layout: post
author: "L Hondong"
header-img: "img/post-bg-3.jpg"
mathjax: true
tags:
  - 视频超分
  - 视频编码
  - 可解释性
---

# Interpreting Super-Resolution CNNs for Sub-Pixel Motion Compensation in Video Coding

MM 2021

BBC Research & Development, Dublin City University

## 摘要

提出了基于卷积神经网络的 open-source software，改进了 fractional precision 运动补偿所需的参考样本的插值。

与之前的工作相比，网络全部是线性的，可解释。

使用从训练模型导出的全插值滤波器集，易于集成到传统视频编码方案中，在 VVC 中测试表明，与全神经网络插值相比，本文的学习插值的复杂性大大降低，同时在低分辨率视频序列上实现显著的编码效率改进。

## 一、简介

### 1.1 Motivation

亚像素插值。

由于超分辨率过程是一种插值，因此超分辨率 CNN 已成功地用于生产新的高效插值滤波器。然而，大多数提出的方法都是在 HEVC 中实现的，并且具有非常高的计算要求 [10,13]，在 CPU 环境中测试时，编码复杂度增加了 380 倍 [12]。

本文提出一种基于解释训练好的线性超分网络的方法，以导出低复杂度的插值滤波器，第一种在 VVC 框架中改进编码的方法，显著降低了计算需求。

### 1.2 Contributions

## 二、相关工作

### 2.1 基于 CNN 的亚像素运动补偿

现代视频编码器通常使用一组固定的插值滤波器来细化帧间预测运动补偿的预测样本，在半像素或者四分之一像素上插值采样，高级视频编码器使用多个过滤器集，以便更好地概括不同的视频内容。除了基于离散余弦变换（DCT）的常见滤波器外，VVC 还为仿射运动引入了一个交替的半像素插值滤波器和一个单独的滤波器集 [3]。

基于学习的方法，使用 CNN 生成额外的亚像素

### 2.2

### 2.3

## 三、方法

### 3.1

### 3.2

### 3.2

## 四、实验

## 五、总结

提出了基于线性 CNN 的亚像素插值滤波器的设计方法，将线性网络结构的层折叠成二维矩阵，可以直接提取经过训练的滤波器，并将其作为可换的插值滤波器集成到 VVC 中。

实验结果表明，编码器和解码器的运行时间大大缩短，在改进的 VVC 编码器环境中，低分辨率序列的编码显著节省，从而使基于神经网络的插值滤波器在视频编解码器中可能得到实际应用。

## 六、思考