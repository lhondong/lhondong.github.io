# Hierarchical Autoregressive Modeling for Neural Video Compression

ICLR 2021

UC Irvine, Caltech

## 摘要

Sequential density estimation by combining masked autoregressive flows with hierarchical latent variable models. 基于掩码自回归流和层级潜可变模型的序列密度估计最近效果很好。

连接了 between such autoregressive generative models and the task of lossy video compression。

将之前的一些视频压缩算法，看作是 instances of generalized stochastic temporal autoregressive transform 广义上随机时间自回归变换的实例，并基于这一见解提出了增强的方法。

大规模视频数据实验表明，这种方法rate-distortion performance 很好。

## 一、简介

### 1.1 Motivation



### 1.2 Contributions

1. A new framework

## 二、相关工作

### 2.1 

### 2.2 

## 三、方法

### 3.1 

$$
\mathcal{L}=\mathcal{D}\left(\mathbf{x}_{1: T}, g\left(\left\lfloor\overline{\mathbf{z}}_{1: T}\right\rceil\right)\right)+\beta \mathcal{R}\left(\left\lfloor\overline{\mathbf{z}}_{1: T}\right\rceil\right)
$$

$$
\tilde{\mathcal{L}}=\mathbb{E}_{q\left(\mathbf{z}_{1: T} \mid \mathbf{x}_{1: T}\right)}\left[-\log p\left(\mathbf{x}_{1: T} \mid \mathbf{z}_{1: T}\right)-\log p\left(\mathbf{z}_{1: T}\right)\right]$$

$$
\mathbf{x}_{t}=h_{\mu}\left(\mathbf{x}_{<t}\right)+h_{\sigma}\left(\mathbf{x}_{<t}\right) \odot \mathbf{y}_{t} ; \Leftrightarrow \mathbf{y}_{t}=\frac{\mathbf{x}_{t}-h_{\mu}\left(\mathbf{x}_{<t}\right)}{h_{\sigma}\left(\mathbf{x}_{<t}\right)}
$$

$$
\hat{\mathbf{x}}_{t}=h_{\mu}\left(\hat{\mathbf{x}}_{t-1}, \mathbf{w}_{t}\right)+h_{\sigma}\left(\hat{\mathbf{x}}_{t-1}, \mathbf{w}_{t}\right) \odot g_{v}\left(\mathbf{v}_{t}, \mathbf{w}_{t}\right)
$$

$$
\hat{\mathbf{x}}_{t}=g\left(\mathbf{z}_{t}, \hat{\mathbf{x}}_{t-1}\right)=h_{\mu}\left(\hat{\mathbf{x}}_{t-1}\right)+h_{\sigma}\left(\hat{\mathbf{x}}_{t-1}\right) \odot \mathbf{y}_{t}, \quad \mathbf{y}_{t}=g_{z}\left(\mathbf{z}_{t}\right)
$$

$$
\hat{\mathbf{x}}_{t}=h_{w a r p}\left(\hat{\mathbf{x}}_{t-1}, g_{w}\left(\mathbf{w}_{t}\right)\right)+g_{v}\left(\mathbf{v}_{t}, \mathbf{w}_{t}\right)
$$

### 3.2 

## 四、实验

### 4.1 

### 4.2 

## 五、总结

