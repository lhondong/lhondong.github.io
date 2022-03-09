---
title: "Axiomatic Attribution for Deep Networks"
subtitle: "Integrated Gradient"
layout: post
author: "L Hondong"
header-img: "img/post-bg-4.jpg"
mathjax: true
tags:
  - 图像分类
  - 可解释性
---

# Axiomatic Attribution for Deep Networks

Mukund Sundararajan, Ankur Taly, Qiqi Yan

Google

## 摘要

本文研究的问题是，将深度网络的预测结果归因到输入的特征中。

主要内容涉及网络输出和输入特征归因（attribution）的两个公理 fundamental axioms: **Sensitivity and Implementation Invariance** 和 **Integrated Gradients** 方法，包含补充的命题证明，并基于这两个公理设计了一种新的归因方法，称为积分梯度法 (Integrated Gradients)。

经证明，之前的许多归因方法并不完全满足这两条公理。积分梯度法 (Integrated Gradients) 使用起来很简单，不需要修改原始模型的结构。

论文在图像、文本等模型上进行了实验，证明了积分梯度法在 debug 神经网络并从网络中提取规则的能力。

## 一、简介

### 1.1 Motivation

#### 人类怎么做归因

人类做归因通常是依赖于反事实直觉。当人类将某些责任归因到一个原因上，隐含地会将缺失该原因的情况作为比较的基线 (baseline)。例如，想睡觉的原因是困了，那么不困的时候就不想睡觉。

#### 深度网络的归因

基于人类归因的原理，深度网络归因也需要一个基线 (baseline) 输入来模拟原因缺失的情况。在许多深度网络中，输入空间天然存在着一个 baseline。例如，在目标识别网络中，纯黑图像就是一个基线。下面给出深度网络归因的正式定义：

> $\textbf{Definition 1.}$ Formally, suppose we have a function $F:\mathbb{R}^n\to [0,1]$ that represents a deep network, and an input $x=(x_1,\cdots,x_n)\in \mathbb{R}^n$. An attribution of the prediction at input $x$ relative to a baseline input $x_0$ is a vector $A_F(x,x_0)=(a_1,\cdots,a_n)\in \mathbb{R}^n$ where $a_i$ is the contribution of $x_i$ to the prediction $F(x)$.
> 
> 假设存在一个函数 $F:\mathbb{R}^n\to [0,1]$，其表示一个神经网络。该网络的输入是 $x=(x_1,\cdots,x_n)\in \mathbb{R}^n$，那么 $x$ 相较于基线输入 $x_0\in \mathbb{R}^n$ 的归因是一个向量 $A_F(x,x_0)=(a_1,\cdots,a_n)\in \mathbb{R}^n$ ，其中 $a_i$ 是输入 $x_i$ 对预测结果 $F(x)$ 的贡献。

即对于一个从 $R_n$ 到实数域的映射，在 $R_n$ 找到一个向量 a 作为输入向量 $x$ 的归因（attribution）。注意，在一个 baseline 的情况下，作者说明：

> $\textbf{Remark 1.}$ Let us briefly examine the need for the baseline in the definition of the attribution problem. A common way for humans to perform attribution relies on counterfactual intuition. When we assign blame to a certain cause we implicitly consider the absence of the cause as a baseline for comparing outcomes. In a deep network, we model the absence using a single baseline input. For most deep networks, a natural baseline exists in the input space where the prediction is neutral. For instance, in object recognition networks, it is the black image. The need for a baseline has also been pointed out by prior work on attribution (Shrikumar et al., 2016; Binder et al., 2016).

即 baseline 是必要的，它总是显式地或隐式地出现。

#### 归因的意义

首先，在使用图像神经网络预测病情的场景中，归因能够帮助医生了解是哪部分导致模型认为该患者生病了；其次，可以利用深度网络归因来为基于规则的系统提供洞见；最后，还可以利用归因来为推荐结构提供依据。

## 二、Two Fundamental Axioms

### 2.1 Axiom: Sensitivity(a)

> An attribution method satisfies Sensitivity(a) if for every input and baseline that differ in one feature but have different predictions then the differing feature should be given a non-zero attribution.
> 
> 如果对于所有仅在一个特征上具有不同取值的输入 (input) 和基线 (baseline)，并且模型为两者给出了不同的预测，那么，这个不同取值的特征应该被赋予一个非 0 归因。若一个归因方法满足上面的要求，则称该归因方法满足 Sensitivity(a)。

即输入和 baseline 在某个特征不同时，且输出不同，则该特征应该有一个非零的归因。

#### 直接使用梯度 (gradients) 是一个好的归因方法吗？

在线性模型中，如果要 debug 预测结果，模型开发人员仅需要检测模型系数和输入特征的乘积即可。对于深度网络来说，梯度可以看做是线性模型系数的类似物。因此，看起来将梯度和输入特征的乘积作为归因方法的起点是一个合理的选择。但是，梯度违反了 Sensitivity。

举例来说，一个单变量 ReLU 网络：

$$
f(x)= 1 - ReLU(1 - x)= 
\begin{cases}
x, x<1 \\
1, x \geq 1
\end{cases}
$$

假设基线 (baseline) 为 $x=0$ 并且输入 $x=2$，那么显然 $f(0)=0,f(2)=1$。现在来检测梯度是否违反了 Sensitivity。

首先，输入 $x=2$ 和基线 $x=0$ 不同；其次，输出 $f(x=2)=1$ 与基线 $f(x=2)=0$ 也不同；不违反 Sensitivity 的归因方法应该为输入 $x$ 归因一个非 0 值，但是梯度在 $x=2$ 处为 0。因此，这个例子中梯度违反了 Sensitivity。

直觉上解释上面问题的原因就是，预测函数可能在输入处是平坦的，那么不论其与基线有多大的差异，其梯度均为 0，这也导致缺乏敏感性。

例如，基于 ReLU 的梯度的方法使用梯度乘特征来作为该特征的归因，会违反此公理。例如 $y=1-ReLU(1-x)$，在 $x>1$ 时，梯度为零，则归因（因为有梯度作为因子）为零，而在 0 和 2 时，输出为 0 和 1；

再如基于 BP 的方法，如 DeConvNets 和 Guided back-propagation，和基于梯度的方法类似，由于 ReLU 的存在使得不满足 sensitivity。

DeepLift 和 LRP(Layer-wise relevance propagation) 则使用基线来解决 Sensitivity 问题，其使用某种意义上的“离散梯度”来代替原始的连续梯度，并通过一个大的、离散的 step 来避免平坦的区域，从而避免破坏 Sensitivity。但是，DeepLift 和 LRP 违反了 Implementation Invariance（下一个公理）。

### 2.2 Axiom: Implementation Invariance

> Two networks are functionally equivalent if their outputs are equal for all inputs, despite having very different implementations. Attribution methods should satisfy Implementation Invariance, i.e., the attributions are always identical for two functionally equivalent networks.
> 
> 如果两个网络对所有的输入均有相同的输出，则称这两个网络 functionally equivalent（即忽略了网络的实现）。
> 
> Implementation Invariance: 一个归因方法对于两个 functionally equivalent 网络的归因总是一致的，那么称该归因方法满足 Implementation Invariance。

即归因只和模型输入输出的函数相关，不应由实现的差异而不同。

例如，某些模型架构，可能存在参数和函数非单射的情况，导致因为初始化或其他情况下，同一个函数有不同的参数，则归因方法不应该由于参数不同而归因结果不同。如 LRP、DeepLIFT 方法。

注：作者这个 implementation invariance 应该是指，如果两个模型实际是同一个映射（输入输出对应关系相同），那么解释应该一样，例如，使用梯度（输出直接对输入求导）作为解释是满足的（因为直接对函数求导）。作者可能主要是想要排除一种情况：比如同一个映射的两个模型，根据链式方法乘起来结果（输出对输入的梯度）是一样的，但是中间的状态不一样（链式法则相乘的项数个数、值），如果用中间的状态解释，使用了这些不同的中间项，可能会导致解释不同。

#### 梯度满足 Implementation Invariance

梯度的链式法则 $\frac{\partial{f}}{\partial{g}}=\frac{\partial{f}}{\partial{h}}\cdot \frac{\partial{h}}{\partial{g}}$ 本质上是满足 Implementation Invariance。将 $g$ 看做是一个网络的输入，$f$ 是网络的输出，$h$ 是实现网络的细节，那么 $f$ 对 $g$ 的梯度可以直接通过 $\frac{\partial{f}}{\partial{g}}$ 来计算，也可以通过包含 $h$ 的链式法则来实现。也就是说，最终的梯度结果并不依赖于中间的细节 $h$。

#### 离散梯度不满足 Implementation Invariance

DeepLift 和 LRP 使用离散梯度来替代梯度，并使用改进的反向传播来计算归因中的离散梯度。但是，离散梯度通常不满足链式法则，因此这些方法不满足 Implementation Invariance。

## 三、积分梯度 Integrated Gradients

令函数 $F:\mathbb{R}^n\to [0,1]$ 表示深度网络， $x\in \mathbb{R}^n$ 表示输入， $x_0\in \mathbb{R}^n$ 表示基线输入。那么 $x$ 的第 $i$ 个分量的归因，可以看做是基线 $x_0$ 到输入 $x$ 的直线路径上所有梯度的累计。即分量 $i$ 的归因是 $x_0$ 到 $x$ 直线上的梯度路径积分，正式地定义为：

$$\text {IntegratedGrad}_{i}(x)::=(x_{i}-x_{i}^{\prime}) \times \int_{\alpha=0}^{1} \frac{\partial F\left(x^{\prime}+\alpha \times\left(x-x^{\prime}\right)\right)}{\partial x_{i}} d \alpha$$

其中，$\frac{\partial F(x)}{\partial x_{i}}$ 是 $F(x)$ 在第 $i$ 维度的梯度。

### 3.1 Axiom: Completeness

> Integrated gradients satisfy an axiom called completeness that the attributions add up to the difference between the output of $F$ at the input $x$ and the baseline $x_0$.

公理 Completeness 指的是所有的归因相加等于 $F(x)-F(x_0)$，即所有的归因值加起来等于输入和 baseline 输入的输出之差。

#### 积分梯度法满足 Completeness

对于文中定义的 IntegratedGrads，满足 Completeness：

> $\textbf{Proposition 1.}$ If $F:\mathbb{R}^n\to\mathbb{R}$ is differentiable almost everywhere then: $\sum^n_1 \text{IntegratedGrads}_i(x)=F(x)-F(x_0)$
> 
> 如果函数 $F:\mathbb{R}^n\to [0,1]$ 几乎处处可微，那么有 $\sum^n_1 \text{IntegratedGrads}_i(x)=F(x)-F(x_0)$

对于许多深度网络来说，选择一个预测结果接近 0 的基线是有可能的 ($F(x_0)\approx 0$)。这样的话，解释归因结果时就可以忽略基线，仅将归因的结果分配到输入上（不考虑基线）。

神经网络由有限个不可微点的函数复合而成，是几乎处处可微的实函数。

定义 $\gamma=(\gamma_1,\cdots,\gamma_n): [0,1]\to\mathbb{R}^n$ 为一路径，在这里 $\gamma(\alpha)=(\gamma_1(\alpha),\cdots,\gamma_n(\alpha))=x_0+\alpha(x-x_0)$，于是由链式法则有：

$$\begin{aligned}
\sum_{1}^{n} \text {IntegratedGrads}_{i}(x) &=\sum_{1}^{n}\left(\gamma_{i}(1)-\gamma_{i}(0)\right) \int_{0}^{1} \frac{\partial F(\gamma)}{\partial \gamma_{i}} d \alpha \\
&=\sum_{1}^{n} \int_{0}^{1} \frac{\partial F(\gamma)}{\partial \gamma_{i}} \frac{\partial \gamma_{i}}{\partial \alpha} d \alpha \\
&=\int_{0}^{1} \frac{\partial F(\gamma)}{\partial \gamma} \cdot \frac{\partial \gamma}{\partial \alpha} d \alpha \\
&=\int_{0}^{1} \frac{\partial F(\gamma)}{\partial \alpha} d \alpha \\
&=F(\gamma(1))-F(\gamma(0))=F(x)-F\left(x^{\prime}\right)
\end{aligned}$$

对于模型，baseline 一般要求其 $F(x_0)\approx 0$，这样更为方便。

#### 积分梯度法满足公理 Sensitivity(a)

> $\textbf{Remark 2.}$ Integrated gradients satisfies Sensivity(a) because Completeness implies Sensivity(a) and is thus a strengthening of the Sensitivity(a) axiom. This is because Sensitivity(a) refers to a case where the baseline and the input differ only in one variable, for which Completeness asserts that the difference in the two output values is equal to the attribution to this variable. Attributions generated by integrated gradients satisfy Implementation Invariance since they are based only on the gradients of the function represented by the network.

即 Completeness 成立是 Sensitivity 成立的充分条件，因为有：

$$\begin{aligned}
\text {IntegratedGrads}_{i}(x) &=\left(\gamma_{i}(1)-\gamma_{i}(0)\right) \int_{0}^{1} \frac{\partial F(\gamma)}{\partial \gamma_{i}} d \alpha \\
&=\int_{0}^{1} \frac{\partial F(\gamma)}{\partial \gamma_{i}} \frac{\partial \gamma_{i}}{\partial \alpha} d \alpha \\
&=\int_{0}^{1} \frac{\partial F\left(\gamma\left(\gamma_{i}\right)\right)}{\partial \alpha} d \alpha \\
&=F\left(\gamma\left(\gamma_{i}(1)\right)\right)-F\left(\gamma\left(\gamma_{i}(0)\right)\right)
\end{aligned}$$

当 $x$ 和 $x_0$ 只有 $i$ 维度不同时，则有 $F(\gamma(\gamma_i(1)))-F(\gamma(\gamma_i(0)))=F(\gamma(1))-F(\gamma(0))=F(x)-F(x_0)$，所以 Sensitivity 成立。

Sensitivity(a) 指的是输入和基线仅在一个变量上不同并导致了预测结果不同，那么这个变量的归因为非 0 值。公理 Completeness 是 Sensitivity(a) 的强化，因此 Completeness 指明了这个变量的归因等于 $F(x)-F(x_0)\neq 0$。

#### 积分梯度法满足公理 Implementation Invariance

积分梯度法是基于网络所表示的函数定义的，而不是具体的网络结构，只和函数的梯度相关，因此其满足 Implementation Invariance。

## 四、Uniqueness of Integrated Gradients

下面通过两步来证明积分梯度法的唯一性。首先，定义了一种称为路径方法 (Path Methods) 的归因方法，路径方法是积分梯度法的推广，并证明路径方法 (Path Methods) 是唯一满足既定公理的方法；其次，指出由于 Path method 的唯一性，且由于对称性，Integrate gradients 方法是最准确（canonical）的方法。

### 4.1 Path Methods

积分梯度法是沿着基线 (baseline input) 和输入 (input) 间的直线对梯度进行累计。但是，在这两点间存在着许多的路径，每种路径则对应着不同的归因方法。例如，在输入为 2 维的例子中，下图中的三条路径分别对应不同的归因方法。

<div align=center><img src="/images/Axiomatic_Attribution_for_Deep_Networks-2022-02-16-17-11-54.png" alt="Axiomatic_Attribution_for_Deep_Networks-2022-02-16-17-11-54" style="zoom:50%;" /></div>

指出一般的 Path Method，及其 PathIntegratedGrads 定义：

> Given a path function $\gamma$, path integrated gradients are obtained by integrating the gradients along the path $\gamma(\alpha)$ for $\alpha \in [0,1]$. Formally, path integrated gradients along the $i$th dimension for an input $x$ is defined as follows:
> 
> 令 $\gamma=(\gamma_1,\cdots,\gamma_n):[0,1] \to \mathbb{R}^n$ 是 $\mathbb{R}^n$ 上从基线 $x_0$ 到输入 $x$ 的任意路径，其中 $\gamma(0)=x_0$ 并且 $\gamma(1)=x$。那么给定一个路径函数 $\gamma$，则路径方法 (Path Methods) 指的是，沿路径 $\gamma(\alpha)$ 对梯度进行积分，其中 $\alpha \in [0,1]$。更加正式的来说，输入 $x$ 在第 $i$ 维的路径梯度积分 (Path Integrated Gradients) 为：

$$\text{PathIntegratedGrads}\gamma_i(x)::=\int_0^1\frac{\partial F(\gamma(\alpha))}{\partial\gamma_i(\alpha)}\frac{\partial\gamma_i(\alpha)}{\partial\alpha}d\alpha$$

其中 $\frac{\partial F(\gamma(\alpha))}{\partial\gamma_i(\alpha)}$ 是函数 $F$ 在 $x$ 处沿维度 $i$ 的梯度。基于路径梯度积分的归因方法统称为路径方法 (Path Methods)。

注意：积分梯度法 (Integrated Gradients) 是路径方法 (Path Methods) 沿直线方向的一个特例，即 $\gamma(\alpha)=x_0+\alpha\times(x-x_0)$，其中 $\alpha \in [0,1]$。

#### 路径方法所满足的公理

> $\textbf{Remark 3.}$ All path methods satisfy Implementation Invariance. This follows from the fact that they are defined using the underlying gradients, which do not depend on the implementation. They also satisfy Completeness (the proof is similar to that of Proposition 1) and Sensitvity(a) which is implied by Completeness (see Remark 2).

##### 1. 路径方法满足 Implementation Invariance

原因同积分梯度法一致。即路径方法是基于网络所表示函数的梯度进行归因，与网络的实现无关。

##### 2. 路径方法满足公理 Completeness

$$\text{PathIntegratedGrads}^\gamma_i(x)=F(x)-F(x_0)$$

##### 3. 路径方法满足公理 Sensitivity(b)

公理 Sensitivity(b) 指的是，由网络实现的函数如果不依赖某些变量，那么这些变量上的归因为 0（是 Implementation Invariance 的充分条件）：

> $\textbf{Axiom: Sensitivity(b).}$ (called Dummy in (Friedman, 2004)) If the function implemented by the deep network does not depend (mathematically) on some variable, then the attribution to that variable is always zero.

基于梯度的方法均满足这个公理，路径方法同样也满足。

##### 4. 路径方法满足公理 Linearity

规定线性性：

> $\textbf{Axiom: Linearity.}$ Suppose that we linearly composed two deep networks modeled by the functions f1 and f2 to form a third network that models the function $a×f1+b×f2$, i.e., a linear combination of the two networks. Then we'd like the attributions for $a × f1 + b × f2$ to be the weighted sum of the attributions for f1 and f2 with weights a and b respectively. Intuitively, we would like the attributions to preserve any linearity within the network.

如果使用两个深度网络建模的函数 $f_1$ 和 $f_2$ 线性合并为第三个网络所表示的函数 $a×f1+b×f2$ ，那么网络 $a×f1+b×f2$ 的归因应该等于 $f_1$ 和 $f_2$ 归因分别以权重 $a$ 和 $b$ 进行加权求和。由于路径方法满足 Completeness，那么显然满足 Linearity。

满足上诉 Sensitivity、Implementation Invariance、Completeness、Linearity，Path Method 是唯一的方法：

> $\textbf{Proposition 2.}$ (Theorem 1 (Friedman, 2004)) Path methods are the only attribution methods that always satisfy Implementation Invariance, Sensitivity(b), Linearity, and Completeness.

### 4.2 Integrated Gradients is Symmetry-Preserving

积分梯度法满足对称保持 (Integrated Gradients is Symmetry-Preserving)，这解决了积分梯度法为什么是一个最优的选择。首先，其在数学上的定义最简单；其次，该方法满足对称保存。

定义对称性：

> $\textbf{Symmetry-Preserving.}$ Two input variables are symmetric $w.r.t.$ a function if swapping them does not change the function. For instance, $x$ and $y$ are symmetric $w.r.t.$ F if and only if $F(x, y) = F(y, x)$ for all values of $x$ and $y$. An attribution method is symmetry preserving, if for all inputs that have identical values for symmetric variables and baselines that have identical values for symmetric variables, the symmetric variables receive identical attributions.

如果交换两个输入变量不改变函数的值，那么称输入变量关于该函数对称。举例来说，如果对于所有的 $x$ 和 $y$ 均有 $F(x,y)=F(y,x)$ ，则称 $x$ 和 $y$ 关于函数 $F$ 对称。

如果输入中的对称变量具有相同的值，并且基线 (baseline) 在对称变量上也具有相同的值，如果输入中对称变量的归因相同，那么就称这样的归因方法是满足对称保持的。

举例来说，存在一个逻辑回归模型 $\text{Sigmoid}(x_1+x_2+\cdots)$ ，其中 $x_1$ 和 $x_2$ 对该模型是对称变量。若输入为 $x_1=x_2=1$，基线 $x_1=x_2=0$，那么一个满足对称保持的归因方法必须为 $x_1$ 和 $x_2$ 提供相同的归因。

说明 IntegrateGradients 满足对称性：

> $\textbf{Theorem 1.}$ Integrated gradients is the unique path method that is symmetry-preserving.

如果允许对多条路径的归因进行平均，那么也存在一些其他的方法满足对称保持。但是，多条路径平均的方法在计算成本上太高，不太适合深度网络。

## 五、Applying Integrated Gradients

近似计算的方法：

选择基线：

1. 应用积分梯度法的关键步骤是选择一个好的基线，基线在模型中的得分最好接近 0，这样有助于对归因结果的解释。
2. 基线必须代表一个完全没有信息的样本，这样才能区别出原因是来自输入还是基线。
3. 在图像任务中可以选择全黑图像，或者由噪音组成的图像。在文本任务中，使用全 0 的 embedding 是一个较好的选择。
4. 图像中的全黑图像也代表着一种有意义的输入，但是文本中的全 0 向量完全没有任何有效的意义。

$$\text {IntegratedGrads}_{i}^{ approx}(x)::=\left(x_{i}-x_{i}^{\prime}\right) \times \sum_{k=1}^{m} \frac{\partial F\left(x^{\prime}+\frac{k}{m} \times\left(x-x^{\prime}\right)\right)}{x_{i}} \times \frac{1}{m}$$

实验证明一般 m 在 20 和 300 之间。

## 六、实验

代码如下：

```python
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations

def normalize(array):
    arr_min = np.min(array)
    arr_max = np.max(array)
    return (array - arr_min) / (arr_max - arr_min + K.epsilon())

def linearize_activation(model, custom_objects=None):
    model.layers[-1].activation = activations.linear
    return model

def compute_gradient(model, output_index, input_image):
    input_tensor = model.input
    output_tensor = model.output

    loss_fn = output_tensor[:, output_index]

    grad_fn = K.gradients(loss_fn, input_tensor)[0]

    compute_fn = K.function([input_tensor], [grad_fn])

    grads = compute_fn([input_image])[0]

    return grads

def int_visualize_saliency(model, output_index, input_image, custom_objects=None):
    model = linearize_activation(model, custom_objects)
    grads = compute_gradient(model, output_index, input_image)
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1

    attri = np.transpose(grads, (1, 2, 3, 0))
    attri = np.mean(attri, axis=3)              # 梯度平均值
    attri = attri * input_image[-1]             # 乘以 x-x'，x'=0
    attri = np.sum(attri, axis=channel_idx)     # 把三个通道的值加起来
    attri = np.maximum(0, attri)                # 只显示正值
    
    return 1.0 - normalize(attri)

if __name__ == '__main__':
    from keras.applications import vgg16
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.imagenet_utils import decode_predictions
    import matplotlib.pyplot as plt

    filename = 'cat.jpg' #285
    n_intervals = 10

    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    intervals = np.linspace(0, 1, n_intervals + 1)[1:].reshape((n_intervals, 1, 1, 1))
    image_batch = np.tile(image_batch, (n_intervals, 1, 1, 1)) * n_intervals
    processed_image = vgg16.preprocess_input(image_batch.copy())

    vgg_model = vgg16.VGG16(weights='imagenet')
    predictions = vgg_model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
    print(label_vgg)

    saliency_map = int_visualize_saliency(vgg_model, 285, processed_image)
    plt.matshow(saliency_map, cmap='gray')
    plt.savefig('cat_saliency_map_int.png')
    plt.show()
```

在 m=10 时，分别是原图，gradients 和 integrated gradients 方法：

<div align=center><img src="/images/Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-28-46.png" alt="Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-28-46" style="zoom:100%;" /></div>

<div align=center><img src="/images/Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-28-59.png" alt="Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-28-59" style="zoom:100%;" /></div>

<div align=center><img src="/images/Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-29-11.png" alt="Axiomatic_Attribution_for_Deep_Networks-2022-01-11-22-29-11" style="zoom:100%;" /></div>
