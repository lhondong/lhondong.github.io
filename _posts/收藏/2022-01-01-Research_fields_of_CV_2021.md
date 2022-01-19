---
title: "2021 年计算机视觉值得研究的领域"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-5.jpg"
tags:
  - CV
---

# 2021年深度学习哪些方向比较新颖，处于上升期或者朝阳阶段，比较有研究潜力？

## 一、生成模型

回答几个我最近在研究的方向～ 我主要在做生成模型，不过乱七八糟的也有涉猎

### 可解释性

feature-based 研究的很多了，instance-based个人感觉在上升期，从研究的角度来说缺乏benchmark/axiom/sanity check. 主流方法是influence function, 我觉得这里面self influence的概念非常有趣，应该很值得研究。当然，更意思的方向是跳出influence function本身，比如像relatIF 加一些regularization，也是水文章的一贯套路(relatIF是好文章)。

Influence function for generative models也是很值得做的。Influence function for GAN已经有人做了，虽然文章直接优化FID是有点问题的，但是框架搭好了，换一个evaluation换个setting就可以直接发paper。

我最近写了Influence function for VAE, 有不少比较有意思的observation ( [paper](https://arxiv.org/pdf/2105.14203.pdf); code repo: VAE-TracIn-pytorch)。

### 无监督生成学习

最近的denoising diffusion probabilistic model(DDPM)绝对是热坑，效果好，但是速度慢没有meaningful latent space限制了很多应用，有待发掘。我去年实习写了一篇DiffWave是这个方法在语音上的应用，效果很好，最近应该能看到这个模型的application井喷，比如3D point cloud生成。

DDPM的加速最近已经有不少paper了，目前来看有几类，有的用conditioned on noise level去重新训练，有的用jumping step缩短Markov Chain，有的在DDPM++里面研究更快的solver. 我最近写了FastDPM, 是一种结合noise level和jumping step的快速生成的框架(无需retrain, original DDPM checkpoint拿来直接用)，统一并推广了目前的好几种方法，给出了不同任务(图像, 语音)的recipe (paper: https//arxiv.org/pdf/2106.00132.pdf; code repo: FastDPM_pytorch)。

生成模型里的Normalizing flow模型，用可逆网络转化数据分布，很fancy 能提供likelihood和比较好的解释性但是效果偏偏做不上去，一方面需要在理论上有补充，因为可逆或者Lipschitz网络的capacity确实有限。另一方面，实际应用中，training不稳定可能是效果上不去的原因，其中initialization 和training landscape都是有待研究的问题。潜在的突破口：augmented dimension或者类似surVAE那种generalized mapping. 除此之外，normalizing flow on discrete domain也是很重要的问题，潜在突破口是用OT里面的sinkhorn network。

我对residual flow这个模型有执念，很喜欢这个框架，虽然它不火。今年早些时候我写了residual flow的universal approximation in MMD的证明，很难做，需要比较特殊的假设 ([paper](https://arxiv.org/pdf/2103.05793.pdf))。之后可能继续钻研它的capacity和learnability。

再补充一个：

生成模型的overfitting是一个长久的问题，但是本身很难定义，很大一个原因是mode collapse和copy training data耦合在一起。我们组去年发表了data-copying test用于检测相关性质，不过这个idea还停留在比较初级的阶段，我觉得这一块需要更多high level的框架。

### Meta learning

Meta learning + Generative model方向个人十分看好，meta learning 框架可以直接套，loss改成生成模型的loss就可以了。Again, GAN已经被做了，不过GAN的paper那么多，随便找上一个加上meta learning还是很容易的。类似可以做multitask + GAN。

## 二、深度学习本质问题

1. 是否存在神经网络之外的推理方式？  
当前，神经网络成为训练以后的唯一产物，而几乎所有算法均假设将输入送给神经网络以后，一次性地得到输出结果。然而，是否能够设计直接向前传递以外的其他推理方式？例如，当一个物体处于罕见的视角或者被严重遮挡时，能否通过多次迭代式的处理，逐渐恢复其缺失的特征，最终完成识别任务？这就涉及到将强化学习引入训练，或者通过类似于image warping的方式找到一条困难样例和简单样例之间的路径。后者可以导向一个非常本质的问题：**如何以尽可能低的维度刻画语义空间？**GAN以及相关的方法或许能够提供一些思路，但是目前还没有通用的、能够轻易跨越不同domain的方法。
2. 是否存在更精细的标注方式，能够推进视觉的理解？  
我最近提出了一个假想：[当前所有的视觉识别算法都远远没有达到完整](https://zhuanlan.zhihu.com/p/376145664)，而这很可能是当前不够精细的标注所导致的。那么，是否能够在可行的范围内，定义一种超越instance segmentation的标注方式，进一步推进视觉识别？这就涉及到一系列根本问题：**什么是一个物体？如何定义一个物体？物体和部件之间有什么联系？**这些问题不得到解决，物体检测和分割将步图像分类的后尘，迅速陷入过拟合的困境。
3. 如何解决大模型和小样本之间的矛盾？  
当前，大模型成为AI领域颇有前景的规模化解决方案。然而，大模型的本质在于，通过预训练阶段大量吸收数据（有标签或者无标签均可），缓解下游小样本学习的压力。这就带来了一个新的矛盾：**大模型看到的数据越多，模型就越需要适应一个广泛而分散的数据分布，因而通过小样本进行局部拟合的难度就越大**。这很可能是制约大模型思路落地的一个瓶颈。
4. 能否通过各种方式生成接近真实的数据？  
生成数据（包括虚拟场景或者GAN生成的数据）很可能会带来新的学习范式，然而这些数据和真实数据之间存在一种难以逾越的domain gap，制约了其在识别任务中发挥作用。我们提出问题：**这种domain gap，本质上是不是特定的识别任务带来的learning bias？**我们希望通过改变学习目标，使得这种domain gap得到缓解甚至消失，从而能够在有朝一日消灭人工标注，真正开启新的学习范式。
5. 是否存在更高效的人机交互模式？  
目前，人机之间的交互效率还很低，我就经常因为为做PPT而头疼不已。我认为AI算法或许会深刻地改变人机交互的模式，使得以下场景变得更容易：**多媒体内容设计和排版、跨模态信息检索、游戏微操作**，等等。多模态算法很可能会在这波“人机交互革命”中发挥重要作用。

在我看来，上述任何一个问题，相比于无止境的烧卡刷点，都要有趣且接近本质，但是风险也要更大一些。因此，大部分研究人员迫于现实压力而选择跟风，是再正常不过的事情。只要有人在认真思考这些问题并且稳步推进它们，AI就不是一个遥不可及的梦。

限于时间，无法将上述每个点写得太仔细；同时限于水平和视野，我也无法囊括所有重要的问题（如可解释性——虽然我对深度学习的可解释性感到悲观，不过看到有学者在这个领域深耕，还是能够感觉到勇气和希望）。

### NAS、对比学习、Transformer的局限性：

1. 原本以NAS为代表的AutoML技术受到了广泛的期待，我还主张“自动机器学习之于深度学习，就好比深度学习之于传统方法”，不过后来发现它的缺陷是明显的。在搜索空间指数级扩大之后，算法就必须在精度和速度之间做出选择。后来盛行的权重共享类搜索方法，相当于追求搜索空间中的平摊精度，而平摊精度与最佳个体的精度往往并不吻合。
2. 对比学习被广泛引入图像领域作为自监督任务以后，前世代的自监督算法（如预测旋转、拼图、上色等）纷纷被吊打，甚至开始在下游任务中超越有监督训练的模型。然而，当前的对比学习类方法（包括BYOL）对于数据扩增（data augmentation）的依赖过重，因而不可避免地陷入了invariance和consistency之间的矛盾：强力的augmentation能够促进学习效果，但是如果augmentation过强，不同view之间的可预测性又无法保证。
3. 至于Transformer，虽然目前还处在比较兴盛的状态，然而它的上限也是明显的。除了更快的信息交换，似乎这种模型并没有体现出显著的优势。问题是：CV任务真的需要频繁而快速的视觉信息交换吗？遗憾的是，学界依然沉浸在“先将所有任务用Transformer刷一遍”的廉价快乐中，鲜有人愿意思考一些更深入的问题。

非常欢迎针对各种问题的讨论，也希望这些观点能够引发更多的思考吧。