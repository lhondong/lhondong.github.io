
# An image is worth 16×16 words: Transformers for image recognition at scale  

ICLR 2021

Google Research, Google Brain Team

<div align=center><img src="/assets/vit.gif" alt="fit" style="zoom:50%;" /></div>

Transformer 应用在 CV 有难点吗？
计算像素的 self-attention，序列长，维度爆炸
Trnasformer 的计算复杂度是 序列长度 n 的 平方 O（n^2）
224 分辨率的图片，有 50176 个像素点，（2d 图片 flatten）序列长度是 BERT 的近 100 倍。

###  如何理解 CNN 整体结构不变？

ResNet 50 有 4 个 stages (res2, res3, res4, res5), stage 不变，Attention 取代 每一个 stage 每一个 block 里的这个操作。

### Transformer 应用在 CV 的难点

- 计算像素的 self-attention，序列长，维度爆炸
- Transformer 的计算复杂度是 序列长度 n 的 平方 $O(n^2)$
224 分辨率的图片，有 50176 个像素点，（2d 图片 flatten）序列长度是 BERT 的近 100 倍。

### CNN 的归纳偏置

Inductive biases 归纳偏置，可以理解为先验知识。CNN 的 Inductive biases 是 locality 和 平移等变性 translation equaivariance（平移不变性 spatial invariance）。

- locality: CNN用滑动窗口在图片上做卷积。假设是图片相邻的区域有相似的特征。i.e., 桌椅在一起的概率大，距离近的物品 相关性越强。
- translation equaivariance：$f(g(x)) = g(f(x))$  ，卷积 $f$ 和平移 $g$ 函数的顺序不影响结果。

CNN 的卷积核 像一个 template 模板，同样的物体无论移动到哪里，遇到了相同的卷积核，它的输出一致。

CNN 有 locality 和 translation equivariance 归纳偏置，即 CNN 有很多先验信息，所以只需要较少的数据就可以学好一个模型。

Transformer 没有这些先验信息，只能从图片数据里，自己学习对**视觉世界**的感知。

### 怎么验证 Transformer 无 inductive bias 的假设？

在 1400万(ImageNet-21K) - 3000 万(JFT-300)得到图片数据集上预训练 trumps inductive bias, ViT + 足够训练数据，CV SOTA。 