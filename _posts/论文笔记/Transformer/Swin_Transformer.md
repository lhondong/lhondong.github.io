# Swin Transformer

Swin Transformer 提出了一种针对视觉任务的通用的 Transformer 架构。Transformer 架构在 NLP 任务中已经算得上一种通用的架构，但是如果想迁移到视觉任务中有一个比较大的困难就是处理数据的尺寸不一样。

1. 最主要的原因是两个领域涉及的 scale 不同，NLP 任务以 token 为单位，scale 是标准固定的，而 CV 中基本元素的 scale 变化范围非常大。
2. CV 比起 NLP 需要更大的分辨率，而且 CV 中使用 Transformer 的计算复杂度是图像尺度的平方，这会导致计算量过于庞大， 例如语义分割，需要像素级的密集预测，这对于高分辨率图像上的 Transformer 来说是难以处理的。

- 图片预处理：分块和降维 (Patch Partition)
- 线性变换 (Linear Embedding)
- Swin Transformer Block
- Stage 2/3/4

作为 CNN 的一个替代品，Transformer 设计了一个自注意力机制来获取内容之间全局交互的信息，并且在多种视觉任务上取得了不错的性能。但是针对图像复原的视觉 Transformer 通常将输入图像分割为固定大小的 patch，并且独立的处理每个 patch。这种策略不可避免的引入两类缺陷。

1. 边界像素点不能使用 patch 范围之外的邻接像素来做图像复原。
2. 复原后的图像中容易在 patch 周围引入边界伪影。虽然这个问题可以通过 patch 交叠来减轻，但是这会引入额外的计算负担。

由于结合了 CNN 和 Transformer 的优势，SwinTransformer 展现出了巨大的优势。一方面，由于局部注意力机制，SwinTransformer 具有 CNN 处理大尺寸图像的优势。另一方面，由于 shiftd window 机制，它又具有 Transformer 对长依赖建模的能力。
