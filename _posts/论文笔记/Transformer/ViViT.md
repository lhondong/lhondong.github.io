# ViViT: A Video Vision Transformer

- 如何构建时空 token，即左侧灰色框
- 如何设计 Transformer 结构。一共提出四种结构，第一种是最朴素的，后三种都是构建时间、空间的 transformer/self-attention

<div align=center><img src="/assets/ViViT-2022-05-23-15-35-37.png" alt="ViViT-2022-05-23-15-35-37" style="zoom:50%;" /></div>

## 如何构建 token

如何将一个视频转换为一组序列，作为 Transformer 的输入

- Uniform frame sampling：就是先提取帧，每一帧按照 ViT 的方法提取 token，然后把不同帧的 token 拼接起来作为输入。
- Tubelet embedding：前一种方法是提取 2D 图像特征，这种方法是提取立方体，假设每个 tublet 的 shape 是 t, w, h，那就是说每 t 帧提取一次特征，取每一帧相同位置的 w, h patch 组成输入

<div align=center><img src="/assets/ViViT-2022-05-23-15-37-26.png" alt="ViViT-2022-05-23-15-37-26" style="zoom:50%;" /></div>

<div align=center><img src="/assets/ViViT-2022-05-23-15-37-41.png" alt="ViViT-2022-05-23-15-37-41" style="zoom:50%;" /></div>

## 如何设计 Transformer 结构

### Transformer 结构的变种一

- 直接将前面提取到的时空token作为transformer的属于，使用普通transformer结构得到最终结果。
- 这个没啥好说的，就是最普通、最直接的方法。

### Transformer 结构的变种二 Factorised encoder

- 使用两个 transformer
- 第一个是 spatial transformer，输入是某一帧的多个token，输出一个token
- 第二个是temporal transformer，输入是前一步多帧的token（每帧对应一个token），输出结果就通过mlp进行分类

<div align=center><img src="/assets/ViViT-2022-05-23-15-41-16.png" alt="ViViT-2022-05-23-15-41-16" style="zoom:50%;" /></div>

### Transformer 结构的变种三 Factorised self-attention

- 通过 self-attention 层将时空数据分开处理
- 空间层只在同一帧内不同token间进行attention操作
- 时间层对不同帧同一位置的token进行attention操作

<div align=center><img src="/assets/ViViT-2022-05-23-15-41-39.png" alt="ViViT-2022-05-23-15-41-39" style="zoom:50%;" /></div>

### Transformer 结构的变种四 Factorised dot-product attention

- 与变种三类似，只不过时间、空间heads是并行的，而不是串行的。
- spatial还是同一帧内不同token，temporal是不同帧同一位置的token

<div align=center><img src="/assets/ViViT-2022-05-23-15-43-43.png" alt="ViViT-2022-05-23-15-43-43" style="zoom:50%;" /></div>