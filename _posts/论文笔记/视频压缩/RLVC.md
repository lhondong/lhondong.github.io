# Learning for Video Compression with Recurrent Auto-Encoder and Recurrent Probability Model

[IEEE J-STSP]

ETH Zurich Vision Lab

## 摘要

现有的方法仅使用少量的参考帧压缩视频帧，限制了充分利用视频帧之间相关性的能力。

提出了Recurrent Learned Video Compression，使用Recurrent Auto-encoder（RAE）和 Recurrent Probability Model（RPM）。

- RAE在编码器和解码器中都使用循环单元，这样很大事件范围内的时域信息都可以用于生成潜表示，重构解压缩帧。
- RPM以潜表示的分布为条件，递归估计潜在表征的概率质量函数Probability Mass Function（PMF）。

由于连续帧之间的相关性，条件交叉熵可能低于独立交叉熵，从而降低比特率。

在PSNR和MS-SSIM两个方面都达到了最先进的视频压缩性能，此外，在PSNR上优于x265的默认低延迟P(LDP)设置，在MS-SSIM上也比SSIM调优的x265和最慢的x265设置性能更好。

## Introduction

现有的手工制作的和基于学习的视频压缩方法都利用non-recurrent结构来压缩顺序视频数据，因此只有有限数量的参考帧可以用于压缩新视频帧，这样就限制了它们挖掘时间相关性和减少冗余的能力。

强调采用循环压缩框架，可以充分利用连续帧中的相关信息，更加有利于视频压缩。

此外，在以往的基于学习的方法的熵编码中，对每个帧独立估计潜表示的PMF，这样忽略了相邻帧之间潜表示的相关性。类比像素域中的参考帧，充分利用latent domain中的相关性也有利于潜表示的压缩。

Intuitively, the temporal correlation in the latent domain also can be explored in a recurrent manner.