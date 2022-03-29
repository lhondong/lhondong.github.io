# Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement

## 摘要

提出层级式视频压缩(Hierarchical Learned Video Compression,HLVC)，包括双向深度预测(Bi-Directional Deep Compression, BDDC)和单运动深度压缩(Single Motion Deep Compression,SMDC)，BDDC主要用于压缩第二层视频帧，SMDC采用单个运动向量预测多个帧，可以节省运动信息的码率。在解码端，利用加权递归质量增强(Weighted Recurrent Quality Enhancement,WRQE)网络，此网络将压缩帧和比特流作为输入。

框架结构、分层结构与Video Compression through Image Interpolation很相似，都是将一个GOP中的帧分为三层。