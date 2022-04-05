# Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement

## 摘要

提出层级式视频压缩(Hierarchical Learned Video Compression,HLVC)，包括双向深度预测(Bi-Directional Deep Compression, BDDC)和单运动深度压缩(Single Motion Deep Compression,SMDC)，BDDC主要用于压缩第二层视频帧，SMDC采用单个运动向量预测多个帧，可以节省运动信息的码率。在解码端，利用加权递归质量增强(Weighted Recurrent Quality Enhancement,WRQE)网络，此网络将压缩帧和比特流作为输入。

框架结构、分层结构与Video Compression through Image Interpolation很相似，都是将一个GOP中的帧分为三层。

$$
\hat{q}_{m}=\operatorname{round}\left(E_{m}\left(\left[f_{5 \rightarrow 0}, f_{5 \rightarrow 10}\right]\right)\right) 
$$

$$
{\left[\hat{f}_{5 \rightarrow 0}, \hat{f}_{5 \rightarrow 10}\right]=D_{m}\left(\hat{q}_{m}\right)}
$$

$$
x_{0 \rightarrow 5}^{C}=W_{b}\left(x_{0}^{C}, \hat{f}_{5 \rightarrow 0}\right), x_{10 \rightarrow 5}^{C}=W_{b}\left(x_{10}^{C}, \hat{f}_{5 \rightarrow 10}\right)
$$

$$
\tilde{x}_{5}=MP\left(\left[x_{0 \rightarrow 5}^{C}, x_{10 \rightarrow 5}^{C}, \hat{f}_{5 \rightarrow 0}, \hat{f}_{5 \rightarrow 10}\right]\right)
$$

$$
\hat{q}_{r}=\operatorname{round}\left(E_{r}\left(x_{5}-\tilde{x}_{5}\right)\right)
$$

$$
x_{5}^{C}=D_{r}\left(\hat{q}_{r}\right)+\tilde{x}_{5}
$$

$$
f_{\mathrm{inv}}(a+\Delta a(a, b), b+\Delta b(a, b))=-f(a, b)
$$

$$\hat{f}_{1 \rightarrow 0}=\operatorname{Inverse}(\underbrace{0.5 \times \underbrace{\text { Inverse }\left(\hat{f}_{2 \rightarrow 0}\right)}_{\hat{f}_{0 \rightarrow 2}}}_{\hat{f}_{0 \rightarrow 1}})
$$

$$
\hat{f}_{1 \rightarrow 2}=\operatorname{Inverse}(\underbrace{0.5 \times \hat{f}_{2 \rightarrow 0}}_{\hat{f}_{2 \rightarrow 1}})
$$

$$
L = \lambda D +R
$$

$$
L_{\mathrm{BD}}=\lambda_{\mathrm{BD}} \cdot \underbrace{D\left(x_{5}, x_{5}^{C}\right)}_{\text {Distortion }}+\underbrace{R\left(\hat{q}_{m}\right)+R\left(\hat{q}_{r}\right)}_{\text {Total bit-rate }}
$$

$$
L_{\mathrm{SM}}=\lambda_{\mathrm{SM}} \cdot \underbrace{\left(D\left(x_{1}, x_{1}^{C}\right)+D\left(x_{2}, x_{2}^{C}\right)\right)}_{\text {Total distortion }} +\underbrace{R\left(\hat{q}_{m}\right)+R\left(\hat{q}_{r 1}\right)+R\left(\hat{q}_{r 2}\right)}_{\text {Total bit-rate }}
$$

$$
L_{\mathrm{QE}}=\frac{1}{N} \sum_{i=1}^{N} D\left(x_{i}, x_{i}^{D}\right)
$$