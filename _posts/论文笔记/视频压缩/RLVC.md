# Learning for Video Compression with Recurrent Auto-Encoder and Recurrent Probability Model

[IEEE J-STSP]

ETH Zurich Vision Lab

## 摘要

现有的方法仅使用少量的参考帧压缩视频帧，限制了充分利用视频帧之间相关性的能力。

Recurrent Learned Video Compression，提出递归自编码器（Recurrent Auto-Encoder，RAE）和递归概率模型（Recurrent Probability Model，RPM）以充分利用视频像素域（pixel domain）和表征域（latent domain）的帧间相关性，实现高效视频压缩。

- RAE 在编码器和解码器中都使用循环单元，这样很大事件范围内的时域信息都可以用于生成潜表示，重构解压缩帧。
- RPM 以潜表示的分布为条件，递归估计潜在表征的概率质量函数 Probability Mass Function（PMF）。

由于连续帧之间的相关性，条件交叉熵可能低于独立交叉熵，从而降低比特率。

在 PSNR 和 MS-SSIM 两个方面都达到了最先进的视频压缩性能，此外，在 PSNR 上优于 x265 的默认低延迟 P(LDP) 设置，在 MS-SSIM 上也比 SSIM 调优的 x265 和最慢的 x265 设置性能更好。

## Introduction

现有的手工制作的和基于学习的视频压缩方法都利用 non-recurrent 结构来压缩顺序视频数据，因此只有有限数量的参考帧可以用于压缩新视频帧，这样就限制了它们挖掘时间相关性和减少冗余的能力。

强调采用循环压缩框架，可以充分利用连续帧中的相关信息，更加有利于视频压缩。

此外，在以往的基于学习的方法的熵编码中，对每个帧独立估计潜表示的 PMF，这样忽略了相邻帧之间潜表示的相关性。类比像素域中的参考帧，充分利用 latent domain 中的相关性也有利于潜表示的压缩。

Intuitively, the temporal correlation in the latent domain also can be explored in a recurrent manner.

<div align=center><img src="/assets/RLVC-2022-04-22-16-05-13.png" alt="RLVC-2022-04-22-16-05-13" style="zoom:50%;" /></div>

### RAE

RLVC 方法的宏观编码框架与现有的深度视频压缩方法类似，主要包括运动估计网络、运动信息压缩网络、运动补偿网络，残差压缩网络等。不同之处在于，压缩运动信息和残差的网络，采用本文提出的递归自编码器（RAE）以更好地挖掘和利用视频的时域相关性。

<div align=center><img src="/assets/RLVC-2022-04-22-16-05-58.png" alt="RLVC-2022-04-22-16-05-58" style="zoom:50%;" /></div>

递归自编码器（RAE）的网络结构如下图，RAE 的编码器和解码器中分别含有一个 ConvLSTM 模块，以实现递归编码。

<div align=center><img src="/assets/RLVC-2022-04-22-16-06-31.png" alt="RLVC-2022-04-22-16-06-31" style="zoom:50%;" /></div>

递归自编码器（RAE）的优点：由于 RAE 编码器（图中 Encoder）中的 LSTM 网络可将 $x_{t-1}$ 中的信息传递给下一帧的 RAE 的编码器，同时 RAE 解码器（图中 Decoder）中的 LSTM 网络也可将压缩后的 $\hat{x}_{t-1}$ 中的信息传递给下一帧的 RAE 解码器。因此，RAE 中的 $y_t$ 仅需表征 $x_t$ 与 $x_{t-1}$ 之间的残差（即运动信息的残差，或帧间残差的残差），解码器再将前一帧 $\hat{x}_{t-1}$ 中信息与该压缩后的残差相结合，即可得到 $\hat{x}_t$。这使得 $y_t$ 需要表征的信息减少，因此可以提高率失真性能。

### 递归概率模型（RPM）

<div align=center><img src="/assets/RLVC-2022-04-22-16-09-16.png" alt="RLVC-2022-04-22-16-09-16" style="zoom:50%;" /></div>

RPM 网络通过 LSTM 获得前面各帧的 $y_t$ 信息以预测当前帧 $y_t$ 的时域条件概率，即

$$
q_{t}\left(\boldsymbol{y}_{t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)=\prod_{i=1}^{N} q_{i t}\left(y_{i t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)
$$

量化后，离散值 $y_{it}$ 通过 integrating the continuous logistic distribution 得到 conditional PMF：

$$
q_{i t}\left(y_{i t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)=\int_{y_{i t}-0.5}^{y_{i t}+0.5} \operatorname{Logistic}\left(y ; \mu_{i t}, s_{i t}\right) dy
$$

其中，logistic distribution 定义为：

$$
\operatorname{Logistic}(y ; \mu, s)=\frac{\exp (-(y-\mu) / s)}{s(1+\exp (-(y-\mu) / s))^{2}}
$$

its integral is the sigmoid distribution：

$$
\int \operatorname{Logistic}(y ; \mu, s) d y=\operatorname{Sigmod}(y ; \mu, s)+C
$$

由此，码率期望值即为条件交叉熵：

$$
H\left(p_{t}, q_{t}\right)=\mathbb{E}_{\boldsymbol{y}_{t} \sim p_{t}}\left[-\log _{2} q_{t}\left(\boldsymbol{y}_{t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)\right]
$$

$$
R_{\mathrm{RPM}}\left(\boldsymbol{y}_{t}\right) =-\log _{2}\left(q_{t}\left(\boldsymbol{y}_{t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)\right) =\sum_{i=1}^{N}-\log _{2}\left(q_{i t}\left(y_{i t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)\right)
$$

递归概率模型（RPM）的优点：由于帧与帧之间的时域相关性，在获得前面各帧的 $y_t$ 信息的先验条件后，当前帧 $y_t$ 取值的确定性一般会增加，这使得条件熵一般小于独立熵，即根据条件概率编码可以获得比根据独立概率编码更小的码率：

$$
\mathbb{E}_{\boldsymbol{y}_{t} \sim p_{t}}\left[-\log _{2} q_{t}\left(\boldsymbol{y}_{t} \mid \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t-1}\right)\right]<\mathbb{E}_{\boldsymbol{y}_{t} \sim p_{t}} [-\log _{2} q_{t}(\boldsymbol{y}_{t})]
$$

因此本文提出的 RPM 网络可以有效提高视频压缩效率。

### 损失函数

$$
\mathcal{L}_{1}=\lambda \cdot D\left(\boldsymbol{f}_{1}, \hat{\boldsymbol{f}}_{1}\right)+R_{1}\left(\boldsymbol{y}_{1}^{m}\right)+R_{1}\left(\boldsymbol{y}_{1}^{r}\right)
$$

### 率失真性能

RLVC 的压缩性能优于已有的深度视频压缩方法，并且在 MS-SSIM 指标上好于 x265 的大部分配置（包括 SSIM-tuned x265），PSNR 性能好于 x265 的 Low-delay P 配置。