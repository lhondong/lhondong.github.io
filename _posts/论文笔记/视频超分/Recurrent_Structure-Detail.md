# Video Super-Resolution with Recurrent Structure-Detail Network

CVPR2020

高频分量(Detail)和低频分量(Structure)分开计算，RNN结构，动态卷积。

输入部分：
- 从当前帧 $I_{t}^{L}$ 和上一帧 $I_{t-1}^{L}$ 中提取高频分量 $D_{t}^{LR}$、$D_{t-1}^{LR}$ 和低频分量 $S_{t}^{LR}$、$S_{t-1}^{LR}$
- 高频分量的分支：
  - 拼接 $D_{t}^{LR}$、$D_{t-1}^{LR}$、上一帧的隐藏层输出 $h_{t-1}^{SD}$ 经过隐藏状态自适应处理(Hidden-state Adaption)后的输出 $\hat h_{t-1}^{SD}$ 和上一帧的高频分量预测值 $\hat D_{t-1}$
  - 经过一个卷积和ReLU

主要是为了让高低频的处理过程有交互

上面说的隐藏状态自适应处理(Hidden-state Adaption)实际上就是基于动态卷积的特征对齐。