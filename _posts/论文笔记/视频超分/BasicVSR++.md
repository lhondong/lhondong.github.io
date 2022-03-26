# BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

CVPR2021

验证现有的各种网络结构对VSR任务的影响，并进而找到了一种SOTA方案，RNN结构，光流

- 双向RNN
- RNN单元内部：
  - 用当前帧 $x_i$ 和前一帧 $x_{i-1}$(或是后一帧 $x_{i+1}$)计算光流
  - 借助光流对前一帧的隐藏层输出 $h^f_{i-1}$(或是后一帧的隐藏层输出 $h^b_{i+1}$)进行特征对齐
  - 将对齐后的特征与原图进行ResBlock计算，得到 $h^f_{i}$ 和 $h^b_{i}$
- $h^f_{i}$ 和 $h^b_{i}$ 拼接后进行Upsample得到高清输出

## IconVSR：在BasicVSR使用关键帧补充信息

- 在特征对齐后加一个步骤，在关键帧处把前后帧的特征混入 $h^f_{i}$ 和 $h^b_{i}$ 中
- 正向RNN以反向RNN的输出作为输入

具体怎么混入：在关键帧处从前后帧和当前帧中提取特征然后与 $h^f_{i}$ 和 $h^b_{i}$ 进行卷积，非关键帧处不变

# BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment

二阶网状传播、光流引导的可形变卷积对齐，RNN结构，光流、可形变卷积。

- 双向传播叠四层
- 在传播中跨一级连接
- 在每一个传播模块中：
  - 对之前的输出进行光流引导的可形变卷积
  - 与当前输入进行拼接
  - 经过一堆卷积和ReLU
- 最后输出的高清残差是输出特征的PixelShuffle
- 高清残差与原图上采样结果相加
- 二阶网状传播为何有效：从更多的地方获取信息
- 光流引导的可形变卷积为何有效：从临近区域的特征中提取信息，帮助恢复细节

光流引导的可形变卷积：
- 以光流为可形变卷积偏置量的基础值
- 训练可形变卷积偏置量在基础值上增加的残差

这样可以保证训练的稳定性