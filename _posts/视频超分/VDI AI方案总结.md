# VDI AI方案总结

VDI AI算法研发路径、技术思路拆解，RIFE等算法方案调研:

## AI插帧方面：推理速度层面大概率难以应用
综合推理速度与质量指标，SOTA为RIFE（开源，[code](https://github.com/hzwer/arXiv2020-RIFE)）

经过实际测试验证：  
推理质量方面，对于视频2倍插帧整体可用，对于办公场景下拖动等连续突变帧，效果不佳（但可以用if-else控制）;  
GeForce 2080 GPU推理耗时45ms，单核CPU耗时3s，推理速度层面难以应用。

AIM超分竞赛中的“时域超分”赛道，内容为视频插帧。

## AI超分方面：推理速度层面大概率难以应用
综合推理速度与质量指标，SOTA为TDAN（开源，[code](https://github.com/YapengTian/TDAN-VSR-CVPR-2020)），根据资料阅读，基于GeForce 1080，13ms 4倍超分，推理耗时层面分析，可能也难以应用。

相关竞赛：AIM、NTIRE等视频超分竞赛

## AI案例分析
Maxine（未开源，开放SDK因此可以猜测技术点）可以将视频会议的流量降的非常低。其中的压缩部分，核心技术点是求变化差值 + 超分（还有脸部识别等不太相关的部分）；RBX情况类似。

当前推进AI增强在质控方面的调研：[弱网下的极限视频通信](https://www.infoq.cn/video/aGaqktPVM0c2ApLztDIO?utm_source=home_video&utm_medium=video)
