LiveNAS

SIGCOMM2020

Neural-Enhanced Live Streaming: Improving Live Video Ingest via Online Learning

## 摘要

用神经网络在服务端增强直播上传视频流的清晰度。

在线训练将低清晰度视频转成高清晰度视频的神经网络。直播上传流主要上传低清晰度视频，同时上传少量高清晰度视频供训练。训练和推断同时进行。

第一次将神经网络视频增强用在直播上传流中，这个场景最大的特点就是要训练和视频增强必须同时进行。

- 怎么让在线训练快点收敛？
- 如何调节上行数据发送速率？
- 如何规划上行带宽的使用？