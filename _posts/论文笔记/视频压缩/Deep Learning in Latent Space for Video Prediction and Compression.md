# Deep Learning in Latent Space for Video Prediction and Compression

先做图像压缩，然后使用前一帧的latent representation预测下一帧的latent representation。然后做 element-wise的残差，如果预测的很好的话，残差是稀疏的，熵很低。