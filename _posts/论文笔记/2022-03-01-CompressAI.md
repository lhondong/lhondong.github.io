---
title: "CompressAI"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-33.jpg"
mathjax: true
tags:
  - 视频压缩
---

# CompressAI

InterDigital AI Lab 做了一个叫做 CompressAI 的研究平台，基于 Pytorch，是一个提供自定义操作、层、模型和工具以研究、开发和评估端到端图像和视频压缩编解码器的平台，可以降低基于深度学习的图片、视频压缩的入门门槛，方便研究人员。

一个关于 End-to-End Image Compression 的 pytorch 库，复现以下四篇论文的模型。

1.  End-to-end Optimized Image Compression
2.  Variational Image Compression With A Scale Hyperprior
3.  Joint Autoregressive and Hierarchical Priors for Learned Image Compression
4.  Learned Image Compression with Discretized Gaussian Mixture Likelihoods and  Attention Modules

[开源](https://github.com/InterDigitalInc/CompressAI/)

CompressAI 有以下几个特点：

1. 构建端到端压缩的神经网络；
2. 提供了一些预训练的 Model zoo；
3. 开发了一些实用的命令行程序；
4. 实现了一些评估工具。

## 安装

1.  安装 Cudatoolkit 和 Cudnn 包，需要和 Pytorch 版本相对应。
2.  安装 Pytorch。
3.  下载 CompressAI 库，并且用 pip 进行安装。
4.  检测安装结果。

用 conda 创建 python=3.8, cuda=10.1 的虚拟环境（建议）

```bash
conda create -n env_name python=3.8 cudatoolkit=10.1 cudnn
```

创建环境后需要激活虚拟环境，以在该虚拟环境下下载对应的库包。

```bash
conda activate env_name
```

如果使用基础的环境，则直接通过 conda 下载 cuda 即可

```bash
conda install cudatoolkit=10.1 cudnn
```

```bash
conda install pytorch torchvision torchaudio
```

torch 下载完成后，根据官方指导，开始下载 CompressAI, 从 clone 工程到你的机器上，下载结束后，进行 pip 安装。

```bash
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install -U pip && pip install -e . //该命令用下一条命令替换，更快地安装。   
pip install -e . -i https://pypi.douban.com/simple  //pip 豆瓣源比清华源好
```

安装结束后，输入

```bash
conda list  //查看该环境下的安装包，如果出现 compressai，即安装成功
```

用 python 验证也可

```bash
python
import compressai //不报错即安装成功
```

## 使用

CompressAI 的具体使用 [API 指导](https://interdigitalinc.github.io/CompressAI/)

主要关注两个目录，CompressAI 目录下即 pip 编译的源码，修改这里的代码会修改 CompressAI 的 API 应用， example 目录下的是代码是使用范例。   

### 数据准备

在某个文件夹下准备数据集，/path/to/my/image/dataset/ 表示数据集的目录， 该数据集下分为 train 和 test 目录， train 内部放 train 的 .png 图像， test 放测试图像。

### 训练

-m 指模型， -d 数据集地址，-e epoch 数， --lambda 拉格朗日乘子，–batch-size 训练时的 batchsize 根据数据而定，–patch-size 图像块大小。–cuda 使用 GPU，–save 保存训练好的模型。

```bash
python examples/train.py -m "mbt2018" -d /path/to/my/image/dataset/ -e 100 --lambda 1e-2 --batch-size 32 --test-batch-size 16 --patch-size 256 256 --cuda --save
```

```bash
python examples/train.py -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 16 --cuda --save
```

### 训练结束后需要更新 CDF 保证熵编码的正常运行

```bash
python -m compressai.utils.update_model --architecture mbt2018 checkpoint_best_loss.pth.tar
```

```bash
python -m compressai.utils.update_model [-h] [-n NAME] [-d DIR] [–no-update] [–architecture {factorized-prior,jarhp,mean-scale-hyperprior,scale-hyperprior}] filepath
```

```bash
python -m compressai.utils.update_model  [-d DIR]  [--architecture {factorized-prior,jarhp,mean-scale-hyperprior,scale-hyperprior}] filepath
```

### 评价模型

/path/to/images/folder/ 和上述的不同，该文件夹内直接存储需要 test 的 png 图像。  
-a $ARCH 表示采用的预设定的模型，列表如下六种。

1. bmshj2018_factorized
2. bmshj2018_hyperprior
3. mbt2018
4. mbt2018_mean
5. cheng2020_anchor
6. cheng2020_attn

-p $MODEL_CHECKPOINT 表示存储的网络模型。

```bash
python -m compressai.utils.eval_model checkpoint /path/to/my/image/dataset/test  -a mbt2018 -p checkpoint_best_loss-a57a3f14.pth.tar
```

```bash
python -m compressai.utils.eval_model checkpoint /path/to/images/folder/ -a $ARCH -p $MODEL_CHECKPOINT...
```

## 注意

使用 inference 的时候：

1. 对于 entropy estimation 使用 CUDA 会比使用 CPU 快
2. 对于自回归模型，不建议使用 CUDA 编解码，因为熵编码部分，会在 CPU 上顺序执行。
3. GPU 对非自回归模型推理，在码率估计和实际压缩都能起到加速作用。GPU 对自回归模型不能起到加速左右，因为熵编码是在 CPU 中线性运算编码的。
4. 使用 GPU 或者 CPU，码率估计结果是与实际结果是接近的。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/CompressAI-2022-04-24-10-14-09.png" alt="CompressAI-2022-04-24-10-14-09" style="zoom:100%;" /></div>

### 更新CDF

由于训练结束需要更新 Entropy 的 CDF 以正常进行测试阶段的熵编码工作，但是上述的 CDF 更新制定了预先定义好的框架，当采用自己的框架的时候，CDF 的更新需要自行阅读对应源码并且修改进行 CDF 的更新。

训练好的模型无法更新CDF，此时更改examples/train.py中的save_checkpoint。

```python
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
```

另外保存代码也更新一下：

```python
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
            )
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.module.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename="checkpoint_best_loss.pth.tar"
                )
```