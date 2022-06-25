---
title: "2020 年计算机视觉值得研究的领域"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-5.jpg"
tags:
  - CV
---

# 2020 年计算机视觉值得研究的领域

## [原文](https://www.zhihu.com/question/366016644/answer/971983556)

毫无疑问，3D 方向，是非常值得研究的，包括深度估计，立体匹配，3D 检测（包括单目，双目，lidar 和 rgbd，19 年也终于出现了真正的点云卷积 pointconv），3D 分割，三维重建，3Dlandmark，并且我个人认为如何减少 3D 标注，完全使用多视图几何做是一个很有意义，有前途，并且有挑战的方向。

## 视频方向

更新一下，补充一个非常重要的方向，视频方向，也就是考虑时间维度的 cv，这包括运动目标检测，目标跟踪，运动语义分割。目标跟踪受相关滤波启发的一系列 siamese 工作已经非常漂亮了，剩下运动目标检测，运动语义分割，大体有几种思路：

1. Conv+LSTM（memory based），slow fast 架构，还有两者的结合，另外还有基于光流的架构，在已知光流的情况下，通过前向 warp 或者后向 warp，能在时间维度上前后转移 featuremap，这是基本的出发点。个人其实挺喜欢光流的，因为如果不追求 end2end 的话，光流可以被用在很多地方（**当然，如果考虑时间的话，memory based 方法产生的 feature map 也可以用在其他任何地方，只是不像光流那样可以从网络里面拆出来**），当然对于特别追求精度的地方，e2e 会更好。memory based 方面的工作我个人非常推崇 google 的 looking fast and slow。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-52-37.png" alt="Research_fields_of_CV_2020-2022-01-11-20-52-37" style="zoom:30%;" /></div>

memory 结合 slowfast，fast 的参数一般很少。架构是通用的，修改 head 它能被用在其他任何 task 上。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-53-43.png" alt="Research_fields_of_CV_2020-2022-01-11-20-53-43" style="zoom:50%;" /></div>

slowfast 交错在一起（并且可以是异步的），能同时提高检测分割等其他各类任务的精度和速度。

1. **当然光流也可以 e2e，光流完全可以作为 Conv+LSTM 或者 slowfast 的旁支输出，然后作用在 featuremap 上**，但是一般深度学习光流的计算量都比较大，需要在一个比较大的区域内做匹配。并且如果联合训练的话，flow 本身的自监督算法不一定是使用，比如 unflow 之类的算法。
2. **memory based 和 flow based 方法的结合点会非常有趣**，或者说是否可以通过 memory 去估计 flow，因为 memory 可以和 slowfast 架构结合，从而减小计算量。
3. **3D 卷积，随着 TCN 崛起成为新的序列建模方法，时间卷积和空间卷积可以合成成为 3D 卷积**，另外 slowfast 架构里面，fast 可以看成 dilation rate 更大的时间卷积，这方面的代表工作有 C3D，I3D，TSN 等，另外不得不提 19 年的 Temporal Shift Module，它利用了时间卷积基本都是前向这个特点，用移位极大的减小了计算量。从数字图像开始，本人就是卷积的忠实粉丝，我个人热爱一切全卷积架构。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-54-07.png" alt="Research_fields_of_CV_2020-2022-01-11-20-54-07" style="zoom:30%;" /></div>

光流可以把 feature map 在时间维度上前向后向 warp。这决定了 flow 的另一个好处，它能找到两帧计算结果之间的对应关系。flow 的缺点是计算量可能会稍大。

## 3D 方向

3-D 部分具体说来包括

1. 单目深度估计如何提高计算性能，如何提高自监督的鲁棒性，前者有 fastdepth，在 tx2 上已经能达到 140fps 的预测性能，后者包括 monodepth2 ，Struct2depth 和 geonet 等一系列工作，使用多视图几何和运动估计来进行自监督，loss 一般都是重投影误差的推广。Struct2depth 使用了一个预训练的实例分割模型，这样做的好处是能够单独建模每个物体的运动，**另外和分割的联合能让深度估计 aware 物体的尺度，物体的大小通常和深度有直接联系，geonet 使用刚性流来替代光流，将相机运动和物体运动隔离开，同时预测物体深度和刚性流**。进一步的发展一定是在线训练，在相机运动的过程中自我训练改进。
2. 立体匹配的话，如何解决低纹理区域处的匹配，如何和语义分割联合，如何提高计算性能。  
立体匹配方面的在线训练模型已经出现了，就是 madnet，19 年的 CVPR oral，仔细看了一下基本没用 3D Conv，所以会不会还有改进的空间也是很有意思的，madnet 冻结一部分网络，在线训练只训练底层的几个 adaption domain。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-55-46.png" alt="Research_fields_of_CV_2020-2022-01-11-20-55-46" style="zoom:100%;" /></div>

<center>在线训练的意思是在运行的时候训练</center>

1. 3D 检测点的话，全卷积的架构，如何 fuse 不同传感器的信息，比如 fuse camera 和 lidar 点云，19 年出现真正的点云卷积，pointconv 和 kpconv，相信能为点云分割和 3D 检测带来更丰富的内容。双目的话，MSRA 今年有一篇论文，triangulation learning network，个人感觉很惊艳，使用 2D anchor 来引导 3D anchor。单目 6D 姿态估计的话，还需要补充。
2. 3D landmark，自监督的方法，如何提高性能，代表性的工作有 learable triangulation of human pose，最惊艳的是它的 volumetric triangulation，直接将 2D heatmap 投影到 3D heatmap，然后使用 3D convnet refine heatmap，个人感觉是一个非常优的架构，但是是否还可以考虑投影 part affinity 呢，目前 part affinity 代表一个向量，投影回三维有很严重的不唯一性问题，因为从三维的一个点投影到二维，有很多可能性得到同一个向量，考虑非向量的 part affinity 是否可以，也是可以思考的。**这里我想到的是直接在二维情况下估计一个 3D 的 paf 出来，然后重投影到 volume 里，也可以估 2D 的 paf，然后重投影的时候认为 paf 的第三个分量为 1，后面再用 3D convnet refine。**

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-58-15.gif" alt="Research_fields_of_CV_2020-2022-01-11-20-58-15" style="zoom:100%;" /></div>

<center>重投影过程</center>

这样的重投影也许也能用来重投影 featuremap，但是 volume 的大小和分辨率与 task 直接相关，从而直接影响计算量。一个直接的改进是给多视图每个 featuremap 一个 weightmap，也就是每个点一个权重，加权融合到一起。

这是一个非常好的架构，直接把 2D 提升到了 3D，可能被用在多视角的各个领域，包括三维重建，并且最后的结果可以投影回原视角，做自监督，缺点可能是计算量会比较大。

MSRA 的一篇论文 cross view fusion of human pose 也很惊艳，**使用 epipolar 几何来融合不同视角的 2D heatmap 达到互相改进的效果，个人感觉这一点不止可以用在 landmark 上**（凡是使用了 heatmap 的地方都可以考虑用这种方式 fuse，其实不止如此，这个方法会把一个视图里的极限上所有的 heatmap 值通过一个权重矩阵 w 加权相加到另一个视图的极线上的点，而这个矩阵本质上是全局的，可能只和对极几何相关，它是否能被用来 fuse featuremap 个人感觉是非常有意思的一件事，但是这个计算量应该会很大）。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-58-53.png" alt="Research_fields_of_CV_2020-2022-01-11-20-58-53" style="zoom:100%;" /></div>

fuse 可能只和对极几何相关，并能够被用在其他地方，但是计算量会大很多。我跟作者交流过这个方法，可行性是有的，但是问题是参数冗余和计算量大，很明显的其实作者自己也说过，这种连接方式应该沿着极线，而不是所有像素都连接上。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-21-06-04.png" alt="Research_fields_of_CV_2020-2022-01-11-21-06-04" style="zoom:50%;" /></div>

<center>fuse 只和对极几何有关</center>

这里还推另一篇文章 DAVANet: Stereo Deblurring with View Aggregation，这是双目去模糊的，主要思路是使用 dispnet 提取左右视差，然后将左右 featuremap 进行 warp 然后经过 fusion layer，这里面有一点问题是，dispnet 的监督其实和其他分支是半独立的，fusion layer 里面也会考虑把这个 dispmap concat 起来。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-20-59-30.png" alt="Research_fields_of_CV_2020-2022-01-11-20-59-30" style="zoom:50%;" /></div>

<center>先估计视差，然后利用视差进行 fusion 也许才是更合理的做法</center>

dispnet 的计算量会比较大，双目特征融合还有一种方法被称为 stereo attention，主要思路就是生成两个 mask，这个 mask 表示对应视图极线上的 feature 对本视图的贡献的权重。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-21-02-56.png" alt="Research_fields_of_CV_2020-2022-01-11-21-02-56" style="zoom:100%;" /></div>

关于 3D pose，另外有一篇 epipolarpose 也是自监督的，epipolarpose 使用 2D pose 和对极几何来自监督生成 3D posr。

1. 三维重建的话，如何提升重建细节，是否有自监督的算法，代表性的工作有 pointmvsnet，rmvsnet。相信 meshcnn 的出现能被应用到重建里。

<div align=center><img src="https://lhondong-pic.oss-cn-shenzhen.aliyuncs.com/img/assets/Research_fields_of_CV_2020-2022-01-11-21-03-20.png" alt="Research_fields_of_CV_2020-2022-01-11-21-03-20" style="zoom:50%;" /></div>

另外，自监督的 mvsnet 果然已经出来了。

1. 深度和光流的联合训练，19 年有一篇论文 bridge optical flow and depth estimation。3D 的 flow，sceneflow 也就是场景流，这里待补充。
2. 自监督学习很重要，尤其在一些很难获得标注的场景，比如上面说到的立体匹配，深度估计，我在做的时候还遇到过特征点检测和描述子生成，自监督学习通常要有一个定义良好的，即使没有监督数据也能反应问题的 loss。弱监督学习也很重要，比如在分割这种标注比较困难的场景。

还有一些传统方法做的比较好的领域也可以尝试，图像自动曝光，图像增强，特征点提取。

细粒度识别，19 年的 learn to navigate，个人觉得如何构建一个可微分的子模块是一个有意思的问题，难点在 nms 通过 attention module 或者 learned nms 或许有这个希望。centernet 出现之后没有 nms，或许会改进这个问题。

再补一个方向，scene parsing，如何利用物体之间的先验关系建模提高检测，反过来是否可以帮助无监督或者弱监督学习。

最后补上我个人的一些想法，**深度学习如如何高效使用数据，如何做更好的 multitasking，一个网络，如果既有检测头，又有分割头，我们希望图像本身既有检测又有分割标注，但是实际上一般是一部分有检测标注，一部分有分割标注，如何发明出一个更好的训练算法充分利用数据是个很大的问题**。我个人探索过交错训练法，也就是以不同的采样率分别训练不同的头，只要数据没有语义冲突，类似的想法应该能 work。

**总结一下就是，考虑时间连贯性，考虑多视角，考虑新的传感器和传感器之间的融合，更好的 multitasking，更好的训练方法使得数据能被更好的利用，自监督和弱监督的算法，轻量化网络。**
