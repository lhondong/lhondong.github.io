---
title: "Scale Space"
subtitle: "图像的 Scale 和 Resolution 区别"
layout: post
author: "L Hondong"
header-img: "img/post-bg-36.jpg"
mathjax: true
tags:
  - 笔记
---

## 尺度空间

### 概念

尺度空间方法的基本思想是：在视觉信息处理模型中引入一个被视为尺度的参数，通过连续变化尺度参数获得不同尺度下的视觉处理信息，然后综合这些信息以深入地挖掘图像的本质特征。尺度空间方法将传统的单尺度视觉信息处理技术纳入尺度不断变化的动态分析框架中，因此更容易获得图像的本质特征。尺度空间的生成目的是模拟图像数据多尺度特征。高斯卷积核是实现尺度变换的唯一线性核。

### 直观理解

图像的尺度是指图像内容的粗细程度。尺度的概念是用来模拟观察者距离物体的远近的程度，具体来说，观察者距离物体远，看到物体可能只有大概的轮廓；观察者距离物体近，更可能看到物体的细节，比如纹理，表面的粗糙等等。**从频域的角度来说，图像的粗细程度代表的频域信息的低频成分和高频成分**。粗质图像代表信息大部分都集中在低频段，仅有少量的高频信息。细致图像代表信息成分丰富，高低频段的信息都有。  

**尺度空间又分为线性尺度空间和非线性尺度空间**。在本文中，仅仅讨论线性尺度空间的构造，非线性尺度空间不在本文的讨论之中。在数学上，空间 (space) 指的是具有约束条件的集合 (set)。而图像的尺度空间是指同一张图像不同尺度的集合。在该集合中，粗尺度图像不会有细尺度图像中不存在的信息，换言之，任何存在于粗尺度图像下的内容都能在细尺度图像下找到，细尺度图像通过 filter 形成粗尺度图像过程，不会引入新的杂质信息。粗尺度图像形成的过程是高频信息被过滤的过程，smooth filter 理所当然成为首选，而加入不引入信息杂质的线性滤波器结构约束，通过证明，**高斯核便是实现尺度变换的唯一线性核**。

由此可见，图像的尺度空间是一幅图像经过几个不同高斯核后形成的模糊图片集合，用来模拟人眼看到物体的远近程度，模糊程度。**注意：图像尺度的改变不等于图像分辨率在改变，下图便是很好的例子，图像的分辨率是一样的，但是尺度却不一样**。

```python
from skimage import data, filters,io
import matplotlib.pyplot as plt
%matplotlib inline

image = io.imread('/Dataset/mxnet_mtcnn_face_detection/anner.jpeg')
img1 = filters.gaussian(image, sigma=1.0)
img2 = filters.gaussian(image, sigma=2.0)
img3 = filters.gaussian(image, sigma=3.0)

plt.figure('gaussian',figsize=(8,8))
plt.subplot(221)
plt.imshow(image)
plt.axis('off')
plt.title('original image')
plt.subplot(222)
plt.imshow(img1)
plt.axis('off')
plt.title('gaussian kernel with sigmma=1.0')
plt.subplot(223)
plt.imshow(img2)
plt.axis('off')
plt.title('gaussian kernel with sigmma=2.0')
plt.subplot(224)
plt.imshow(img3)
plt.title('gaussian kernel with sigmma=3.0')
plt.axis('off')
```

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-47-08.png" alt="ScaleSpace-2022-04-04-10-47-08" style="zoom:100%;" /></div>

### 为什么需要尺度空间？Motivation

研究表明，物体在不同的尺度下能展现出不同的结构，如：粗尺度图片能够更高的体现物体的轮廓和形态，细尺度图片能更有效表示物体的局部细节特征。尺度对于图片来说，就是一种“measurement", 像是一种可调节的放大镜。计算机进行图片分析时，可使用这个放大镜观察图片的宏观与微观世界，从而提取出自己的 interesting points.

- 现实世界的物体由不同尺度的结构所组成；
- 在人的视觉中，对物体观察的尺度不同，物体的呈现方式也不同；
- 对计算机视觉而言，无法预知某种尺度的物体结构是否有意义，因此有必要将所有尺度的结构表示出来；
- 从测量的角度来说，对物体的测量数据必然是依赖于某个尺度的，例如温度曲线的采集，不可能是无限的，而是在一定温度范围进行量化采集。温度范围即是选择的尺度；
- 采用尺度空间理论对物体建模，即将尺度的概念融合入物理模型之中。

## 图像的分辨率

图像的分辨率 (Image Resolution) 本质上是图像的在水平和垂直方向的量化程度，直观上理解是指图像能展现的细节程度。量化的过程是模拟信号转变成数字信号的过程，这一过程是**不可逆**的信息损失过程。因此，量化级别的高低决定了该数字信号能否更好的表示原本的模拟信号。图像是二维数组，水平像素和垂直像素的数量便是图像量化的级别，多像素图像更能展示图像的细节。如下图：

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-52-29.png" alt="ScaleSpace-2022-04-04-10-52-29" style="zoom:100%;" /></div>

## 图像金字塔

图像金字塔 (image pyramid) 是同一张图片不同分辨率的集合。大尺度原图在底层，越往上，尺度逐渐减小，堆叠起来便形成了金字塔状。

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-52-53.png" alt="ScaleSpace-2022-04-04-10-52-53" style="zoom:100%;" /></div>

在金字塔的每一层图片上可以进行尺度解析，即：用不同 $\sigma$ 的高斯核去处理每一层图片，从而形成一个“octve”, 图像的尺度解析和图像金字塔便形成了图片的多尺度多分辨率解析的基础。

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-53-29.png" alt="ScaleSpace-2022-04-04-10-53-29" style="zoom:100%;" /></div>

图像金字塔又分为**高斯金字塔（低通）和拉普拉斯金字塔（带通）**。

- 高斯金字塔：使用一个高斯滤波器对原图进行平滑滤波，并对其进行下采样（下采样因子通常为 2)。重复此步骤，可以得到一系列不同尺度，不同分辨率的图像集合。并将此集合按照图像大小从底部开始堆叠，便是高斯金字塔。  
- 拉普拉斯金字塔：高斯金字塔的相邻两层相互做差运算，由于相邻两层的分辨率不一样，在做差之前，低分辨率图片需要进行插值运算，因此拉普拉斯金字塔便是高斯金字塔相邻两层的差。

```python
import matplotlib.pyplot as plt

from skimage import data, transform,io
from skimage.transform import pyramid_gaussian
import numpy as np

image = io.imread('/Users/xiaojun/Desktop/Programme/DataSet/mxnet_mtcnn_face_detection-master/anner.jpeg')
image = transform.resize(image, [512,512])
rows, cols, dim = image.shape
print(rows,cols)
pyramid = tuple(pyramid_gaussian(image, downscale=2,sigma=3))

composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.axis('off')
plt.show()
```

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-54-26.png" alt="ScaleSpace-2022-04-04-10-54-26" style="zoom:100%;" /></div>

```python
import matplotlib.pyplot as plt

from skimage import data, transform,io
from skimage.transform import pyramid_gaussian, pyramid_laplacian
import numpy as np

image = io.imread('/Users/xiaojun/Desktop/Programme/DataSet/mxnet_mtcnn_face_detection-master/anner.jpeg')
image = transform.resize(image, [512,512])
rows, cols, dim = image.shape
print(rows,cols)
pyramid = tuple(pyramid_laplacian(image, downscale=2,sigma=3))

composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.axis('off')
plt.show()
```

<div align=center><img src="/assets/ScaleSpace-2022-04-04-10-54-43.png" alt="ScaleSpace-2022-04-04-10-54-43" style="zoom:100%;" /></div>
