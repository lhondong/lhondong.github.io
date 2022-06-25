---
title: "Python 读取图片"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-26.jpg"
tags:
  - Python
  - 读取图片
---

# Python Image Read

### cv2.imread

```python
import cv2
img = cv2.imread(path)
```

### PIL

```python
from PIL import Image
img = Image.open(path)
```

### matplolib

```python
import matplotlib.pyplot as plt
img = plt.imread(path)
```

### skimage.io

```python
import skimage.io as io
img = io.imread(path)
```

### scipy.misc

```python
from scipy.misc import imread
img = imread(path)
```

- 类型上，除了 PIL 特殊，为 `PIL.JpegImagePlugin.JpegImageFile`，其他 4 种读取的图片格式均为 `numpy.ndarray` 格式；
- 维度上，除了 PIL 其它都是 `H,W,C`，PIL 是 `W,H,C`。pytorch 是 `N,C,H,W`，tensorflow 是 `N,H,W,C`
- 通道上，除了 opencv(cv2) 读进来的顺序是 `BGR`，其他都是 `RGB`。

## 显示图片

- matpltlib.pyplot（plt）显示 numpy 数组格式的 RGB 图像或者 tensor 格式图片。如果是 float32 类型的图像，范围 0-1；如果是 uint8 图像，范围是 0-255；plt.imshow(image,cmap = ‘gray’)，灰度图显示要设置 cmap 参数，显示 cv2 的图像需要转换通道为 RGB。
- python 自带的 show()，显示 PIL 读取的图片
- cv.imshow()，显示 numpy 格式的图片，显示的图片通道顺序和 cv2.imread() 读取得到的图片的通道顺序一样，要求是 BGR。

## PIL,ndarray,tensor 三者的转换

PIL.Image/numpy.ndarray 转化为 Tensor，常常用在训练模型阶段的数据读取，而 Tensor 转化为 PIL.Image/numpy.ndarray 则用在验证模型阶段的数据输出。tensor 分为 gpu 上和 cpu 上的，GPU 上的 tensor 不能直接转换为 numpy，需要先转换为 CPU 上的 tensor。ndarray（cv2）就是各种格式之间的中转）。

### PIL->ndarray

```python
img = Image.open('path')
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv.imshow('img', img)
cv2.waitKey()
```

### ndarray->PIL

ndarray 转 PIL 要求数据类型 dtype=uint8, range[0, 255] and shape H×W×C

```python
img = cv2.imread('path')
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)  #fromarray 并不会做通道变换
img.show(img)
```

### PIL->tensor

```python
def PIL_to_tensor(img):
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    return image.to(device, torch.float)
```

把像素值范围为 [0, 255] 的 PIL.Image 或者 numpy.ndarray 型数据，shape=(H×W×C) 转换成的像素值范围为 [0.0, 1.0] 的 torch.FloatTensor，shape 为 (N×C×H×W)。

对于 PILImage 转化的 Tensor，其数据类型是 torch.FloatTensor。

Image.open 返回的图片类型为 PIL Image, 数值类型为 uint8，值为 0-255，尺寸为 W×H×C（宽度高度通道数）。通过 img=np.array(img) 转为 numpy 数组后，统一尺寸为 H×W×C。

### tensor->PIL

tensor 转 PIL, 要求 tensor 必须是 float 类型的，为 C×H×W 格式，double 的不可以。

```python
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = torchvision.transforms.ToPILImage()(image)
    return image
```

### ndarray->tensor

对 ndarray 的数据类型没有限制，但转化成的 Tensor 的数据类型是由 ndarray 的数据类型决定的。

```python
def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    # or: img = torch.from_numpy(image).permute(2, 0, 1)
    return img.float().div(255).unsqueeze(0)
```

不与输入共享内存的：

- torch.Tensor(data)：会给生成 tensor 默认 float32 类型，且不能用 dtype 参数进行修改
- torch.tensor(data)：会根据输入 ndarray 数据类型自动推断，且能通过 dtype = … 修改生成 tensor 类型

与输入共享内存的（共用一个数据地址，会修改调原始数据）：

- torch.as_tensor(ndarray)：会自动推断类型，接受 ndarray 类型及 tensor 类型。注意因为 ndarray 是放在 cpu 上的，若用 GPU 则需要从 cpu copy 到 GPU。共享内存对 python 的内置类型如 list 等不支持。
- torch.from_numpy(data)：会自动推断类型，只接受 numpy 的 ndarray 类型

### tensor—>ndarray

```python
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().suqeeze(0).transpose((1, 2, 0))
    return img
```

## Dataset 定义

```python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
 
import cv2
from PIL import Image
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transforms.Compose([
            transforms.ToTensor()      # 这里仅以最基本的为例
        ])
        self.transform = transforms.Compose([
            transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224), # 从图片中间切出224*224的图片
            transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
        ])
        images_path = Path(root)
        
        images_list = list(images_path.glob('*.jpg')) # list(images_path.glob('*.png'))
        images_list_str = [ str(x) for x in images_list ]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        
        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        image = self.transform(image)   # transform 转化 image 为：C, H, W

        label = 1 if 'dog' in image_path.split('\\')[-1] else 0 # 这是一个label的示例，可自定义
        return image, label

    def __len__(self):
        return len(self.images)
```

## PIL 用法

### 图片加载、灰度图、 显示和保存

```python
from PIL import Image
 
img = Image.open('01.jpg')
imgGrey = img.convert('L')
 
img.show()
imgGrey.show()
 
img.save('img_copy.jpg')
imgGrey.save('img_gray.jpg')
```

### 图片宽、高、通道模式、平均值获取

```python
from PIL import Image
import numpy as np
 
img = Image.open('01.jpg')
 
width, height = img.size
channel_mode = img.mode
mean_value = np.mean(img)
 
print(width)
print(height)
print(channel_mode)
print(mean_value)
```

### 创建指定大小，指定通道类型的空图像

```python
from PIL import Image
 
width = 200
height = 100
 
img_white = Image.new('RGB', (width,height), (255,255,255))
img_black = Image.new('RGB', (width,height), (0,0,0))
img_L = Image.new('L', (width, height), (255))
 
img_white.show()
img_black.show()
img_L.show()
```

### 访问和操作图像像素

```python
from PIL import Image
 
img = Image.open('01.jpg')
 
width, height = img.size
 
# 获取指定坐标位置像素值
pixel_value = img.getpixel((width/2, height/2))
print(pixel_value)
 
# 或者使用load方法
pim = img.load()
pixel_value1 = pim[width/2, height/2]
print(pixel_value1)
 
# 设置指定坐标位置像素的值
pim[width/2, height/2] = (0, 0, 0)
 
# 或使用putpixel方法
img.putpixel((w//2, h//2), (255,255,255))
 
# 设置指定区域像素的值
for w in range(int(width/2) - 40, int(width/2) + 40):
    for h in range(int(height/2) - 20, int(height/2) + 20):
        pim[w, h] = (255, 0, 0)
        # img.putpixel((w, h), (255,255,255))
img.show()
```

### 图像通道分离和合并

```python
from PIL import Image
 
img = Image.open('01.jpg')
 
# 通道分离
R, G, B = img.split()
 
R.show()
G.show()
B.show()
 
# 通道合并
img_RGB = Image.merge('RGB', (R, G, B))
img_BGR = Image.merge('RGB', (B, G, R))
img_RGB.show()
img_BGR.show()
```

### 在图像上输出文字

```python
from PIL import Image, ImageDraw, ImageFont
 
img = Image.open('01.jpg')
 
# 创建Draw对象:
draw = ImageDraw.Draw(img)
# 字体颜色
fillColor = (255, 0, 0)
 
text = 'print text on PIL Image'
position = (200,100)
 
draw.text(position, text, fill=fillColor)
img.show()
```

### 图像缩放

```python
from PIL import Image
 
img = Image.open('01.jpg')
 
width, height = img.size
 
img_NEARESET = img.resize((width//2, height//2))  # 缩放默认模式是NEARESET(最近邻插值)
img_BILINEAR = img.resize((width//2, height//2), Image.BILINEAR)  # BILINEAR 2x2区域的双线性插值
img_BICUBIC = img.resize((width//2, height//2), Image.BICUBIC)  # BICUBIC 4x4区域的双三次插值
img_ANTIALIAS = img.resize((width//2, height//2), Image.ANTIALIAS)  # ANTIALIAS 高质量下采样滤波
```

### 图像遍历操作

```python
from PIL import Image
 
img = Image.open('01.jpg').convert('L')
 
width, height = img.size
 
pim = img.load()
 
for w in range(width):
    for h in range(height):
        if pim[w, h] > 100:
            img.putpixel((w, h), 255)
            # pim[w, h] = 255
        else:
            img.putpixel((w, h), 0)
            # pim[w, h] = 0
            
img.show()
```

###  图像阈值分割、 二值化

```python
from PIL import Image
 
img = Image.open('01.jpg').convert('L')
 
width, height = img.size
 
threshold = 125
 
for w in range(width):
    for h in range(height):
        if img.getpixel((w, h)) > threshold:
            img.putpixel((w, h), 255)
        else:
            img.putpixel((w, h), 0)
 
img.save('binary.jpg')
```

### 图像裁剪

```python
from PIL import Image
 
img = Image.open('01.jpg')
 
width, height = img.size
 
# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (100, 100, 550, 350)
 
region = img.crop(box)
 
region.save('crop.jpg')
```

### 图像边界扩展

```python
from PIL import Image
 
img = Image.open('test.png')
 
width, height = img.size
channel_mode = img.mode
 
img_makeBorder_full = Image.new(channel_mode, (2*width, height))
img_makeBorder_part = Image.new(channel_mode, (width+200, height))
 
# 图像水平扩展整个图像
img_makeBorder_full.paste(img, (0, 0, width, height))
img_makeBorder_full.paste(img, (width, 0, 2*width, height))
img_makeBorder_full.show()
 
# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (width-200, 0, width, height)
region = img.crop(box)
 
# 图像水平右侧扩展一个ROI
img_makeBorder_part.paste(img, (0, 0, width, height))
img_makeBorder_part.paste(region, (width, 0, width+200, height))
img_makeBorder_part.show()
```

```python

```

```python

```

```python

```