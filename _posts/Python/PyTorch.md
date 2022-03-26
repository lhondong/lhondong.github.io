---
title: "PyTorch"
subtitle: ""
layout: post
author: "L Hondong"
header-img: "img/post-bg-48.jpg"
mathjax: ture
tags:
  - PyTorch
  - 算法
---

[TOC]

# 一、PyTorch 基础

## 1. 张量

张量是一个**多维数组**，它是标量、向量、矩阵的高维拓展。

### 1.1 Tensor 与 Variable

Variable 是 torch.autograd 中的数据类型，主要用于封装 Tensor，进行**自动求导**。

- data：被包装的 Tensor
- grad：data 的梯度
- grad_fn：创建 Tensor 的 Function，是**自动求导**的关键
- requires_grad：指示是否需要梯度 
- is_leaf：指示是否是叶子结点（张量）

PyTorch 0.4.0 版开始，Variable 并入 Tensor。（多了三个属性，共八个，其中四个与数据相关，四个与梯度求导相关）

- dtype：张量的数据类型，如 torch.FloatTensor, torch.cuda.FloatTensor
- shape：张量的形状，如 (64, 3, 224, 224)
- device：张量所在设备，GPU/CPU，是加速的关键

### 1.2 Tensor 创建

#### 1. 直接创建

torch.tensor()

功能：从 data 创建 Tensor

- data：数据，可以是 list，numpy
- dtype：数据类型，默认与 data 的一致
- device：所在设备，cuda/cpu
- requires_grad：是否需要梯度
- pin_memory：是否存于锁业内存

```python
torch.tensor(data,          
             dtype=None,
             device=None,
             requires_grad=False,
             pin_memory=False)
```

torch.from_numpy(ndarray)

功能：从 numpy 创建 tensor

注意事项：从 torch.from_numpy 创建的 tensor 于原 ndarray 共享内存，当修改其中一个的数据，另外一个也将会被改动。

#### 2. 依据数值创建

torch.zeros()

- size：张量的形状，如 (3, 3)、(3, 224,224)
- out：输出的张量
- layout：内存中布局形式，有 strided，sparse_coo 等
- device：所在设备，gpu/cpu
- requires_grad：是否需要梯度

```python
torch.zeros(*size,
            out=None,
            dtype=None, 
            layout=torch.strided, 
            device=None, 
            requires_grad=False)
```

torch.zeros_like(input)

功能：依 input 形状创建全 0 张量

torch.ones()

torch.ones_like()

torch.full()

torch.full_like()

```python
torch.full(size,
           fill_value,
           ...)
torch.full((3,3), 10)
```

torch.arange()

功能：创建等差的 1 维张量

注意事项：数值区间为 [start, end)

- start：数列起始值
- end：数列“结束值”
- step：数列公差，默认为 1

```python
torch.arrange(2, 10, 2)
输出：tensor([2,4,6,8]) #没有 10
```

torch.linspace() 

功能：创建均分的 1 维张量 

注意事项：数值区间为 [start, end]

- start：数列起始值
- end：数列结束值 
- steps：数列长度

```python
torch.linspace(2, 10, 5)
输出：tensor([2., 4., 6., 8., 10.])
torch.linspace(2, 10, 6)
输出：tensor([2.0000, 3.6000 ,5.2000, 6.8000, 8.4000, 10.0000])
```

torch.logspace()

功能：创建对数均分的 1 维张量 

注意事项：长度为 steps, 底为 base 

- start：数列起始值
- end：数列结束值
- steps：数列长度
- base：对数函数的底，默认为 10

torch.eye() 

功能：创建单位对角矩阵 (2 维张量） 

注意事项：默认为方阵

- n：矩阵行数
- m：矩阵列数

#### 3. 依概率分布创建张量

##### torch.normal() 

功能：生成正态分布（高斯分布）

- mean：均值
- std：标准差

四种模式：

- mean 为标量，std 为标量 （需要设置 size）
- mean 为标量，std 为张量 
- mean 为张量，std 为标量 
- mean 为张量，std 为张量

##### torch.randn()

##### torch.randn_like()

功能：生成**标准正态分布** 

- size：张量的形状

##### torch.rand()

##### torch.rand_like()

功能：在区间 [0, 1) 上，生成**均匀分布** 

##### torch.randint()

##### torch.randint_like()

功能：区间 [low, high) 生成整数均匀分布 

- low
- high
- size：张量的形状

##### torch.randperm() 

功能：生成生成从 0 到 n-1 的随机排列

- n：张量的长度

##### torch.bernoulli()

功能：以 input 为概率，生成伯努力分布 (0-1 分布，两点分布）

- input：概率值

### 1.3 Tensor 操作

#### 1. 张量拼接与切分

#####  torch.cat()

功能:将张量按维度dim进行拼接 

- tensors:张量序列

- dim:要拼接的维度

##### torch.stack()

功能:在新创建的维度dim上进行拼接 

- tensors:张量序列
- dim:要拼接的维度

```python
a = torch.ones((2,3))
a_stack = torch.stack([a,a],dim = 2)
输出：
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]]) #新建一个维度，(2,3,2)
a_stack = torch.stack([a,a],dim = 0)
输出：
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]]) #在第0维新建，(2,2,3)
```

##### torch.chunk()
功能:将张量按维度dim进行平均切分

返回值:张量列表

注意事项:若不能整除，最后一份张量小于其他张量

- input: 要切分的张量
- chunks : 要切分的份数 
- dim : 要切分的维度

```python
a = torch.ones((2,5))
list_of_tensor = torch.chunk(a, dim=1, chunks=2)

for idx,t in enumerate(list_of_tensor):
    print("tensor {}: {}".format(idx, t))
输出：
tensor 0: tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor 1: tensor([[1., 1.],
        [1., 1.]]) #切分为两个张量，最后一个比较小
```

##### torch.split()

功能:将张量按维度dim进行切分

返回值:张量列表

- tensor: 要切分的张量
- split_size_or_sections : 为int时，表示每一份的长度;为list时，按list元素切分
- dim : 要切分的维度

#### 2. 张量索引

##### torch.index_select()

功能:在维度dim上，按index索引数据

返回值:依index索引数据拼接的张量

- input: 要索引的张量

- dim: 要索引的维度
- index : 要索引数据的序号

##### torch.masked_select()

功能:按mask中的True进行索引

返回值:一维张量

- input: 要索引的张量
- mask: 与input同形状的布尔类型张量

```python
a = torch.randint(0, 9, (3,3))
idx = torch.tensor([0,2], dtype=torch.long)
a_select = torch.index_select(a, dim=0, index=idx)

print(a, '\n',a_select)
输出：
tensor([[2, 2, 4],
        [6, 2, 0],
        [3, 4, 4]]) 
tensor([[2, 2, 4],
        [3, 4, 4]]) #从第0个维度，选出第0行和第2行
```

##### torch.masked_select()

功能:按mask中的True进行索引

返回值:一维张量

- input: 要索引的张量
- mask: 与input同形状的布尔类型张量

```python
a = torch.randint(0, 9, (3,3))
mask = a.ge(5) # ge means greater or equal/ gt means greater than/le,lt
a_select = torch.masked_select(a, mask)

print(a, '\n', a_select)
输出：
tensor([[2, 1, 5],
        [0, 6, 3],
        [8, 8, 2]]) 
 tensor([5, 6, 8, 8])
```

#### 3. 张量变换

##### torch.reshape()

功能:变换张量形状

注意事项:当张量在内存中是连续时，新张量与input共享数据内存

- input: 要变换的张量
- shape: 新张量的形状

##### torch.transpose()

功能:交换张量的两个维度 

- input: 要变换的张量
- dim0: 要交换的维度
- dim1: 要交换的维度

##### torch.t() 

功能:2维张量转置，对矩阵而言，等价于 torch.transpose(input, 0, 1)

##### torch.squeeze()

功能:压缩长度为1的维度(轴)

- dim: 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除;

##### torch.unsqueeze() 

功能:依据dim扩展维度 

- dim: 扩展的维度

#### 4. 张量数学运算

##### torch.add()

功能:逐元素计算 input+alpha×other

- input: 第一个张量
- alpha: 乘项因子
- other: 第二个张量

```python
torch.add(input,
          alpha=1,
					other, 
          out=None)
```

##### torch.addcmul() 

$$
out = input+value \times tensor1 \times tensor2
$$

```python
 torch.addcmul(input, 
               value=1, 
               tensor1, 
               tensor2,
							 out=None)
```

##### torch.addcdiv()

$$
out = input+value \times \frac{tensor1}{tensor2}
$$

torch.sub() 

torch.div() 

torch.mul()

- 用法与*乘法相同，也是element-wise的乘法，也是支持broadcast的。

torch.mm

- 数学里的矩阵乘法，要求两个Tensor的维度满足矩阵乘法的要求。

torch.matmul

- torch.mm的broadcast版本

torch.log(input, out=None) 

torch.log10(input, out=None) 

torch.log2(input, out=None) 

torch.exp(input, out=None) 

torch.pow()

torch.abs(input, out=None) 

torch.acos(input, out=None) 

torch.cosh(input, out=None) 

torch.cos(input, out=None) 

torch.asin(input, out=None)

torch.atan(input, out=None) 

torch.atan2(input, other, out=None)

## 2. 线性回归

线性回归是分析一个变量与另外一(多)个变量之间关系的方法

求解步骤: 

1. 确定模型 Model: y = wx + b 
2. 选择损失函数 MSE:
3. 求解梯度并更新 $w,b$

$$
w = w – LR \times w.grad \\ b = b – LR \times w.grad
$$

```python
# 线性回归模型
import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)

lr = 0.1 

# 创建训练数据
x = torch.rand(20,1) * 10
y = 2 * x + (5 + torch.randn(20,1)) # 加上噪声

# 构建线性回归参数
w = torch.randn((1),requires_grad=True)
b = torch.zeros((1),requires_grad=True)

for iteration in range(100):

    # 前向传播
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-',lw=5)
        plt.text(2,20,'Loss=%.4f' % loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\n w: {} b:{}" .format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        plt.show()

        if loss.data.numpy() < 1:
            break
```

## 3. 计算图与动态图机制

### 计算图

计算图是用来描述运算的有向无环图 

计算图有两个主要元素：结点(Node)和边(Edge) 

- 结点表示数据，如向量，矩阵，张量

- 边表示运算，如加减乘除卷积等

叶子结点:用户创建的结点称为叶子结点

设置叶子结点主要是为了节省内存，在计算梯度时，只有叶子结点会计算梯度，而非叶子结点内存被释放掉。is_leaf: 指示张量是否为叶子结点

如非叶子结点需要计算梯度，使用retain_grad()来保留梯度不被释放。

grad_fn: 记录创建该张量时所用的方法 (函数)

- y.grad_fn = <MulBackward0> 
- a.grad_fn = <AddBackward0> 
- b.grad_fn = <AddBackward0>

#### 动态图

动态图：运算与搭建同时进行，灵活易调节

静态图：先搭建图，后运算，高效不灵活

## 4. autograd 自动求导

autograd

##### torch.autograd.backward 

功能:自动求取梯度

- tensors: 用于求导的张量，如 loss
- retain_graph : 保存计算图
- create_graph : 创建导数计算图，用于高阶求导
- grad_tensors:多梯度权重

```python
loss = torch.cat([y0,y1], dim=0)
grad_tensor = torch.tensor([1., 2.])

loss.backward(gradient=gradient_tensor) #权重设置为y0+2*y1
```

张量的backward()直接调用torch.autograd.backward()。

使用retain_graph=True可以多次反向传播，不会被内存释放。

##### torch.autograd.grad 

功能:求取梯度

- outputs: 用于求导的张量，如 loss
- inputs : 需要梯度的张量
- create_graph : 创建导数计算图，用于高阶求导
- retain_graph : 保存计算图
- grad_outputs:多梯度权重

```python
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)

grad_1 = torch.autograd.grad(y, x, create_graph=True) #创建导数的计算图，对导数再次求导
print(grad_1)

grad_2 = torch.autograd.grad(grad_1[0], x)
print(grad_2)

输出：
(tensor([6.], grad_fn=<MulBackward0>),)
(tensor([2.]),)
```

##### autograd特性

1. 梯度不自动清零
2. 依赖于叶子结点的结点，requires_grad默认为True 
3. 叶子结点不可执行in-place

梯度清零 

```python
loss.grad.zero_()
```

其中的下划线_表示in-place操作，原地操作(a += 1是原位操作，而a = a + 1不是原位操作)

## 5. 逻辑回归

逻辑回归是线性的二分类模型 

模型表达式：

$$
y = f(WX+b)
$$

Sigmiod 函数：

$$
f(x) = \frac{1}{1+e^{-x}}
$$

$$
\text{class} = \begin{cases}
0, y<0.5\\
1, y\ge 0.5
\end{cases}
$$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

# 生成数据
sample_nums = 100
mean_values = 1.7
bias = 1
n_data = torch.ones((sample_nums, 2))
print(n_data.shape)
x_0 = torch.normal(mean_values * n_data, 1) + bias
x_1 = torch.normal(-mean_values * n_data, 1) + bias
y_0, y_1 = torch.zeros(sample_nums), torch.ones(sample_nums)
train_x = torch.cat((x_0, x_1), 0)
train_y = torch.cat((y_0, y_1), 0)

# 选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR() # 实例化逻辑回归模型

# 选择损失函数
loss_fn = nn.BCELoss()

# 选择优化器
lr = 0.01 # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# 模型训练
for iteration in range(1000):
    
    # 前向传播
    y_pred = lr_net(train_x)

    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清空梯度
    optimizer.zero_grad()

    # 打印训练信息
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze() # 将预测值大于0.5的转换为1，小于0.5的转换为0
        #print(mask)
        correct = (mask == train_y).sum() 
        #print(correct)
        accuracy = correct.item() / train_y.size(0) # 计算正确率

        plt.scatter(x_0.data.numpy()[:, 0], x_0.data.numpy()[:, 1], c='red', label='class 0')
        plt.scatter(x_1.data.numpy()[:, 0], x_1.data.numpy()[:, 1], c='blue', lable='class 1')
        
        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.fewatures.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, accuracy))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if accuracy > 0.99:
            break
```

# 二、PyTorch 数据处理

- 数据收集：Img，Label
- 数据划分：train valid test
- 数据读取：Dataloader
  - Sampler -> Index
  - Dataset -> Img, Label
- 数据预处理：transforms

## 1. Dataloader

##### torch.utils.data.DataLoader 

功能:构建可迭代的数据装载器

- dataset: Dataset类，决定数据从哪读取及如何读取
- batchsize : 批大小
- num_works: 是否多进程读取数据
- shuffle: 每个epoch是否乱序
- drop_last:当样本数不能被batchsize整除时，是否舍弃最后一批数据

关于Epoch，Iteration，Batchsize

- Epoch: 所有训练样本都已输入到模型中，称为一个Epoch 

- Iteration:一批样本输入到模型中，称之为一个Iteration 

- Batchsize:批大小，决定一个Epoch有多少个Iteration 

如样本总数:80，Batchsize:8，则 1 Epoch = 10 Iteration

如样本总数:87， Batchsize:8

1 Epoch = 10 Iteration if drop_last = True 

1 Epoch = 11 Iteration if drop_last = False

## 2. DataSet

##### torch.utils.data.Dataset

功能:Dataset抽象类，所有自定义的 Dataset 需要继承它，并且复写 `__getitem__()`

- getitem : 接收一个索引，返回一个样本

## Transforms

## 数据标准化

## 数据预处理

## 数据增强

# 三、PyTorch 模型搭建

## 模型搭建要素及 Sequential

## 常用网络层介绍及使用

## nn.Module

## 模型容器 

## AlexNet 构建

## nn 网络层

卷积层

池化

线性

激活函数

## 权值初始化（10 种）

# 四、PyTorch 损失优化

## 特殊的 Module：Fuction

## 损失函数（17 种）

## 优化器（10 种）

## 学习率调整（6 种）

# 五、PyTorch 训练过程

## TensorBoard

## Loss 及 Accuracy 可视化

## 卷积核及特征图可视化

## 梯度及权值分布可视化

## 混淆矩阵及其可视化

## 类激活图可视化（Grad-CAM）

## hook 函数 & CAM 可视化

# 六、PyTorch 正则化

## 过拟合正则化

## L1 和 L2 正则项

## Dropout

## Batch Normalization

## module.eval() 对 dropout 及 BN 的影响

# 七、PyTorch 训练技巧

## 模型 Fine-tune

## 模型保存与加载

## Early Stop

## GPU 使用

# 八、PyTorch 常见报错

# 九、PyTorch 实例

## 图像分类：ResNet

## 图像分割：Unet

## 目标检测：Faster RCNN

## 生成对抗网络：GAN

## 循环神经网络：RNN