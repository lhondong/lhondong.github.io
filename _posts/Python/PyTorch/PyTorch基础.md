
# PyTorch 基础

```python
train_data_loader = Dataloader()
model = Model() # nn.Module
criterion = torch.nn.MSEloss()
optimizer = torch.optim.Adam(model.parameter(), lr=1e-3)

for epoch in range(num_epoches):
        model.train()
        for batch in train_data_loader:
                x, y = batch
                y_pred = model(x)
                loss = criterion(y, y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        eval_data_loader = Dataloader()
        with torch.no_grad():
                Y = []
                Y_pred = []
                for batch in eval_data_loader:
                        x, y = batch
                        Y.append(y)
                        y_pred = model(x)
                        Y_pred.append(y_pred)
                score = evaluate(Y, Y_pred)
                # score 最大就停止或者 early stopping
```

## 1. 数据模块

- 如何把数据从硬盘读到内存？
- 如何组织数据进行训练？
- 图片如何预处理及数据增强？

## 2. 模型

- 如何构建模型模块？
- 如何组织复杂网络？
- 如何初始化网络参数？
- 如何定义网络层？

## 3. 损失函数

- 如何创建损失函数？
- 如何设置损失函数超参数？
- 如何选择损失函数？

## 4. 优化器

- 如何管理模型参数？
- 如何管理多个参数组实现不同学习率？
- 如何调整学习率？

## 5. 迭代训练

- 如何观察训练效果？
- 如何绘制 Loss/Accuracy 曲线？
- 如何使用 TensorBoard 分析？

# PyTorch 基础

## 1. 张量

张量是一个**多维数组**，它是标量、向量、矩阵的高维拓展。

- Tensor 与 Variable

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

## Tensor 创建

### 1. 直接创建

#### torch.tensor()

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

`torch.from_numpy(ndarray)`

功能：从 numpy 创建 tensor

注意事项：从 torch.from_numpy 创建的 tensor 于原 ndarray 共享内存，当修改其中一个的数据，另外一个也将会被改动。

### 2. 依据数值创建

#### torch.zeros()

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

```python
out_t = torch.tensor([1])
t = torch.zeros((3, 3), out=out_t)
# 将生成的 out 赋值给 out_t
```

#### torch.zeros_like(input)

功能：依 input 形状创建全 0 张量

- torch.ones()
- torch.ones_like()
- torch.full()
- torch.full_like()

```python
torch.full(size,
           fill_value,
           ...)
torch.full((3,3), 10)
```

#### torch.arange()

功能：创建等差的 1 维张量

注意事项：数值区间为 [start, end)

- start：数列起始值
- end：数列“结束值”
- step：数列公差，默认为 1

```python
torch.arrange(2, 10, 2)
Out:
tensor([2,4,6,8]) #没有 10
```

#### torch.linspace() 

功能：创建均分的 1 维张量 

注意事项：数值区间为 [start, end]

- start：数列起始值
- end：数列结束值 
- steps：数列长度

```python
torch.linspace(2, 10, 5)
Out: tensor([2., 4., 6., 8., 10.])

torch.linspace(2, 10, 6)
Out: tensor([2.0000, 3.6000 ,5.2000, 6.8000, 8.4000, 10.0000])
```

#### torch.logspace()

功能：创建对数均分的 1 维张量 

注意事项：长度为 steps, 底为 base 

- start：数列起始值
- end：数列结束值
- steps：数列长度
- base：对数函数的底，默认为 10

#### torch.eye() 

功能：创建单位对角矩阵 (2 维张量） 

注意事项：默认为方阵

- n：矩阵行数
- m：矩阵列数

### 3. 依概率分布创建张量

#### torch.normal() 

功能：生成正态分布（高斯分布）

- mean：均值
- std：标准差

四种模式：

- mean 为标量，std 为标量 （需要设置 size）
- mean 为标量，std 为张量 
- mean 为张量，std 为标量 
- mean 为张量，std 为张量

```python
# mean, std 均为张量
mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1, 5, dtype=torch.float)
t_normal = torch.normal(mean, std)
Out:
mean: tensor([1., 2., 3., 4.])
std: tensor([1., 2., 3., 4.])
tensor([1.6614, 2.5338, 3.1850, 6.4853])

# mean：标量 std: 标量
t_normal = torch.normal(0., 1., size=(4,))
Out:
tensor([0.6614, 0.2669, 0.0617, 0.6213])

# mean：张量 std: 标量
mean = torch.arange(1, 5, dtype=torch.float)
std = 1
t_normal = torch.normal(mean, std)
Out:
tensor([1.6614, 2.2669, 3.0617, 4.6213])
```

#### torch.randn()

功能：生成**标准正态分布** 

- size：张量的形状

- torch.randn_like()

#### torch.rand()

- 功能：在区间 [0, 1) 上，生成**均匀分布**
- torch.rand_like()

#### torch.randint()

功能：区间 [low, high) 生成整数均匀分布 

- low
- high
- size：张量的形状

#### torch.randint_like()

#### torch.randperm()

- 功能：生成生成从 0 到 n-1 的随机排列
- n：张量的长度

#### torch.bernoulli()

- 功能：以 input 为概率，生成伯努力分布 (0-1 分布，两点分布）
- input：概率值

## Tensor 操作

### 1. 张量拼接与切分

####  torch.cat()

功能：将张量按维度 dim 进行拼接 

- tensors: 张量序列
- dim: 要拼接的维度

```python
t = torch.ones((2, 3))
t_0 = torch.cat([t, t], dim=0)
t_1 = torch.cat([t, t, t], dim=1)

Out:
t_0:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]]) shape:torch.Size([4, 3])
t_1:
tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1.]]) shape:torch.Size([2, 9])
```

#### torch.stack()

功能：在新创建的维度 dim 上进行拼接 

- tensors: 张量序列
- dim: 要拼接的维度

```python
a = torch.ones((2,3))
a_stack = torch.stack([a,a], dim = 2)

Out:
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]]) #新建一个维度，(2,3,2)

a_stack = torch.stack([a,a],dim = 0)

Out:
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]]) #在第 0 维新建，(2,2,3)
```

#### torch.chunk()

- 功能：将张量按维度 dim 进行平均切分
- 返回值：张量列表

注意事项：若不能整除，最后一份张量小于其他张量

- input: 要切分的张量
- chunks: 要切分的份数
- dim: 要切分的维度

```python
a = torch.ones((2,5))
list_of_tensor = torch.chunk(a, dim=1, chunks=2)

for idx,t in enumerate(list_of_tensor):
    print("tensor {}: {}".format(idx, t))

Out:
tensor 0: 
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor 1: 
tensor([[1., 1.],
        [1., 1.]]) #切分为两个张量，最后一个比较小
```

#### torch.split()

- 功能：将张量按维度 dim 进行切分
- 返回值：张量列表

- tensor: 要切分的张量
- split_size_or_sections: 为 int 时，表示每一份的长度；为 list 时，按 list 元素切分
- dim : 要切分的维度

```python
t = torch.ones((2, 5))
list_of_tensors = torch.split(t, [2, 1, 2], dim=1)

Out:
tensor([[1., 1.],
        [1., 1.]]), shape is torch.Size([2, 2])
tensor([[1.],
        [1.]]), shape is torch.Size([2, 1])
tensor([[1., 1.],
        [1., 1.]]), shape is torch.Size([2, 2])
```

### 2. 张量索引

#### torch.index_select()

- 功能：在维度 dim 上，按 index 索引数据
- 返回值：依 index 索引数据拼接的张量

- input: 要索引的张量
- dim: 要索引的维度
- index: 要索引数据的序号

```python
t = torch.randint(0, 9, size=(3, 3))
idx = torch.tensor([0, 2], dtype=torch.long) 
t_select = torch.index_select(t, dim=0, index=idx)

Out:
t:
tensor([[4, 5, 0],
        [5, 7, 1],
        [2, 5, 8]])
t_select:
tensor([[4, 5, 0],
        [2, 5, 8]]) #从第 0 个维度，选出第 0 行和第 2 行
```

注意：**index 必须是 dtype=torch.long，否则 torch.float 也会报错。**

#### torch.masked_select()

- 功能：按 mask 中的 True 进行索引
- 返回值：一维张量

- input: 要索引的张量
- mask: 与 input 同形状的布尔类型张量

```python
a = torch.randint(0, 9, (3,3))
mask = a.ge(5) # a >= 5, ge means greater or equal/ gt means greater than/le,lt
a_select = torch.masked_select(a, mask)

Out:
tensor([[2, 1, 5],
        [0, 6, 3],
        [8, 8, 2]]) 
tensor([5, 6, 8, 8])
```

### 3. 张量变换

#### torch.reshape()

功能：变换张量形状

注意事项：当张量在内存中是连续时，新张量与 input 共享数据内存

- input: 要变换的张量
- shape: 新张量的形状（-1 表示该维度不定义，根据其他维度计算而来）

#### torch.transpose()

功能：交换张量的两个维度 

- input: 要变换的张量
- dim0: 要交换的维度
- dim1: 要交换的维度

```python
t_transpose = torch.transpose(t, dim0=1, dim1=2)    # c*h*w -> h*w*c
```

#### torch.t() 

功能：2 维张量转置，对矩阵而言，等价于 torch.transpose(input, 0, 1)

#### torch.squeeze()

功能：压缩长度为 1 的维度（轴）

- dim: 若为 None，移除所有长度为 1 的轴；若指定维度，当且仅当该轴长度为 1 时，可以被移除；

```python
t = torch.rand((1, 2, 3, 1))
t_sq = torch.squeeze(t)
t_0 = torch.squeeze(t, dim=0)
t_1 = torch.squeeze(t, dim=1)

Out:
t.shape: torch.Size([1, 2, 3, 1])
t_sq.shape: torch.Size([2, 3])
t_0.shape: torch.Size([2, 3, 1])
t_1.shape: torch.Size([1, 2, 3, 1]) # 当且仅当该轴长度为 1 时，可以被移除
```

#### torch.unsqueeze() 

功能：依据 dim 扩展维度

- dim: 扩展的维度

### 4. 张量数学运算

#### torch.add()

功能：逐元素计算 input+alpha×other

- input: 第一个张量
- alpha: 乘项因子
- other: 第二个张量

```python
torch.add(input,
          alpha=1,
	        other, 
          out=None)

t_add = torch.add(t_0, 10 ,t_1)
```

#### torch.addcmul() 

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

#### torch.addcdiv()

$$
out = input+value \times \frac{tensor1}{tensor2}
$$

torch.sub() 

torch.div()

#### torch.dot()

向量点乘，得到的结果是 scale 标量。对应元素相乘并相加

`torch.dot(a,b)` 相当于 `torch.sum(torch.mul(a,b))`

#### torch.mul()

用法与 * 乘法相同，也是 element-wise 的乘法，也是支持 broadcast 的。

`torch.mul(mat1, other, out=None)`，其中 other 乘数可以是标量也可以是任意维度的矩阵，只要满足最终相乘是可以 broadcast 的即可，即该操作是支持 broadcast 操作的。

- other 是标量：例如 mat1 是维度任意的矩阵，那么输出一个矩阵，其中每个值是 mat1 中原值乘以 other，维度保持不变。
- other 是矩阵：只要 other 与 mat1 的维度可以满足 broadcast 条件，就可以进行逐元素乘法操作，例如：

```python
import torch
a = torch.randn(2, 3, 4)
b = torch.randn(3, 4)
print (torch.mul(a,b).shape) # 输出 torch.size(2,3,4)
```

#### torch.mm

- 数学里的矩阵乘法，要求两个 Tensor 的维度满足矩阵乘法的要求。

`torch.mm(mat1, mat2, out=None)`

其中 mat1(n×m), mat2 (m×d), Out (n×d)。一般只用来计算两个二维矩阵的矩阵乘法，而且不支持 broadcast 操作。

#### torch.bmm 三维带 Batch 矩阵乘法 

`torch.bmm(bmat1, bmat2, out=None)`

其中 bmat1(B×n×m), bmat2 (B×m×d), Out (B×n×d)。两个输入必须是三维矩阵且第一维相同（表示 Batch 维度），不支持 broadcast 操作。

#### torch.matmul "混合"矩阵乘法

torch.mm 的 broadcast 版本，具体操作取决于两个 tensor 的 shape，按两个矩阵维度的不同可分为以下五种

1. 如果两个矩阵都是 1 维，则执行向量点乘 dot 操作。
2. 如果两个矩阵都是 2 维，则执行矩阵相乘 mm 操作。
3. 第一个矩阵是 1 维，第二个是 2 维，行向量乘以矩阵。（线性代数矩阵和向量相乘），向量 × 矩阵相当于矩阵行向量的线性组合
4. 第一个矩阵是 2 维，第二个是 1 维，矩阵乘以列向量。矩阵 × 向量，相当于矩阵列向量的线性组合
5. 如果两个都至少是 1 维，并且至少一个维度大于 2。会执行 batch 矩阵相乘 torch.bmm。后两维进行矩阵相乘 mm。

相当于将每个矩阵看做一个元素，然后逐元素进行矩阵乘法。例如 a.shape=[j,k,m,n] b.shape=[k,n,k]。将每个矩阵看做一个 element 时，a.shape=[j,k],b.shape=[k]

然后按 element-wise 的方式进行 mm 矩阵乘法。element-wise 方式要求维度个数和大小必须相同。不相同执行广播机制。

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

线性回归是分析一个变量与另外一（多）个变量之间关系的方法

求解步骤：

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
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(100):

    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

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

计算图是用来描述运算的有向无环图。计算图有两个主要元素：结点 (Node) 和边 (Edge) 

- 结点表示数据，如向量，矩阵，张量
- 边表示运算，如加减乘除、卷积等

叶子结点：用户创建的结点称为叶子结点

设置叶子结点主要是为了节省内存，在计算梯度时，只有叶子结点会计算梯度，而非叶子结点内存被释放掉。is_leaf: 指示张量是否为叶子结点

如非叶子结点需要计算梯度，使用 retain_grad() 来保留梯度不被释放。`a.retain_grad()`

grad_fn: 记录创建该张量时所用的方法 （函数）

- `y.grad_fn = <MulBackward0>`
- `a.grad_fn = <AddBackward0>`
- `b.grad_fn = <AddBackward0>`

#### 动态图

动态图：运算与搭建同时进行，灵活易调节
静态图：先搭建图，后运算，高效不灵活

## 4. autograd 自动求导

#### torch.autograd.backward 

功能：自动求取梯度

- tensors: 用于求导的张量，如 loss
- retain_graph: 保存计算图
- create_graph: 创建导数计算图，用于高阶求导
- grad_tensors: 多梯度权重

```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)     # retain_grad()
b = torch.add(w, 1)

y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

loss = torch.cat([y0,y1], dim=0)
gradient_tensor = torch.tensor([1., 2.])

loss.backward(gradient=gradient_tensor) # 权重设置为 y0+2*y1
print(w.grad)
```

张量的 `backward()` 直接调用 `torch.autograd.backward()`

使用 retain_graph=True 可以多次反向传播，不会被内存释放。

#### torch.autograd.grad 

功能：高阶求导

- outputs: 用于求导的张量，如 loss
- inputs : 需要梯度的张量 
- create_graph : 创建导数计算图，用于高阶求导
- retain_graph : 保存计算图
- grad_outputs: 多梯度权重

```python
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)

grad_1 = torch.autograd.grad(y, x, create_graph=True) # 创建导数的计算图，对导数再次求导
print(grad_1)

grad_2 = torch.autograd.grad(grad_1[0], x)
print(grad_2)

Out:
(tensor([6.], grad_fn=<MulBackward0>),)
(tensor([2.]),)
```

#### autograd 特性

1. 梯度不自动清零
2. 依赖于叶子结点的结点，requires_grad 默认为 True 
3. 叶子结点不可执行 in-place

梯度清零 

`loss.grad.zero_()`，其中的下划线_表示 in-place 操作，原地操作 (`a += 1` 是原位操作，而 `a = a + 1` 不是原位操作）

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
x0 = torch.normal(mean_value * n_data, 1) + bias   # 类别0 数据 shape=(100, 2)
y0 = torch.zeros(sample_nums)                      # 类别0 标签 shape=(100)
x1 = torch.normal(-mean_value * n_data, 1) + bias  # 类别1 数据 shape=(100, 2)
y1 = torch.ones(sample_nums)                       # 类别1 标签 shape=(100)
train_x = torch.cat((x_0, x_1), 0)
train_y = torch.cat((y_0, y_1), 0)

# 选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1) # 相当于通过线性变换 y=x*T(A)+b 可以得到对应的各个系数
        self.sigmoid = nn.Sigmoid() # 相当于通过激活函数的变换

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
        mask = y_pred.ge(0.5).float().squeeze() # 将预测值大于 0.5 的转换为 1，小于 0.5 的转换为 0
        # print(mask)
        correct = (mask == train_y).sum() 
        # print(correct)
        accuracy = correct.item() / train_y.size(0) # 计算正确率，correct.item() 将 correct Tensor 转为整数

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
