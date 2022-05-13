# Numpy

- 低效的 python for 循环
- numpy 支持向量化操作

1. 编译型语言 vs 解释型语言
   Numpy 使用了优化过的 C API，不同于解释型语言，C 语言执行时对代码进行整理编译，速度更快
2. 连续单一类型存储 vs 分散多变类型存储
   1. Numpy 数组内的数据类型必须是统一的，如全部是浮点型，而 Python 列表支持任意类型数据的填充
   2. Numpy 数组内的数据**连续存储在内存中**，而 Python 列表的数据**分散在内存中**
   3. **这种存储结构，与一些更加高效的底层处理方式更加的契合**
3. 多线程 vs 线程锁
   Python 语言执行时有线程锁，无法实现真正的多线程并行，而 C 语言可以

NumPy 的主要对象是 ndarray 对象，它其实是一系列同类型数据的集合。因为 ndarray 支持创建多维数组，所以就有两个行和列的概念。

- 创建 ndarray 的第一种方式是利用 array 方式
- 第二种办法则使用 Numpy 的内置函数
  - 使用 arange 或 linspace 创建连续数组
  - 使用 zeros()，ones()，full() 创建数组
  - 使用 eye() 创建单位矩阵
  - 使用 diag() 创建对角矩阵 `x = np.diag([1, 2, 3])`
  - 使用 random 创建随机数组

### 什么时候用 Numpy

**在数据处理的过程中，遇到使用 “Python for 循环” 实现一些向量化、矩阵化操作的时候，要优先考虑用 Numpy**，如两个向量的点乘、矩阵乘法。

## 创建数组

### 从列表创建

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x)
# Output: [1 2 3 4 5]

print(type(x))
print(x.shape)
# Output: <class 'numpy.ndarray'>
# (5,)
```

#### 设置数组的数据类型

```python
x = np.array([1, 2, 3, 4, 5], dtype="float32")
print(x)
print(type(x[0]))
# Output: [1. 2. 3. 4. 5.]
# <class 'numpy.float32'>
```

#### 二维数组

```python
x = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])
print(x)
print(x.shape)
# Output: [[1 2 3]
#          [4 5 6]
#          [7 8 9]]
# (3, 3)
```

### 从头创建

#### 1. 创建长度为 5 的数组，值都为 0

```python
np.zeros(5, dtype=int)
```

#### 2. 创建一个 2×4 的浮点型数组，值都为 1

```python
np.ones((2, 4), dtype=float)
# Output: array([[1., 1., 1., 1.],
#                [1., 1., 1., 1.]])
```
#### 3. 创建一个 3×5 的数组，值都为 8.8

```python
np.full((3, 5), 8.8)
```

#### 4. 创建一个 3×3 的单位矩阵

```python
np.eye(3)
Output:
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

#### 5. 创建一个线性序列数组，从 1 开始，到 15 结束，步长为 2

```python
np.arange(1, 15, 2)
```

#### 6. 创建一个 4 个元素的数组，这四个数均匀的分配到 0~1（等差数列）

```python
np.linspace(0, 1, 4)
Output:
array([0.        , 0.33333333, 0.66666667, 1.        ])
```

#### 7. 创建一个 10 个元素的数组，形成 1~10^9 的等比数列

```python
np.logspace(0, 9, 10)
Output:
array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,
       1.e+08, 1.e+09])
```

np.logspace(0, 9, 10) 中 0 表示 10 的 0 次方，9 表示 10 的 9 次方，10 表示 10 个数字

#### 8. 创建一个 3×3 的，在 0~1 之间均匀分布的随机数构成的数组

```python
np.random.random((3,3))
```

#### 9. 创建一个 3×3 的，均值为 0，标准差为 1 的正态分布随机数构成的数组

```python
np.random.normal(0, 1, (3,3))
```

#### 10. 创建一个 3×3 的，在 [0,10) 之间随机整数构成的数组

```python
np.random.randint(0, 10, (3,3))
```

0 可以取到，10 取不到。

#### 11. 随机重排列

```python
x = np.array([10, 20, 30, 40])
np.random.permutation(x)
np.random.shuffle(x)
```

其中 permutation 不会改变原数组，而是生成新的数组，而 shuffle 直接改变原数组。

#### 12. 随机采样

- 按指定形状采样

```python
x = np.arange(10, 25, dtype = float)
np.random.choice(x, size=(4, 3))
# Output:
# array([[19., 23., 22.],
#        [22., 21., 13.],
#        [15., 21., 17.],
#        [14., 23., 19.]])

np.random.choice(10, (2, 5))
# Output
# array([[1, 2, 4, 8, 2],
#        [6, 2, 0, 5, 9]])
```

```python
x = np.arange(5).reshape(1, 5)
Out: array([[0, 1, 2, 3, 4]])

x.sum(axis=1, keepdims=True)
Out: array([[10]])
```

- 按概率采样

```python
x = np.arange(10, 25, dtype = float)
np.random.choice(x, size=(4, 3), p=x/np.sum(x))
Out:array([[17., 14., 24.],
       [14., 19., 17.],
       [23., 12., 16.],
       [15., 24., 19.]])
```

## 数组的性质

### 属性

- 形状 x.shape
- 维度 x.ndim
- 大小 x.size (3 行 4 列，size = 12)
- 类型 x.dtype (dtype('float64'))

### 索引

多维数组索引 `x[0, 0]` 和 `x[0][0]` 相同，而 Python list 只有 `x[0][0]` 索引。

注意：numpy 数组的数据类型是固定的，向一个整型数组插入一个浮点值，浮点值会向下进行取整

### 切片

```python
x = np.random.randint(20, size=(3,4)) 
x[:2, :3] # 前两行，前三列
x[:2, 0:3:2] # 前两行 前三列（每隔一列）
x[::-1, ::-1] # 反向
```

#### 获取数组的行和列

```python
x = np.random.randint(20, size=(3,4)) 
x[1, :] #第一行，从 0 开始计数
x[1] # 第一行简写
x[:, 2] # 第二列，从 0 开始计数
```

#### 切片获取的是视图，而非副本

注意：视图元素发生修改，则原数组亦发生相应修改

修改切片的安全方式：copy

```python
x = np.random.randint(20, size=(3,4)) 
x1 = x[:2, :2].copy()
```

此时修改 x1 中的元素，x 不随之改变。

### 变形

x.reshape(3, 4)

注意：reshape 返回的是视图，而非副本

#### 一维向量转行向量

```python
x = np.random.randint(0, 10, (12,))
x1 = x.reshape(1, x.shape[0])
x1 = x.reshape(1, x.size)
Output:
array([[2, 0, 8, 1, 2, 9, 4, 7, 1, 8, 3, 0]])

x2 = x[np.newaxis, :] # 切片的方法，新指定一个维度
```

#### 一维向量转列向量

```python
x1 = x.reshape(x.shape[0], 1)
x2 = x[:, np.newaxis] 
```

#### 多维向量转一维向量

- `x.flatten()` flatten 返回的是副本
- `x.ravel()`  ravel 返回的是视图
- `x.reshape(-1)` reshape 返回的是视图

### 拼接

- 水平拼接
  - `x3 = np.hstack([x1, x2])`
  - `x4 = np.c_[x1, x2]`
- 垂直拼接
  - `x5 = np.vstack([x1, x2])`
  - `x6 = np.r_[x1, x2]`

注意：上述方法都不是视图而是副本，改变新数组，原数组都不会改变

### 分裂

```python
x = np.arange(10)
# Out: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x1, x2, x3 = np.split(x, [2, 7]) # 从第 2 个，第 7 个位置分
# Out: [0 1] [2 3 4 5 6] [7 8 9]
```

#### hsplit

```python
x = np.arange(1, 26).reshape(5, 5)

array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25]])

left, middle, right = np.hsplit(x, [2,4])
print("left:\n", left)            # 第 0~1 列
print("middle:\n", middle)        # 第 2~3 列
print("right:\n", right)          # 第 4 列

Out:
left:
 [[ 1  2]
 [ 6  7]
 [11 12]
 [16 17]
 [21 22]]
middle:
 [[ 3  4]
 [ 8  9]
 [13 14]
 [18 19]
 [23 24]]
right:
 [[ 5]
 [10]
 [15]
 [20]
 [25]]
```

#### vsplit

```python
upper, middle, lower = np.vsplit(x7, [2,4])
print("upper:\n", upper)         # 第 0~1 行
print("middle:\n", middle)       # 第 2~3 行
print("lower:\n", lower)         # 第 4 行

Out:
upper:
 [[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
middle:
 [[11 12 13 14 15]
 [16 17 18 19 20]]
lower:
 [[21 22 23 24 25]]
```

## 四大运算

### 向量化

#### 1. 与数字的加减乘除

```python
x = np.arange(1,6)
print("x1+5", x1+5)
print("x1-5", x1-5)
print("x1*5", x1*5)
print("x1/5", x1/5)

Out:
x1+5 [ 6  7  8  9 10]
x1-5 [-4 -3 -2 -1  0]
x1*5 [ 5 10 15 20 25]
x1/5 [0.2 0.4 0.6 0.8 1. ]

print("-x1", -x1)
print("x1**2", x1**2)
print("x1//2", x1//2)
print("x1%2", x1%2)

Out:
-x1 [-1 -2 -3 -4 -5]
x1**2 [ 1  4  9 16 25]
x1//2 [0 1 1 2 2]
x1%2 [1 0 1 0 1]
```

#### 2. 绝对值、三角函数、指数、对数

- 绝对值
```python
abs(x)
np.abs(x)
```

- 三角函数
```python
theta = np.linspace(0, np.pi, 3)

Out:
array([0.        , 1.57079633, 3.14159265])

print("sin(theta)", np.sin(theta))
print("con(theta)", np.cos(theta))
print("tan(theta)", np.tan(theta))

Out:
sin(theta) [0.0000000e+00 1.0000000e+00 1.2246468e-16]
con(theta) [ 1.000000e+00  6.123234e-17 -1.000000e+00]
tan(theta) [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16]

x = [1, 0 ,-1]
print("arcsin(x)", np.arcsin(x))
print("arccon(x)", np.arccos(x))
print("arctan(x)", np.arctan(x))

Out:
arcsin(x) [ 1.57079633  0.         -1.57079633]
arccon(x) [0.         1.57079633 3.14159265]
arctan(x) [ 0.78539816  0.         -0.78539816]
```

#### 3. 指数

```python
x = np.arange(3)
np.exp(x)

Out:
array([1.        , 2.71828183, 7.3890561 ])
```

#### 4. 对数

```python
x = np.array([1, 2, 4, 8 ,10])
print("ln(x)", np.log(x))
print("log2(x)", np.log2(x))
print("log10(x)", np.log10(x))

Out:
ln(x) [0.         0.69314718 1.38629436 2.07944154 2.30258509]
log2(x) [0.         1.         2.         3.         3.32192809]
log10(x) [0.         0.30103    0.60205999 0.90308999 1.        ]
```

### 矩阵化

```python
y = x.T # 矩阵转置

x.dot(y) # 矩阵乘法
np.dot(x, y) # 矩阵乘法

y.dot(x) 
np.dot(y, x)
```

注意跟 `x*y` 的区别： `x*y` 是对应元素相乘。

### 广播

```python
x = np.arange(3).reshape(1, 3)
x + 

x1 = np.ones((3,3))
x2 = np.arange(3).reshape(1, 3)
x1+x2

Out:
array([[1., 2., 3.],
       [1., 2., 3.],
       [1., 2., 3.]])

x3 = np.logspace(1, 10, 10, base=2).reshape(2, 5)
# array([[   2.,    4.,    8.,   16.,   32.],
#        [  64.,  128.,  256.,  512., 1024.]])
x4 = np.array([[1, 2, 4, 8, 16]])
x3/x4

Out:
array([[ 2.,  2.,  2.,  2.,  2.],
       [64., 64., 64., 64., 64.]])

x5 = np.arange(3).reshape(3, 1)
x6 = np.arange(3).reshape(1, 3)
x5 + x6

Out:
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])

扩展成 [[0, 0, 0], + [[0, 1, 2]
       [1, 1, 1],    [0, 1, 2]
       [2, 2, 2]]    [0, 1, 2]] 
```

如果两个数组的形状在维度上不匹配，那么数组的形式会沿着维度为1的维度进行扩展以匹配另一个数组的形状。

### 比较运算与掩码

```python
x1 = np.random.randint(100, size=(10,10))
x1 > 50

Out:
array([[False, False,  True,  True, False, False,  True,  True, False, True],
       [ True,  True,  True, False, False, False,  True, False, False, False],
       [ True,  True,  True,  True,  True, False, False,  True, False, True],
       [False,  True,  True, False, False, False,  True, False, False, False],
       [False, False, False, False, False, False,  True,  True,  True, True],
       [ True,  True,  True,  True, False, False,  True, False,  True, True],
       [ True, False, False,  True,  True,  True,  True, False, False, True],
       [ True, False, False, False, False,  True, False, False,  True, True],
       [ True, False,  True, False,  True, False,  True, False,  True, False],
       [False, False, False, False,  True,  True, False, False, False, True]])
```

#### 操作布尔数组

```python
x = np.random.randint(10, size=(3, 4))

Out:
array([[1, 4, 2, 9],
       [8, 8, 2, 4],
       [9, 5, 3, 6]])

np.sum(x > 5)
Out: 5 # 5 个数字大于 5

np.all(x > 0)
Out: True

np.any(x == 6)
Out: True # 任意一个元素等于 6

np.all(x < 9, axis=1)   # 按行进行判断
Out: array([False,  True, False])

(x < 9) & (x >5)
Out:
array([[False, False, False, False],
       [ True,  True, False, False],
       [False, False, False,  True]])

np.sum((x < 9) & (x >5))
Out: 3
```

#### 将布尔数组作为掩码

```python
x > 5
Out:
array([[False, False, False,  True],
       [ True,  True, False, False],
       [ True, False, False,  True]])

x[x > 5]
Out:
array([9, 8, 8, 9, 6])
```

大于 5 的被取出来。

### 花哨的索引

```python
x = np.random.randint(100, size=10)
Out:
array([43, 69, 67,  9, 11, 27, 55, 93, 23, 82])

ind = [2, 6, 9]
x[ind]
Out:
array([67, 55, 82])

ind = np.array([[1, 0],
               [2, 3]])
x[ind]
Out:
array([[69, 43],
       [67,  9]])
```

注意：结果的形状与索引数组ind一致

```python
x = np.arange(12).reshape(3, 4)
Out:
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

row = np.array([0, 1, 2])
col = np.array([1, 3, 0])
x[row, col]               # x(0, 1) x(1, 3) x(2, 0)
Out:
array([1, 7, 8])

row[:, np.newaxis]       # 列向量
Out:
array([[0],
       [1],
       [2]])

x[row[:, np.newaxis], col]    # 广播机制
Out:
array([[ 1,  3,  0], # 第一行的 (1, 3, 0) 三个元素
       [ 5,  7,  4], # 第二行的 (1, 3, 0) 三个元素
       [ 9, 11,  8]]) # 第三行的 (1, 3, 0) 三个元素
```

## 其他通用函数

### 排序

- `np.sort(x)` 产生新的排序数组
- `x.sort()` 替换原数组
- `i = np.argsort(x)` 获得排序索引

### 最大最小值

- `np.max(x)`, `np.max(x)` 最大值、最小值
- `np.argmax(x)`, `np.argmin(x)` 最大值、最小值的索引

### 求和求积

- `x.sum()`
- `np.sum(x)`
- `np.sum(x, axis=1)` 按行求和
- `np.sum(x, axis=0)` 按列求和
- `x.prod`, `np.prod(x)` 求积

### 统计相关

- `np.median(x)` 中位数
- `x.mean()`, `np.mean(x)` 均值
- `x.var()`, `np.var(x)` 方差
- `x.std()`, `np.std(x)` 标准差