# Numpy Mindmap

## 为什么要用 Numpy

### 低效的 Python for 循环

### Numpy 高效的原因

- C 语言实现
- 编译型语言
- 内存中连续单一类型存储
- 多线程

### 何时考虑 Numpy

- 用到 for 循环实现 向量化、矩阵化操作的时候

## 创建 Numpy 数组

### 从列表开始创建

- np.array([1, 2], dtype="float32")

### 从头开始创建

#### 固定元素

- np.zeros
- np.ones
- np.full
- np.eye

#### 序列

- np.arange
- np.linspace
- np.logspace

#### 随机

- np.random.random
- np.random.randint
- np.normal
- np.random.permutation
- np.random.shuffle
- np.random.choice

## 数组的性质

### 属性

- 形状 shape
- 维度 ndim
- 大小 size
- 数据类型 dtype

### 索引

- 一维与列表类似
- 二维 x[0, 0] x[0][0]

### 切片

- 一维与列表类似
-  二维 x[:2, :3]
- 获取行 x[1]
- 获取列 x[:, 2]
- 切片获得的是视图，安全复制用切片 copy()

### 变形

- 通用格式 x.reshape(m, n)
- 多维转一维 reshape(-1)  ravel  flatten(flatten是副本，另两个是视图)

### 拼接

- 水平拼接 hstack  c_
- 垂直拼接 vtack  r_

### 分裂

- 一维 split
- 水平 hsplit
- 垂直 vsplit

## Numpy 四大运算

### 向量化运算

- +-*/  % // exp log 直接作用于全部数据
- 两个数组的运算：对应位置的元素进行运算

### 矩阵化运算

- 矩阵转置 x.T
- 矩阵乘法 x.dot(y)  np.dot(x, y)

### 广播运算

- 两数组运算，维度不匹配
- 沿着维度为 1 的维度进行扩展

### 比较预算、掩码

- 比较运算的结果为布尔数组
- 布尔数组可作为掩码

### 花哨的索引

- 一维数组 x[ind]
- 二维数组 x[row, col]

## 通用函数

### 排序

- 获取值 np.sort
- 获取索引 np.argsort

### 最大最小值

- 获取值 np.max  np.min
- 获取索引 np.argmax  np.argmins

### 求和、求积

- np.sum 行 axis=1 列 axis=0
- np.prod

### 统计相关

- 中位数 np.median
- 均值 np.mean
- 方差 np.var
- 标准差 np.std