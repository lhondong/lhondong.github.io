# Matplotlib

## 环境配置

### 设置样式

- plt.style.use
- plt.style.context

```python
import matplotlib.pyplot as plt
plt.style.available[:5]

Out:
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight']

with plt.style.context("seaborn-white"):
    plt.plot(x, y) # 暂时设置

# plt.style.use("seaborn-whitegrid") # 永久设置为某一风格
```
### 要不要 show

- jupyter 中可用魔术方法   %matplotlib inline
- IDE: plt.show()

### 保存图片

- plt.savefig('fig_name.png")

## 图片种类

### 折线图 plot

```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
```

#### 绘制多条曲线

```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
```

### 图修饰

#### 线条颜色、风格、宽度

- 线条颜色
```python
offsets = np.linspace(0, np.pi, 5)
colors = ["blue", "g", "r", "yellow", "pink"]
for offset, color in zip(offets, colors):
    plt.plot(x, np.sin(x-offset), color = color) # color可缩写为c
```

- 线条风格
```python
x = np.linspace(0, 10, 11)
offsets = list(range(8))
linestyles = ["solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"]
for offset, linestyle in zip(offsets, linestyles):
    plt.plot(x, x+offset, linestyle=linestyle)        # linestyle可简写为ls
```

- 线条宽度
```python
x = np.linspace(0, 10, 11)
offsets = list(range(0, 12, 3))
linewidths = (i*2 for i in range(1,5))
for offset, linewidth in zip(offsets, linewidths):
    plt.plot(x, x+offset, linewidth=linewidth) # linewidth可简写为lw
```

#### 数据点标记形状、大小和颜色配置

- 调整数据点标记
```python
x = np.linspace(0, 10, 11)
offsets = list(range(0, 12, 3))
markers = ["*", "+", "o", "s"]
for offset, marker in zip(offsets, markers):
    plt.plot(x, x+offset, marker=marker)   
```

- 调整数据点大小
```python
x = np.linspace(0, 10, 11)
offsets = list(range(0, 12, 3))
markers = ["*", "+", "o", "s"]
for offset, marker in zip(offsets, markers):
    plt.plot(x, x+offset, marker=marker, markersize=10) # markersize可简写为ms
```

- 颜色、风格设置的简写
```python
x = np.linspace(0, 10, 11)
offsets = list(range(0, 8, 2))
color_linestyles = ["g-", "b--", "k-.", "r:"] # 绿色实线、蓝色虚线、黑色点划线、红色点线
for offset, color_linestyle in zip(offsets, color_linestyles):
    plt.plot(x, x+offset, color_linestyle)

x = np.linspace(0, 10, 11)
offsets = list(range(0, 8, 2))
color_marker_linestyles = ["g*-", "b+--", "ko-.", "rs:"]
for offset, color_marker_linestyle in zip(offsets, color_marker_linestyles):
    plt.plot(x, x+offset, color_marker_linestyle)
```

#### 调整坐标轴范围、样式

- xlim, ylim
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.xlim(-1, 7)
plt.ylim(-1.5, 1.5)
```

- axis
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.axis([-2, 8, -2, 2])

plt.axis("tight") # 紧凑
plt.axis("equal") # 扁平
```

- 对数坐标
```python
x = np.logspace(0, 5, 100)
plt.plot(x, np.log(x))
plt.xscale("log") # 一条直线
```

- 调整坐标轴刻度
```python
x = np.linspace(0, 10, 100)
plt.plot(x, x**2)
plt.xticks(np.arange(0, 12, step=1), fontsize=15)
plt.yticks(np.arange(0, 110, step=10))
```

- 调整刻度样式
```python
x = np.linspace(0, 10, 100)
plt.plot(x, x**2)
plt.tick_params(axis="both", labelsize=15)
```

#### 设置图形标签、图例、添加文字

- 设置图形标签
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.title("A Sine Curve", fontsize=20)
plt.xlabel("x", fontsize=15)
plt.ylabel("sin(x)", fontsize=15)
```

- 设置图例设置图形标签
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), "b-", label="Sin")
plt.plot(x, np.cos(x), "r--", label="Cos")

plt.ylim(-1.5, 2)
plt.legend(loc="upper center", frameon=True, fontsize=15) # 设置到上层中间位置，frameon 外框
```

- 添加文字和箭头
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), "b-")
plt.text(3.5, 0.5, "y=sin(x)", fontsize=15) # 3.5, 0.5 是文字坐标

plt.annotate('local min', xy=(1.5*np.pi, -1), xytext=(4.5, 0),
             arrowprops=dict(facecolor='black', shrink=0.1),
             ) # xytext 箭头文字的位置
```

### 散点图 scatter

```python
x = np.linspace(0, 2*np.pi, 20)
plt.scatter(x, np.sin(x), marker="o", s=30, c="r")    # s 大小  c 颜色
```

- 颜色配置
```python
x = np.linspace(0, 10, 100)
y = x**2
plt.scatter(x, y, c=y, cmap="inferno")  
plt.colorbar() 
```

[颜色配置参考官方文档](https://matplotlib.org/examples/color/colormaps_reference.html)

- 根据数据控制点的大小
```python
x, y, colors, size = (np.random.rand(100) for i in range(4))
plt.scatter(x, y, c=colors, s=1000*size, cmap="viridis")
```

- 根据数据控制点的大小
```python
x, y, colors, size = (np.random.rand(100) for i in range(4))
plt.scatter(x, y, c=colors, s=1000*size, cmap="viridis")
```

- 透明度
```python
x, y, colors, size = (np.random.rand(100) for i in range(4))
plt.scatter(x, y, c=colors, s=1000*size, cmap="viridis", alpha=0.3)
plt.colorbar()
```

- 随机漫步
```python
from random import choice

class RandomWalk():
    """一个生产随机漫步的类"""
    def __init__(self, num_points=5000):
        self.num_points = num_points
        self.x_values = [0]
        self.y_values = [0]
    
    def fill_walk(self):
        while len(self.x_values) < self.num_points:
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance
            
            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance            
        
            if x_step == 0 or y_step == 0:
                continue
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            self.x_values.append(next_x)
            self.y_values.append(next_y)

rw = RandomWalk(10000)
rw.fill_walk()
point_numbers = list(range(rw.num_points))
plt.figure(figsize=(12, 6))  # 设置画布大小            
plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap="inferno", s=1)
plt.colorbar()
plt.scatter(0, 0, c="green", s=100)
plt.scatter(rw.x_values[-1], rw.y_values[-1], c="red", s=100)

plt.xticks([])
plt.yticks([]) # 隐藏 x,y 坐标轴
```

### 柱形图 bar

```python
x = np.arange(1, 6)
plt.bar(x, 2*x, align="center", width=0.5, alpha=0.5, color='yellow', edgecolor='red')
plt.tick_params(axis="both", labelsize=13)
```

- 替换坐标轴文字
```python
x = np.arange(1, 6)
plt.bar(x, 2*x, align="center", width=0.5, alpha=0.5, color='yellow', edgecolor='red')
plt.xticks(x, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.tick_params(axis="both", labelsize=13) 

x = ('G1', 'G2', 'G3', 'G4', 'G5')
y = 2 * np.arange(1, 6)
plt.bar(x, y, align="center", width=0.5, alpha=0.5, color='yellow', edgecolor='red')
plt.tick_params(axis="both", labelsize=13) 
```

- 设置不同的颜色
```python
x = ["G"+str(i) for i in range(5)]
y = 1/(1+np.exp(-np.arange(5)))

colors = ['red', 'yellow', 'blue', 'green', 'gray']
plt.bar(x, y, align="center", width=0.5, alpha=0.5, color=colors)
plt.tick_params(axis="both", labelsize=13)
```

- 累加柱形图
```python
x = np.arange(5)
y1 = np.random.randint(20, 30, size=5)
y2 = np.random.randint(20, 30, size=5)
plt.bar(x, y1, width=0.5, label="man")
plt.bar(x, y2, width=0.5, bottom=y1, label="women")
plt.legend()
```

- 并列柱形图
```python
x = np.arange(15)
y1 = x+1
y2 = y1+np.random.random(15)
plt.bar(x, y1, width=0.3, label="man")
plt.bar(x+0.3, y2, width=0.3, label="women")
plt.legend()
```

- 横向柱形图 barh

```python
x = ['G1', 'G2', 'G3', 'G4', 'G5']
y = 2 * np.arange(1, 6)
plt.barh(x, y, align="center", height=0.5, alpha=0.8, color="blue", edgecolor="red")
plt.tick_params(axis="both", labelsize=13)
```

宽度用 height 设置

### 多子图 subplot

```python
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.subplot(211)
plt.plot(t1, f(t1), "bo-", markerfacecolor="r", markersize=5)
plt.title("A tale of 2 subplots")
plt.ylabel("Damped oscillation")

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), "r--")
plt.xlabel("time (s)")
plt.ylabel("Undamped")
```

`plt.subplots_adjust(hspace=0.5, wspace=0.3)` 子图之间间隔设置

- 不规则子图
```python
def f(x):
    return np.exp(-x) * np.cos(2*np.pi*x)

x = np.arange(0.0, 3.0, 0.01)
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3) # 2 行 3 列的网格

plt.subplot(grid[0, 0])
plt.plot(x, f(x))

plt.subplot(grid[0, 1:])
plt.plot(x, f(x), "r--", lw=2)

plt.subplot(grid[1, :])
plt.plot(x, f(x), "g-.", lw=3)
```

### 直方图 hist

- 普通频次直方图
```python
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.hist(x, bins=50, facecolor='g', alpha=0.75)
```

bins 为 50 个区间内的频次。

- 概率密度 density=True
```python
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.hist(x, 50, density=True, color="r")
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)

plt.hist(x, bins=50, density=True, color="r", histtype='step') # histtype='step' 边缘台阶
```

```python
from scipy.stats import norm
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

_, bins, __ = plt.hist(x, 50, density=True)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', lw=3)  
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)
```

- 累计概率分布 cumulative=True
```python
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.hist(x, 50, density=True, cumulative=True, color="r")
plt.xlabel('Smarts')
plt.ylabel('Cum_Probability')
plt.title('Histogram of IQ')
plt.text(60, 0.8, r'$\mu=100,\ \sigma=15$')
plt.xlim(50, 165)
plt.ylim(0, 1.1)
```

### 误差图 errorbar bar

yerr=dy 误差
```python
x = np.linspace(0, 10 ,50)
dy = 0.5
y = np.sin(x) + dy*np.random.randn(50)

plt.errorbar(x, y , yerr=dy, fmt="+b")
```

- 柱形图误差图
```python
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = ['G1', 'G2', 'G3', 'G4', 'G5'] 
width = 0.35       

p1 = plt.bar(ind, menMeans, width=width, label="Men", yerr=menStd)
p2 = plt.bar(ind, womenMeans, width=width, bottom=menMeans, label="Men", yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.yticks(np.arange(0, 81, 10))
plt.legend()
```

### 面向对象的绘图风格

```python
x = np.linspace(0, 5, 10)
y = x ** 2

fig = plt.figure(figsize=(8,4), dpi=80)        # 图像
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])      # 轴 left, bottom, width, height (range 0 to 1)

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')
```

- 画中画
```python
x = np.linspace(0, 5, 10)
y = x ** 2

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) 

ax1.plot(x, y, 'r')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

ax2.plot(y, x, 'g')
ax2.set_xlabel('y')
ax2.set_ylabel('x')
ax2.set_title('insert title')
```

- 多子图
```python
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 3.0, 0.01)

fig= plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax1 = plt.subplot(2, 2, 1)
ax1.plot(t1, f(t1))
ax1.set_title("Upper left")

ax2 = plt.subplot(2, 2, 2)
ax2.plot(t1, f(t1))
ax2.set_title("Upper right")

ax3 = plt.subplot(2, 1, 2)
ax3.plot(t1, f(t1))
ax3.set_title("Lower")
```

### 三维绘图简介

- 三维数据点与线
```python
from mpl_toolkits import mplot3d

ax = plt.axes(projection="3d")
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline ,zline)

zdata = 15*np.random.random(100)
xdata = np.sin(zdata)
ydata = np.cos(zdata)
ax.scatter3D(xdata, ydata ,zdata, c=zdata, cmap="spring")
```

- 三维数据曲面图
```python
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y) # 网格化，用于形成曲面
Z = f(X, Y)

ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
```

## Seaborn

- 基于 matplotlib，数据结构与 pandas 统一
- 更加艺术化，提供高级可视化借口

```python
import seaborn as sns

x = np.linspace(0, 10, 500)
y = np.cumsum(np.random.randn(500, 6), axis=0)
sns.set()
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.legend("ABCDEF", ncol=2, loc="upper left")
```

## Pandas

- 线性图
- 柱形图
- 直方图
- 散点图
- 多子图


