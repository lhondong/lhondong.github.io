
# Python 标准库

## time

处理时间的标准库

### 获取现在时间

- 本地时间 `time.localtime()`
- UTC 世界统一时间 `time.gmtime()`
- 北京时间比时间统一时间 UTC 早 8 个小时

```python
import time

t_local = time.localtime()
t_UTC = time.gmtime()
print("t_local", t_local)           # 本地时间
print("t_UTC", t_UTC)               # UTC 统一时间
```

```python
time.ctime() # 返回本地时间的字符串
```

### 时间戳和计时器

- 返回自纪元以来的秒数，记录 sleep `time.time()`
- 随意选取一个时间点，记录现在时间到该时间点的间隔秒数，记录 sleep `time.perf_counter()`
- 随意选取一个时间点，记录现在时间到该时间点的间隔秒数，不记录 sleep `time.process_time()`
- perf_counter() 精度较 time() 更高一些

```python
t_1_start = time.time()
t_2_start = time.perf_counter()
t_3_start = time.process_time()
print(t_1_start)
print(t_2_start)
print(t_3_start)

res = 0
for i in range(1000000):
    res += i
    
time.sleep(5)
t_1_end = time.time()
t_2_end = time.perf_counter()
t_3_end = time.process_time()

print("time 方法：{:.3f}秒".format(t_1_end-t_1_start))
print("perf_counter 方法：{:.3f}秒".format(t_2_end-t_2_start))
print("process_time 方法：{:.3f}秒".format(t_3_end-t_3_start))

Out:
1567068710.7269545
6009.0814064
2.25
time 方法：5.128 秒
perf_counter 方法：5.128 秒
process_time 方法：0.125 秒
```

### 格式化输出

- 自定义格式化输出 `time.strftime`

```python
lctime = time.localtime()
time.strftime("%Y-%m-%d %A %H:%M:%S", lctime)

>>> '2022-06-11 Saturday 21:38:23'
```

### 睡觉

- `time.sleep()`

### timeit

- timeit 模块是 Python 内置的用于统计小段代码执行时间的模块，它同时提供命令行调用接口。

`timeit.timeit(stmt='pass', setup='pass', timer=, number=1000000, globals=None)`

- 命令行 `命令格式： python -m timeit [-n N] [-r N] [-u U] [-s S] [-t] [-c] [-h] [语句 ...]`
  - n：执行次数
  - r：计时器重复次数
  - s：执行环境配置（通常该语句只被执行一次）
  - p：处理器时间
  - v：打印原始时间
  - h：帮助

- jupyter 魔术方法 `%timeit 执行函数` （多次运行取平均值）

## random

处理随机问题的标准库

**Python 通过 random 库提供各种伪随机数，基本可以用于除加密解密算法外的大多数工程应用**

### 随即种子 `seed(a=None)`

1. 相同种子会产生相同的随机数
2. 如果不设置随机种子，以系统当前时间为默认值

```python
from random import *

seed(10)
print(random())
seed(10)
print(random())

>>> 0.5714025946899135

print(random())
>>> 0.20609823213950174
```

### 随机整数

- 产生 [a, b] 之间的随机整数 `randint(a, b)`
- 产生 [0, a) 之间的随机整数 `randrange(a)` 取不到 a
- 产生 [a, b) 之间以 setp 为步长的随机整数 `randrange(a, b, step)`

### 随机浮点数

- 产生 [0.0, 1.0) 之间的随机浮点数 `random()`
- 产生 [a, b] 之间的随机浮点数 `uniform(a, b)`

### 序列用随机函数

- 从序列类型中随机返回一个元素 `choice(seq)`

```python
choice(['win', 'lose', 'draw'])
>>> 'draw'

choice("python")
>>> 'h'
```

- 对序列类型进行 k 次重复采样，可设置权重 `choices(seq,weights=None, k)`

```python
choices(['win', 'lose', 'draw'], [4,4,2], k=10)
>>> ['lose', 'draw', 'lose', 'win', 'draw', 'lose', 'draw', 'win', 'win', 'lose']
```

- 将序列类型中元素随机排列，返回打乱后的序列 `shuffle(seq)`

```python
numbers = ["one", "two", "three", "four"]
shuffle(numbers)
```

- 从 pop 类型中随机选取 k 个元素，以列表类型返回 `sample(pop, k)`
- sample 相当于不放回抽样。如果列表中的数据不重复，抽取数据不重复；choices 相当于放回抽样，数据可能重复。

### 概率分布

- 生产一个符合高斯分布的随机数 `gauss(mean, std)`

- 用 random 库实现简单的微信红包分配

```python
import random

def red_packet(total, num):
    for i in range(1, num):
        per = random.uniform(0.01, total/(num-i+1)*2)          # 保证每个人获得红包的期望是 total/num
        total = total - per
        print("第{}位红包金额： {:.2f}元".format(i, per))
    else:
        print("第{}位红包金额： {:.2f}元".format(num, total))
                   
red_packet(10, 5)
```

- 验证

```python
import random
import numpy as np

def red_packet(total, num):
    ls = []
    for i in range(1, num):
        per = round(random.uniform(0.01, total/(num-i+1)*2), 2)     # 保证每个人获得红包的期望是 total/num
        ls.append(per)
        total = total - per
    else:
        ls.append(total)
        
    return ls
                  
# 重复发十万次红包，统计每个位置的平均值（约等于期望）
res = []
for i in range(100000):
    ls = red_packet(10,5)
    res.append(ls)

res = np.array(res)
np.mean(res, axis=0)

>>> array([2.0083243, 2.0048764, 2.0023768, 1.9958594, 1.9885631])
```

- 生产 4 位由数字和英文字母构成的验证码

```python
import random
import string

print(string.digits)
print(string.ascii_letters)

s = string.digits + string.ascii_letters
v = random.sample(s,4)
print(v)
print(''.join(v))

out:
0123456789
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
['n', 'Q', '4', '7']
nQ47
```

## collections

容器数据类型 

`import collections`

### 具名元组 namedtuple

`collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)`

- 点的坐标，仅看数据，很难知道表达的是一个点的坐标 `p = (1, 2)`
- 构建一个新的元组子类。定义方法如下：typename 是元组名字，field_names 是域名

```python
Point = collections.namedtuple("Point", ["x", "y"])
p = Point(1, y=2)
print(p)

>>> Point(x=1, y=2)
```

namedtuple 是一个函数，它用来创建一个自定义的 tuple 对象，并且规定了 tuple 元素的个数，并可以用属性而不是索引来引用 tuple 的某个元素。

这样用 namedtuple 可以很方便地定义一种数据类型，它具备 tuple 的不变性，又可以根据属性来引用，使用十分方便。

可以验证创建的 Point 对象是 tuple 的一种子类。

- 可以调用属性
- 有元组的性质

```python
print(p.x)
print(p.y) # 可以调用属性

print(p[0])
print(p[1])
x, y = p # 解包赋值
```

#### 模拟扑克牌

```python
Card = collections.namedtuple("Card", ["rank", "suit"])
ranks = [str(n) for n in range(2, 11)] + list("JQKA")    
suits = "spades diamonds clubs hearts".split()
cards = [Card(rank, suit) for rank in ranks
                          for suit in suits]

from random import *

# 洗牌
shuffle(cards)

# 随机抽一张牌
choice(cards)

# 随机抽多张牌
sample(cards, k=5)
```

### 计数器工具 Counter

```python
from collections import Counter
s = "牛奶奶找刘奶奶买牛奶"
colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
cnt_str = Counter(s)
cnt_color = Counter(colors)
print(cnt_str)
print(cnt_color)

>>> Counter({'奶': 5, '牛': 2, '找': 1, '刘': 1, '买': 1})
>>> Counter({'blue': 3, 'red': 2, 'green': 1})
```

- 是字典的一个子类
- 最常见的统计 `most_commom(n)` 提供 n 个频率最高的元素和计数

```python
cnt_color.most_common(2)

>>> [('blue', 3), ('red', 2)]
```

- 元素展开 `elements()`

```python
list(cnt_str.elements())

>>> ['牛', '牛', '奶', '奶', '奶', '奶', '奶', '找', '刘', '买']
```

- 其他一些加减操作

```python
c = Counter(a=3, b=1)
d = Counter(a=1, b=2)
print(c + d)

>>> Counter({'a': 4, 'b': 3})
```

- 从一副牌中抽取 10 张，大于 10 的比例有多少

```python
cards = collections.Counter(tens=16, low_cards=36)
seen = sample(list(cards.elements()), k=10)
seen.count('tens') / 10
```

### 双向队列 deque

- 列表访问数据非常快速
- 插入和删除操作非常慢，通过移动元素位置来实现
- 特别是 `insert(0, v)` 和 `pop(0)`，在列表开始进行的插入和删除操作

**双向队列可以方便的在队列两边高效、快速的增加和删除元素**

```python
from collections import deque

d = deque('cde') 

d.append("f")            # 右端增加
d.append("g")
d.appendleft("b")        # 左端增加
d.appendleft("a")

print(d)
>>> deque(['a', 'b', 'c', 'd', 'e', 'f', 'g'])

d.pop()           # 右端删除 
d.popleft()       # 左端删除
```

## itertools 迭代器

### 排列组合迭代器

- product 笛卡尔积

```python
import itertools

for i in itertools.product('ABC', '01'):
    print(i)

Out:
('A', '0')
('A', '1')
('B', '0')
('B', '1')
('C', '0')
('C', '1')
```

- permutations 排列

```python
for i in itertools.permutations('ABCD', 3):   # 3 是排列的长度
    print(i)

Out:
('A', 'B', 'C')
('A', 'B', 'D')
('A', 'C', 'B')
('A', 'C', 'D')
('A', 'D', 'B')
('A', 'D', 'C')
('B', 'A', 'C')
('B', 'A', 'D')
('B', 'C', 'A')
('B', 'C', 'D')
('B', 'D', 'A')
('B', 'D', 'C')
('C', 'A', 'B')
('C', 'A', 'D')
('C', 'B', 'A')
('C', 'B', 'D')
('C', 'D', 'A')
('C', 'D', 'B')
('D', 'A', 'B')
('D', 'A', 'C')
('D', 'B', 'A')
('D', 'B', 'C')
('D', 'C', 'A')
('D', 'C', 'B')

for i in itertools.permutations(range(3)):
    print(i)

Out:
(0, 1, 2)
(0, 2, 1)
(1, 0, 2)
(1, 2, 0)
(2, 0, 1)
(2, 1, 0)
```

- combinations 组合

```python
for i in itertools.combinations('ABCD', 2):  # 2 是组合的长度
    print(i)

Out:
('A', 'B')
('A', 'C')
('A', 'D')
('B', 'C')
('B', 'D')
('C', 'D')

for i in itertools.combinations(range(4), 3):
    print(i)

Out:
(0, 1, 2)
(0, 1, 3)
(0, 2, 3)
(1, 2, 3)
```

- combinations_with_replacement 元素可重复组合

```python
for i in itertools.combinations_with_replacement('ABC', 2):  # 2 是组合的长度
    print(i)
```

### 拉链

- zip 短拉链

```python
for i in zip("ABC", "012", "xyz"):
    print(i)

Out:
('A', '0', 'x')
('B', '1', 'y')
('C', '2', 'z')
```

- 长度不一时，执行到最短的对象处，就停止
- 注意 zip 是内置的，不需要加 itertools

- zip_longest 长拉链

```python
for i in itertools.zip_longest("ABC", "012345"):
    print(i)

Out:
('A', '0')
('B', '1')
('C', '2')
(None, '3')
(None, '4')
(None, '5')

# 指定使用 '?' 替代
for i in itertools.zip_longest("ABC", "012345", fillvalue = "?"): 
    print(i)

Out:
('A', '0')
('B', '1')
('C', '2')
('?', '3')
('?', '4')
('?', '5')
```

### 无穷迭代器

- count(start=0, step=1) 计数
- 创建一个迭代器，它从 start 值开始，返回均匀间隔的值

```python
itertools.count(10)
>>> 10, 11, 12, ...
```

- cycle(iterable) 循环
- 创建一个迭代器，返回 iterable 中所有元素，无限重复

```python
itertools.cycle('ABC')
>>> A, B, C, A, B, C, ...
```

- repeat(object  [, times]) 重复
- 创建一个迭代器，不断重复 object。除非设定参数 times ，否则将无限重复

```python
for i in itertools.repeat(10, 3):
    print(i)
>>> 10
```

### 锁链 chain(iterables)

- 把一组迭代对象串联起来，形成一个更大的迭代器

```python
for i in itertools.chain('ABC', [1, 2, 3]):
    print(i)

Out:
A
B
C
1
2
3
```

### 枚举 enumerate(iterable, start=0)

- 产出由两个元素组成的元组，结构是（index, item）, 其中 index 从 start 开始，item 从 iterable 中取

```python
for i in enumerate("Python", start=1):
    print(i)

Out:
(1, 'P')
(2, 'y')
(3, 't')
(4, 'h')
(5, 'o')
(6, 'n')
```

### 分组

`groupby(iterable, key=None)`

- 创建一个迭代器，按照 key 指定的方式，返回 iterable 中连续的键和组
- 一般来说，要预先对数据进行排序
-  key 为 None 默认把连续重复元素分组

```python
for key, group in itertools.groupby('AAAABBBCCDAABBB'):
    print(key, list(group))

Out:
A ['A', 'A', 'A', 'A']
B ['B', 'B', 'B']
C ['C', 'C']
D ['D']
A ['A', 'A']
B ['B', 'B', 'B']
```

```python
animals = ["duck", "eagle", "rat", "giraffe", "bear", "bat", "dolphin", "shark", "lion"]
animals.sort(key=len)

>>> ['rat', 'bat', 'duck', 'bear', 'lion', 'eagle', 'shark', 'giraffe', 'dolphin']

# 按长度分组
for key, group in itertools.groupby(animals, key=len):
    print(key, list(group))

Out:
3 ['rat', 'bat']
4 ['duck', 'bear', 'lion']
5 ['eagle', 'shark']
7 ['giraffe', 'dolphin']

# 按首字母进行分组
animals = ["duck", "eagle", "rat", "giraffe", "bear", "bat", "dolphin", "shark", "lion"]
animals.sort(key=lambda x: x[0])
print(animals)

>>> ['bear', 'bat', 'duck', 'dolphin', 'eagle', 'giraffe', 'lion', 'rat', 'shark']

for key, group in itertools.groupby(animals, key=lambda x: x[0]):
    print(key, list(group))

Out:
b ['bear', 'bat']
d ['duck', 'dolphin']
e ['eagle']
g ['giraffe']
l ['lion']
r ['rat']
s ['shark']
```
