---
title: "剑指 Offer 45. 把数组排成最小的数"
subtitle: "LeetCode 刷题笔记"
layout: post
author: "L Hondong"
header-img: "img/post-bg-12.jpg"
mathjax: ture
tags:
  - LeetCode
  - 算法
---

# 剑指 Offer 45. 把数组排成最小的数

## 题目

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

## 示例

示例 1:

```
输入：[10,2]
输出："102"
```

示例 2:

```
输入：[3,30,34,5,9]
输出："3033459"
```

## 题解

```python
class Solution(object):
    def minNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        def sort_rule(x, y):
            a, b = x + y, y + x
            if a > b: return 1
            if a < b: return -1
            if a == b: return 0
        strs = [str(num) for num in nums]
        strs.sort(key = functools.cmp_to_key(sort_rule))
        return ''.join(strs)
```

## 笔记

定义：

- 若拼接字符串 x+y > y+x，则 x “大于” y ；
- 反之，若 x+y < y+x ，则 x “小于” y ；

x “小于” y 代表：排序完成后，数组中 x 应在 y 左边；“大于” 则反之。

### sort 函数

Python 列表有一个内置的 list.sort() 方法可以直接修改列表。还有一个 sorted() 内置函数，它会从一个可迭代对象构建一个新的排序列表。

简单的升序排序非常简单：只需调用 sorted() 函数。它返回一个新的排序后列表：

```python
sorted([5, 2, 3, 1, 4])
输出：[1,2,3,4,5]
```

也可以使用 list.sort() 方法，它会直接修改原列表（并返回 None 以避免混淆），通常来说不如 sorted() 方便 ——— 但如果不需要原列表，它会更有效率。

```python
a = [5,2,3,4,1]
a.sort()
```

另外一个区别是， list.sort() 方法只是为列表定义的，而 sorted() 函数可以接受任何可迭代对象。

```python
list.sort(cmp=None, key=None, reverse=False)
```

- cmp -- 可选参数，如果指定了该参数会使用该参数的方法进行排序。
- key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
- reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。

Python3 中移除了 cmp 内建函数，sorted 函数也没有了 cmp 这个关键字参数，但可以通过 functools 模块中的 cmp_to_key 来对自定义的 cmp 函数进行包装，然后就能赋值给 sorted 函数的关键字参数 key，来间接实现 Python2 中 cmp 函数用于排序的效果。

cmp_to_key 是在 python3 中使用的，其实就是 python2 中的 cmp 函数。

python3 不支持比较函数，在一些接受 key 的函数中（例如 sorted，min，max，heapq.nlargest，itertools.groupby），key 仅仅支持一个参数，就无法实现两个参数之间的对比，采用 cmp_to_key 函数，可以接受两个参数，将两个参数做处理，比如做和做差，转换成一个参数，就可以应用于 key 关键字之后。

cmp_to_key 返回值小于 0，则交换值；如果返回值大于等于 0，则不执行任何操作。

注意点：

x = [("a", 1), ("b", 3), ("c", 2)]

在 cmp_to_key 中，第一个入参是 ("b", 3) ，第二个入参是 ("a", 1)。**本质上相当于 `return 1` 时交换，`return -1` 时不交换**。

```python
from functools import cmp_to_key

# 字典按 key 降序排序，再按 val 升序排序
a = {"a": 5, "c": 3, "e": 2, "d": 2}

def cmp(val1, val2):
    if val1[0] < val2[0]:
        return 1
    elif val1[0] > val2[0]:
        return -1
    else:
        if val1[1] > val2[1]:
            return 1
        else:
            return -1

sd = sorted(a.items(), key=cmp_to_key(cmp))
print(sd)  # [('e', 2), ('d', 2), ('c', 3), ('a', 5)]
```