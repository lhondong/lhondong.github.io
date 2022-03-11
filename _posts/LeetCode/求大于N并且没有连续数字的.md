---
title: "求大于 N 并且没有连续数字的数"
subtitle: "微软笔试笔记"
layout: post
author: "L Hondong"
header-img: "img/post-bg-12.jpg"
mathjax: ture
tags:
  - LeetCode
  - 算法
---

# 求大于 N 并且没有连续数字的数

## 题目

给定一个正整数 N，查找不包含两个相同连续数字的大于 N 的最小整数。

## 示例

示例 1:

```
N = 1765，大于 N 的最小整数为 1766。
然而，1766 的最后两位数字是相同的。
下一个整数 1767 不包含两个相同的连续数字，是大于 1765 且满足条件的最小整数。
请注意，1767 的第二位和第四位数字都可以是7，因为它们不是连续的。
```

示例 2:

```
N = 55, return 56
```

示例 3:

```
N = 98, return 101
99 和 100 都包含两个相同的连续数字。
```

示例 4:

```
N = 44432, return 45010
```

示例 5:

```
N = 3298, return 3401
```

## 题解

```python
def solution(N):
    res = str(N + 1)
    i = 1
    while i < len(res):
        if res[i] == res[i - 1]:
            res = str(int(res[:i + 1]) + 1) + '0' * len(res[i + 1:])
            i = 1
        else:
            i += 1
    return res
```

## 笔记

初始化 res = str(N+1)，用来记录每一位数字，后续比较相邻位的数字通过 res 进行。

从 res 的最高位开始比较是否有连续数字，如果有则将前 i 位 res[:i + 1] 转为数字类型加 1，将后面的 res[i + 1:] 补位 0，并将 i 置为 1，从头开始比较；如果没有则 i + 1，直到最后没有连续的位。

### 优化

遇到重复数字时，将前 i 位 res[:i + 1] 转为数字类型加 1，然后对后面的位用 0 和 1 交替填充，可避免全 0 发生重复。

```python
def solution(n):
    res = str(n + 1)
    flag = 1
    while flag < len(res):
        if res[flag] == res[flag - 1]:
            res = str(int(res[:flag + 1]) + 1) + ''.join([str(j % 2) for j in range(len(res) - flag - 1)])
            flag = 1
        else:
            flag += 1
    return res
```