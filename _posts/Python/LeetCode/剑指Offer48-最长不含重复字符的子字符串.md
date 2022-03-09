---
title: "剑指 Offer 48. 最长不含重复字符的子字符串"
subtitle: "LeetCode 刷题笔记"
layout: post
author: "L Hondong"
header-img: "img/post-bg-12.jpg"
mathjax: ture
tags:
  - LeetCode
  - 算法
---

# 剑指 Offer 48. 最长不含重复字符的子字符串

## 题目

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

## 示例

示例 1:

```
输入："abcabcbb"
输出：3 
解释：因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

示例 2:

```
输入："bbbbb"
输出：1
解释：因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

示例 3:

```
输入："pwwkew"
输出：3
解释：因为无重复字符的最长子串是 "wke"，所以其长度为 3。
请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

## 题解

```python

```

## 笔记

- 状态定义：设动态规划列表 dp，dp[j] 代表以字符 s[j] 为结尾的 “最长不重复子字符串” 的长度。
- 转移方程：固定右边界 j，设字符 s[j] 左边距离最近的相同字符为 s[i]，即 s[i]=s[j]。
  1. 当 i<0，即 s[j] 左边无相同字符，则 dp[j]=dp[j−1]+1；
  2. 当 dp[j−1] < j−i，说明字符 s[i] 在子字符串 dp[j−1] 区间之外，则 dp[j]=dp[j−1]+1；
  3. 当 dp[j−1]≥j−i，说明字符 s[i] 在子字符串 dp[j−1] 区间之中，则 dp[j] 的左边界由 s[i] 决定，即 dp[j]=j−i；

> 当 i<0 时，由于 dp[j−1]≤j 恒成立，因而 dp[j−1] < j−i 恒成立，因此分支 1. 和 2. 可被合并。

- 返回值：max(dp) ，即全局的 “最长不重复子字符串” 的长度。

<div align=center><img src="/images/剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-40-24.png" alt="剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-40-24" style="zoom:50%;" /></div>

### 空间复杂度优化

- 由于返回值是取 dp 列表最大值，因此可借助变量 tmp 存储 dp[j] ，变量 res 每轮更新最大值即可。
- 此优化可节省 dp 列表使用的 O(N) 大小的额外空间。

观察转移方程，可知问题为：每轮遍历字符 s[j] 时，如何计算索引 i？

以下介绍**哈希表，线性遍历**两种方法。

## 方法一：动态规划 + 哈希表

- 哈希表统计：遍历字符串 s 时，使用哈希表（记为 dic）统计各字符最后一次出现的索引位置。
- 左边界 i 获取方式：遍历到 s[j] 时，可通过访问哈希表 dic[s[j]] 获取最近的相同字符的索引 i。

### 复杂度分析

- 时间复杂度 O(N)：其中 N 为字符串长度，动态规划需遍历计算 dp 列表。
- 空间复杂度 O(1)：字符的 ASCII 码范围为 0~127 ，哈希表 dic 最多使用 O(128)=O(1) 大小的额外空间。

<div align=center><img src="/images/剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-49-49.png" alt="剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-49-49" style="zoom:50%;" /></div>

<div align=center><img src="/images/剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-50-10.png" alt="剑指 Offer48-最长不含重复字符的子字符串-2022-01-29-08-50-10" style="zoom:50%;" /></div>

Python 的 get(key,default) 方法代表当哈希表包含键 key 时返回对应 value，不包含时返回默认值 default。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res = tmp = 0
        for j in range(len(s)):
            i = dic.get(s[j], -1) # 获取索引 i
            dic[s[j]] = j # 更新哈希表
            tmp = tmp + 1 if tmp < j - i else j - i # dp[j - 1] -> dp[j]
            res = max(res, tmp) # max(dp[j - 1], dp[j])
        return res
```

## 方法二：动态规划 + 线性遍历 s

左边界 i 获取方式：遍历到 s[j] 时，初始化索引 i=j−1 ，向左遍历搜索第一个满足 [i]=s[j] 的字符即可 。
