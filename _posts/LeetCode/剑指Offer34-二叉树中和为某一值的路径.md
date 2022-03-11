---
title: "剑指 Offer 34. 二叉树中和为某一值的路径"
subtitle: "LeetCode 刷题笔记"
layout: post
author: "L Hondong"
header-img: "img/post-bg-12.jpg"
mathjax: ture
tags:
  - LeetCode
  - 算法
---

# 剑指 Offer 34. 二叉树中和为某一值的路径

## 题目

给你二叉树的根节点`root`和一个整数目标和`targetSum`，找出所有**从根节点到叶子节点**路径总和等于给定目标和的路径。

叶子节点是指没有子节点的节点。

## 示例

示例 1:

<div align=center><img src="/assets/剑指Offer34-二叉树中和为某一值的路径-2022-02-01-09-42-42.png" alt="剑指Offer34-二叉树中和为某一值的路径-2022-02-01-09-42-42" style="zoom:50%;" /></div>

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
```

## 题解

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        path = []
        def recur(root, target):
            if not root: return # 当前节点为空，直接返回
            path.append(root.val)
            target -= root.val
            if target == 0 and not root.left and not root.right:
                res.append(list(path))
            recur(root.left, target)
            recur(root.right, target)
            path.pop()

        recur(root, target)
        return res
```

## 笔记

典型的二叉树方案搜索问题，使用回溯法解决，其包含**先序遍历 + 路径记录**两部分。

### pathSum(root, sum) 函数：

- 初始化：结果列表 res ，路径列表 path 。
- 返回值：返回 res 即可。

### recur(root, tar) 函数：

- 递推参数：当前节点 root ，当前目标值 tar 。
- 终止条件：若节点 root 为空，则直接返回。
- 递推工作：
  1. 路径更新：将当前节点值 root.val 加入路径 path ；
  2. 目标值更新：tar = tar - root.val（即目标值 tar 从 sum 减至 0）；
  3. 路径记录：当 ① root 为叶节点且 ② 路径和等于目标值，则将此路径 path 加入 res 。
  4. 先序遍历：递归左 / 右子节点。
  5. 路径恢复：向上回溯前，需要将当前节点从路径 path 中删除，即执行 path.pop() 。

<div align=center><img src="/assets/剑指Offer34-二叉树中和为某一值的路径-2022-02-01-09-47-51.png" alt="剑指Offer34-二叉树中和为某一值的路径-2022-02-01-09-47-51" style="zoom:50%;" /></div>

### 复杂度分析

- 时间复杂度 O(N)：N 为二叉树的节点数，先序遍历需要遍历所有节点。
- 空间复杂度 O(N)：最差情况下，即树退化为链表时，path 存储所有树节点，使用 O(N) 额外空间。

### 注意！

记录路径时若直接执行 res.append(path) ，则是将 path 对象加入了 res；后续 path 改变时， res 中的 path 对象也会随之改变。

正确做法：res.append(list(path)) ，相当于复制了一个 path 并加入到 res 。

### Leetcode 中的列表生成二叉树

[1, null, 2, 3] 是个串行化格式，表达了一个水平顺序遍历的二叉树。其中，null 表示某一分支上没有子节点。示例：

[1, null, 2, 3]

```
    1
      \
       2
      /
     3
```

[3,9,20,null,null,15,7]

```
    3
   / \
  9  20
    /  \
   15   7
```

这里的分层顺序列表表示二叉树与完全二叉树层次表示不同，如对于第一个二叉树，完全二叉树应表示为 [1,None,2,None,None,3,None]。

根据示例，给我们的是一个分层顺序列表类似于 [3,9,20,null,null,15,7]。我们可以非常直观的知道 3 是根结点， 9 和 20 是 3 的左右子节点， null 和 null 是 9 的左右子节点，15 和 7 是 20 的左右子节点。

从上面的描述中，可以发现分配左右节点的先后顺序也是 3,9,20，跟列表顺序是一致的，即先给 3 分配左右节点，再给 9 分配左右节点，再给 20 分配左右节点，如果列表更长，接下来会跳过 null，给 15 分配左右节点，再给 7 分配左右节点。

<div align=center><img src="/assets/剑指Offer34-二叉树中和为某一值的路径-2022-02-01-20-27-13.png" alt="剑指Offer34-二叉树中和为某一值的路径-2022-02-01-20-27-13" style="zoom:50%;" /></div>

代码实现：

```python
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def generate_tree(nums):
    if len(nums) == 0:
        return None
    que = [] # 定义队列
    fill_left = True # 由于无法通过是否为 None 来判断该节点的左儿子是否可以填充，用一个记号判断是否需要填充左节点
    for i in nums:
        node = TreeNode(i) if i else None # 非空值返回节点类，否则返回 None
        if len(que) == 0:
            root = node # 队列为空的话，用 root 记录根结点，用来返回
            que.append(node)
        elif fill_left:
            que[0].left = node
            fill_left = False # 填充过左儿子后，改变记号状态
            if node: # 非 None 值才进入队列
                que.append(node)
        else:
            que[0].right = node
            if node:
                que.append(node)
            que.pop(0) # 填充完右儿子，弹出节点
            fill_left = True # 
    return root

# 定义一个dfs打印中序遍历
def dfs(node):
    if node is not None:
        dfs(node.left)
        print(node.val, end=' ')
        dfs(node.right)
# 定义一个bfs打印层序遍历
def bfs(node):
    que = []
    que.append(node)
    while que:
        l = len(que)
        for _ in range(l):
            tmp = que.pop(0)
            print(tmp.val, end=' ')
            if tmp.left:
                que.append(tmp.left)
            if tmp.right:
                que.append(tmp.right)
        print('|', end=' ')

null = None
nums = [5,4,8,11,null,13,4,7,2,null,null,5,1]
root = generate_tree(nums)

# test
null = None
vals = [3,9,20,null,null,15,7]
tree = generate_tree(vals) 
print('中序遍历:')    
dfs(tree) # 9 3 15 20 7 
print('\n层序遍历:')
bfs(tree) # 3 | 9 20 | 15 7 |
```

输出结果：

```
中序遍历:
9 3 15 20 7 
层序遍历:
3 | 9 20 | 15 7 |
```
