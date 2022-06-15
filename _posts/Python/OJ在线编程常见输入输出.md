# OJ 在线编程常见输入输出

## 1. 多组输入 a + b

- 输入包括两个正整数 a,b(1 <= a, b <= 1000), 输入数据包括多组。
- 输出 a+b 的结果

```
输入
1 5
10 20

输出
6
30
```

```python
while True:
    try:
        num = list(map(int, input().strip().split()))
        print(num[0] + num[1])
    except:
        break
```

```python
while True:
    try:
        a, b = (eval(i) for i in input().strip().split())
        print(a + b)
    except:
        break
```

## 2. 确定 t 组输入，a + b

- 输入第一行包括一个数据组数 t(1 <= t <= 100)，接下来每行包括两个正整数 a,b(1 <= a, b <= 1000)
- 输出 a+b 的结果

```
输入
2
1 5
10 20

输出
6
30
```

```python
t = eval(input())
for i in range(t):
    num = list(map(int, input().split()))
    print(num[0] + num[1])
```

## 3. 以 (0 0) 为结束 a + b

- 输入包括两个正整数 a,b(1 <= a, b <= 10^9), 输入数据有多组，如果输入为 0 0 则结束输入
- 输出 a+b 的结果

```
输入
1 5
10 20
0 0

输出
6
30
```

```python
num = list(map(int, input().split()))
while num != [0,0]:
    print(num[0] + num[1])
    num = list(map(int, input().split()))
```

```python
a, b = (eval(i) for i in input().strip().split())
while (a, b) != (0, 0):
    print(a + b)
    a, b = (eval(i) for i in input().strip().split())
```

## 4. 多行 n 个数相加，以 0 为结束

- 输入数据包括多组。
- 每组数据一行，每行的第一个整数为整数的个数 n(1 <= n <= 100), n 为 0 的时候结束输入。接下来 n 个正整数，即需要求和的每个正整数。
- 输出每组数据输出求和的结果

```
输入
4 1 2 3 4
5 1 2 3 4 5
0

输出
10
15
```

```python
res = 0
num = list(map(int, input().strip().split()))
while num[0] != 0:
    for i in range(1, num[0]+1):
        res += num[i]
    print(res)
    res = 0
    num = list(map(int, input().strip().split()))
```

简化

```python
num = list(map(int, input().strip().split()))
while num[0] != 0:
    print(sum(num[1:]))
    num = list(map(int, input().strip().split()))
```

## 5. 确定 t 行 n 个数相加

- 输入的第一行包括一个正整数 t(1 <= t <= 100), 表示数据组数。接下来 t 行，每行一组数据。每行的第一个整数为整数的个数 n(1 <= n <= 100)。接下来 n 个正整数，即需要求和的每个正整数。
- 输出每组数据求和的结果

```
输入
2
4 1 2 3 4
5 1 2 3 4 5

输出
10
15
```

```python
t = int(input())
for i in range(t):
    num = list(map(int, input().split()))
    print(sum(num[1:]))
```

## 6. 多行 n 个数相加

- 输入数据有多组, 每行表示一组输入数据。每行的第一个整数为整数的个数n(1 <= n <= 100)。接下来n个正整数, 即需要求和的每个正整数。

```
输入
4 1 2 3 4
5 1 2 3 4 5

输出
10
15
```

```python
while True:
    try:
        num = list(map(int, input().split()))
        print(sum(num[1:]))
    except:
        break
```

## 7. 多行数相加

- 输入数据有多组, 每行表示一组输入数据。每行不定有n个整数，空格隔开。(1 <= n <= 100)。

```
输入
1 2 3
4 5
0 0 0 0 0

输出
6
9
0
```

```python
while True:
    try:
        num = list(map(int, input().strip().split()))
        print(sum(num[:]))
    except:
        break
```


## 8. n 个字符串排序

- 输入有两行，第一行 n，第二行是 n 个字符串，字符串之间用空格隔开
- 输出一行排序后的字符串，空格隔开，无结尾空格

```
输入
5
c d a bb e

输出
a bb c d e
```

```python
n = int(input())
l = input().split()
l.sort()
print(" ".join(l))
```

## 9. 多行字符串排序

- 多个测试用例，每个测试用例一行。每行通过空格隔开，有 n 个字符，n＜100
- 对于每组测试用例，输出一行排序过的字符串，每个字符串通过空格隔开

```
输入
a c bb
f dddd
nowcoder

输出
a bb c
dddd f
nowcoder
```

```python
while True:
    try:
        print(' '.join(sorted(input().split())))
    except:
        break
```


## 10. 多行数相加

- 多个测试用例，每个测试用例一行。每行通过 ',' 隔开，有 n 个字符，n＜100
- 对于每组用例输出一行排序后的字符串，用 ',' 隔开，无结尾空格

```
输入
a,c,bb
f,dddd
nowcoder

输出
a,bb,c
dddd,f
nowcoder
```

```python
while True:
    try:
        l = input().split(',')
        l.sort()
        print(','.join(l))
    except:
        break
```
