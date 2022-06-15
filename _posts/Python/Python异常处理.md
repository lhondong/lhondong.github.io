
# 异常处理

## 常见异常的产生

- `ZeroDivisionError`: 除 0 运算
- `FileNotFoundError`: 找不到可读文件
- `IndexError`: 索引错误，下标超出序列边界
- `ValueError`: 值错误，传入一个调用者不期望的值，即使这个值的类型是正确的
- `TypeError`: 类型错误，传入对象类型与要求不符
- `NameError` 使用一个未被定义的变量  
- `KeyError` 试图访问字典里不存在的键

**当异常发生的时候，如果不预先设定处理方法，程序就会中断**

## 异常的处理

**当异常发生的时候，如果不预先设定处理方法，程序就会中断**

### try except

- 如果 try 内代码块顺利执行，except 不被触发
- 如果 try 内代码块发生错误，触发 except, 执行 except 内代码块

- 单分支

```python
x = 10
y = 0
try:
    z = x/y
except ZeroDivisionError:   # 一般来说会预判到出现什么错误
    # z = x/(y+1e-7)
    # print(z)
    print("0不可以被除！")  
```

- 多分支

```python
ls = []
d = {"name": "Tom"}
try:
    y = m  # NameError
    # ls[3] # 索引超出界限
    # d["age"] # 键不存在
except NameError:
    print("变量名不存在")
except IndexError:
    print("索引超出界限")
except KeyError:
    print("键不存在")
```

- 万能异常 Exception （所有错误的祖先）

```python
ls = []
d = {"name": "Tom"}
try:
    # y = m
    ls[3]
    # d["age"]
except Exception:
    print("出错啦")

>>> 出错啦
```

- 捕获异常的值 as 

```python
ls = []
d = {"name": "Tom"}
# y = x
try:
    y = m
    # ls[3]
    # d["age"]
except Exception as e:    # 虽不能获得错误具体类型，但可以获得错误的值
    print(e)

>>> name 'm' is not defined
```

- try_except_else 如果 try 模块执行，则 else 模块也执行  

```python
try:
    with open("浪淘沙_北戴河.txt") as f:
        text = f.read()
except FileNotFoundError:
    print("找不到该文件")
else:
    for s in ["\n", "，", "。", "？"]:         # 去掉换行符和标点符号
        text = text.replace(s, "")
    print("浪淘沙北戴河共由{}个字组成。".format(len(text)))
```

- try_except_finally 不论 try 模块是否执行，finally 最后都执行

```python
ls = []
d = {"name": "Tom"}
# y = x
try:
    y = m
    # ls[3]
    # d["age"]
except Exception as e:    # 虽不能获得错误具体类型，但可以获得错误的值
    print(e)
finally:
    print("不论触不触发异常，都将执行")

>>> name 'm' is not defined
>>> 不论触不触发异常，都将执行
```

## assert 断言

- 用于判断一个表达式，在表达式条件为 false 的时候触发异常。

```python
assert expression

等价于：

if not expression:
    raise AssertionError
```

assert 后面也可以紧跟参数：

```python
assert expression [, arguments]

等价于：

if not expression:
    raise AssertionError(arguments)
```

- 实例

```python
assert True     # 条件为 true 正常执行
assert False    # 条件为 false 触发异常
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError

assert 1==1    # 条件为 true 正常执行
assert 1==2    # 条件为 false 触发异常
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError

assert 1==2, '1 不等于 2'
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError: 1 不等于 2
```