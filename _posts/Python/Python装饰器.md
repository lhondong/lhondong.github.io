# Python 装饰器

1. 需要对已开发上线的程序添加某些功能
2. 不能对程序中函数的源代码进行修改
3. 不能改变程序中函数的调用方式

## 作用域、内嵌函数、生命周期

Python 中每个函数都是一个新的作用域，也可以理解为命名空间。函数是 Python 中的第一类对象

1. 可以把函数赋值给变量  
2. 对该变量进行调用，可实现原函数的功能
3. 可以将函数作为参数进行传递

一般而言在函数中定义的变量、内嵌函数等，其生命周期即所在函数的作用域，当函数执行完毕后，函数内所定义的变量、内嵌函数等应该会消失。而在下一次函数被调用时又会被重新创建。

### 变量搜索

当在函数中访问一个新的变量时，Python 会在当前的命名空间中寻找该变量是否存在。如果变量不存在则会从上一级命名空间中搜寻，直到顶层命名空间。例如：

```python
def outer():
    x = 1
    def inner():
        print(x)
    inner()
outer() # 打印结果为 1
```

当调用 `outer()` 函数时，内嵌函数 `inner()` 会被调用，当执行到 `print(x)` 语句时，Python 会在 `inner()` 中搜寻局部变量 `x`，但是没有查找到，因此会在上一级命名空间，也就是 `outer()` 中搜寻，在上一级中搜寻到 `x`，最后将其打印。

但是在函数中对一个变量进行定义或者赋值时，Python **只会在当前命名空间中搜寻**该变量，如果不存在则会创建一个新的变量。如果上一级命名空间中有同名的变量，那么上一级同名的变量会在当前作用域中被覆盖。

```python
def outer():
    x = 1
    def inner():
        x = 2
        print("The x in inner() is %d" % x)
    print("The x in outer() is %d" % x)

outer()
# 打印结果为
# The x in inner() is 2
# The x in outer() is 1
```

## 闭包

下面的代码介绍闭包的概念：

```python
def outer():
    x = 1
    def inner():
        print(x)
    return inner

foo = outer()
foo()
print(foo.__closure__)
# 打印结果为
# 1
# (<cell at 0x00000000004AF2E8: int object at 0x000000006F14B440>,)
```

从作用域的角度，`foo()` 实际上是调用了内嵌函数 `inner()`，当执行到 `inner()` 中的 `print(x)` 语句时，在 `inner()` 中没有搜寻到 `x`，然后会在 `outer()` 的命名空间中搜寻，找到 `x` 后进行打印。

但从生命周期的角度，`foo` 的值为 `outer()` 函数的返回值，当执行 `foo()` 时，`outer()` 函数已经执行完毕了，此时其作用域内定义的变量 `x` 也应该已经销毁，因此执行 `foo()` 时，当执行到 `pirnt(x)` 语句应该会出错。但实际执行过程中却没有。

这其实是 Python 支持的函数**闭包**的特性，在上面的例子中可以作这样的理解：在非全局作用域定义的`inner()`函数在定义时会记住外层命名空间，此时 `inner()` 函数包含了外层作用域变量 `x`。`function` 对象的 `__closure__` 属性指示了该函数对象是否是闭包函数，若不是闭包函数，则该属性值为 `None`，否则为一个非空元组。

将上述例子进行一些修改：

```python
def outer(x):
    def inner():
        print("The value of x is %d " % x)
    return inner

foo1 = outer(1)
foo2 = outer(2)

foo1()
foo2()
​
# 打印结果
# The value of x is 1
# The value of x is 2
```

可以看到当传递给 `outer()` 不同的参数时，得到的 `inner()` 打印结果是不同的，这证明了闭包函数 `inner()` 记录了其外层命名空间。

nonlocal 允许内嵌的函数来修改闭包变量。

```python
def outer():
    x = 1
    def inner():
        nonlocal x  # nonlocal 允许内嵌的函数来修改闭包变量
        x = x + 100
        return x  
    return inner
f = outer()             
f()
```

可以利用闭包的特性得到一个对已有函数运行行为进行扩充或修改的新函数，而同时保留已有函数，不用对已有函数的代码进行修改。做到这个的第一步是将函数作为参数传递到我们的 “闭包创建函数” 中。

## 装饰器

装饰器其实就是一个以函数作为参数并返回一个替换函数的可执行函数，即**装饰器是一个函数，它以函数作为参数，返回另一个函数**。

下面是一个装饰器的示例：

```python
def outer(some_func):
    def inner():
        print("before some_func")
        ret = some_func()
        return ret + 1
    return inner

def foo():
    return 1

decorated = outer(foo)
decorated()

# 打印结果
# before some_func
# 2
```

示例中 `decorated` 是 `foo` 的装饰，即给 `foo` 加上了一些东西。在实际使用中实现装饰器后可能就想用装饰器替换原来的函数了，只需要给 `foo` 重新赋值即可：

```python
foo = outer(foo)
```

### 函数装饰器 @符号的应用

Python 通过在函数定义前添加一个装饰器名和 `@` 符号，来实现对函数的包装。在上面代码示例中，用了一个包装的函数来替换包含函数的变量来实现了装饰函数。

```python
foo = wrapper(foo)
```

这种模式可以随时用来包装任意函数。但是如果定义了一个函数，可以用 `@` 符号来装饰函数，如下：

```python
@wrapper
def foo():
    pass
```

值得注意的是，这种方式和简单的使用 `wrapper()` 函数的返回值来替换原始变量的做法没有什么不同， Python 只是添加了一些语法糖来使之看起来更加明确。

使用装饰器很简单！虽说写类似 `staticmethod` 或者 `classmethod` 的实用装饰器比较难，但用起来仅仅需要在函数前添加 `@装饰器名` 即可！

## 更通用的函数装饰器

### 装饰有参函数

```python
def wrapper(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
    return inner

@wrapper
def func1(a, b, c):
    print('a=', a, 'b=', b, 'c=', c)

func1(1, b = 'boy', c = 'cat')
# a= 1 b= boy c= cat

@wrapper
def func2(v1, v2):
    print(v1, v2)

func2(100, 200)
# 100 200
```

```python
import time
def timer(func):  
    def inner(*args, **kwargs):
        print("inner run")
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("{} 函数运行用时{:.2f}秒".format(func.__name__, (end-start)))
    return inner

@timer                # 相当于实现了f1 = timer(f1)
def f1(n):
    print("f1 run")
    time.sleep(n)
    
f1(2)

# Out:
# inner run
# f1 run
# f1 函数运行用时2.00秒
```

### 被装饰函数有返回值的情况

```python
import time
def timer(func):
    def inner(*args, **kwargs):
        print("inner run")
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("{} 函数运行用时{:.2f}秒".format(func.__name__, (end-start)))
        return res
    return inner

@timer      # 相当于实现了f1 = timer(f1)
def f1(n):
    print("f1 run")
    time.sleep(n)
    return "wake up"
    
res = f1(2)
print(res)
```

Output
```
inner run
f1 run
f1 函数运行用时2.00秒
wake up
```

### 带参数的装饰器

装饰器本身要传递一些额外参数，有时需要统计绝对时间，有时需要统计绝对时间的2倍。

```python
def timer(method):  
    def outer(func):
        def inner(*args, **kwargs):
            print("inner run")
            if method == "origin":
                print("origin_inner run")
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                print("{} 函数运行用时{:.2f}秒".format(func.__name__, (end-start)))
            elif method == "double":
                print("double_inner run")
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                print("{} 函数运行双倍用时{:.2f}秒".format(func.__name__, 2*(end-start)))
            return res
        return inner
    return outer

@timer(method="origin")  # 相当于timer = timer(method = "origin")   f1 = timer(f1)
def f1():
    print("f1 run")
    time.sleep(1)  
    
@timer(method="double")
def f2():
    print("f2 run")
    time.sleep(1)

f1()
print()
f2()
```

Output

```
inner run
origin_inner run
f1 run
f1 函数运行用时1.00秒

inner run
double_inner run
f2 run
f2 函数运行双倍用时2.00秒
```

### 何时执行装饰器

一装饰就执行，不必等调用

```python
func_names=[]
def find_function(func):
    print("run")
    func_names.append(func)
    return func

@find_function
def f1():
    print("f1 run")

@find_function
def f2():
    print("f2 run")

@find_function
def f3():
    print("f3 run")
```

在不调用的情况下，会先 `print("run")`。

```
run
run
run
```

实际调用的情况：

```python
for func in func_names:
    print(func.__name__)
    func()
    print()
```

```
f1
f1 run

f2
f2 run

f3
f3 run
```

### 回归本源

原函数的属性被掩盖了

```python
import time

def timer(func):
    def inner():
        print("inner run")
        start = time.time()
        func()
        end = time.time()
        print("{} 函数运行用时{:.2f}秒".format(func.__name__, (end-start)))
    return inner

@timer                # 相当于实现了 f1 = timer(f1)
def f1():
    time.sleep(1)
    print("f1 run")

print(f1.__name__)
# inner
```

因为返回的那个 `wrapper()` 函数名字就是 'inner'，所以，有时需要把原始函数的 `__name__` 等属性复制到 `wrapper()` 函数中，否则，有些依赖函数签名的代码执行就会出错。

```python
import functools
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper
```

例如：

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def inner():
        print("inner run")
        start = time.time()
        func()
        end = time.time()
        print("{} 函数运行用时{:.2f}秒".format(func.__name__, (end-start)))
    
    return inner

@timer                # 相当于实现了f1 = timer(f1)
def f1():
    time.sleep(1)
    print("f1 run")

print(f1.__name__) 
f1()
```

输出：

```
f1
inner run
f1 run
f1 函数运行用时1.00秒
```

## 使用类作为装饰器

类也可以作为装饰器，使用起来可能比函数装饰器更方便。首先看下面一个简单的例子：

```python
class myDecorator(object):
    def __init__(self, f):
        print("inside myDecorator.__init__()")
        f() # Prove that function definition has completed
    def __call__(self):
        print("inside myDecorator.__call__()")

@myDecorator
def aFunction():
    print("inside aFunction()")

print("Finished decorating aFunction()")

aFunction()
```

例子中函数 `aFunction()` 就使用了类 `myDecorator` 作为装饰器修饰。例子的输出结果如下：

```
inside myDecorator.__init__()
inside aFunction()
Finished decorating aFunction()
inside myDecorator.__call__()
```

可以看出，在 `aFunction()` 函数声明处进入了类 `myDecorator` 的 `__init__()` 方法，但要注意，从第 2 个输出可以看出，此时函数`aFunction()`的定义已经完成了，在 `__init__()` 中调用的输入参数 `f()`，实际上是调用了`aFunction()`函数。至此 `aFunction()` 函数的声明完成，包括装饰器声明的部分，然后输出了第 3 个输出。最后执行 `aFunction()` 时，**可以看出实际上是执行了类 `myDecorator` 的 `__call__()` 方法**（定义了 `__call__()` 方法的类的对象可以像函数一样被调用，此时调用的是对象的 `__call__()` 方法）。

这个例子其实不难理解，因为根据装饰器语法的含义，下面的代码：

```python
@myDecorator
def aFunction():
    pass
```

等价于

```python
def aFunction():
    pass
aFunction = myDecorator(aFunction)
```

因此被装饰后的函数 `aFunction()` 实际上已经是类 `myDecorator` 的对象。当再调用 `aFunction()` 函数时，实际上就是调用类 `myDecorator` 的对象，因此会调用到类 `myDecorator` 的 `__call__()` 方法。

因此使用类作为装饰器装饰函数来对函数添加一些额外的属性或功能时，一般会在类的 `__init__()` 方法中记录传入的函数，再在 `__call__()` 调用修饰的函数及其它额外处理。

下面是一个简单的例子：

```python
class entryExit(object):
    def __init__(self, f):
        self.f = f
    def __call__(self):
        print("Entering", self.f.__name__)
        self.f()
        print("Exited", self.f.__name__)

@entryExit
def func1():
    print("inside func1()")

@entryExit
def func2():
    print("inside func2()")

func1()
func2()
```

输出：

```
Entering func1
inside func1()
Exited func1
Entering func2
inside func2()
Exited func2
```

## 使用对象作为装饰器

根据装饰器的语法，对象当然也可以作为装饰器使用，对比使用类作为装饰器，使用对象作为装饰器有些时候更有灵活性，例如能够方便定制和添加参数。下面是一个例子：

```python
class Decorator:
    def __init__(self, arg1, arg2):
        print('执行类 Decorator 的__init__() 方法')
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, f):
        print('执行类 Decorator 的__call__() 方法')
        def wrap(*args):
            print('执行 wrap()')
            print('装饰器参数：', self.arg1, self.arg2)
            print('执行' + f.__name__ + '()')
            f(*args)
            print(f.__name__ + '() 执行完毕')
        return wrap

@Decorator('Hello', 'World')
def example(a1, a2, a3):
    print('传入 example() 的参数：', a1, a2, a3)

print('装饰完毕')

print('准备调用 example()')
example('Wish', 'Happy', 'EveryDay')
print('测试代码执行完毕')
```

输出为：

```
执行类 Decorator 的__init__() 方法
执行类 Decorator 的__call__() 方法
装饰完毕
准备调用 example()
执行 wrap()
装饰器参数： Hello World
执行 example()
传入 example() 的参数： Wish Happy EveryDay
example() 执行完毕
测试代码执行完毕
```

根据装饰器的语法，下面的代码：

```python
@Decorator('Hello', 'World')
def example(a1, a2, a3):
    pass
```

等价于

```python
def example(a1, a2, a3):
    pass

example = Decorator('Hello', 'World')(example)
```

此时就不难理解例子中的输出了，`@Decorator('Hello', 'World')` 实际上生成了一个类 `Decorator` 的对象，然后该对象作为装饰器修饰 `example()` 函数，修饰过程就是调用了 `Decorator` 对象的 `__call__()` 方法来 “封装” `exmaple()`，最后 `example()` 函数的实际上是闭包后，`__call__()` 方法中定义的 `wrap()` 函数。