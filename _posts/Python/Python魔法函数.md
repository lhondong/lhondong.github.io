# Python 魔法函数

魔法函数（Magic Methods）是 Python 的一种高级语法，允许你在类中自定义函数（函数名格式一般为__xx__），并绑定到类的特殊方法中。比如在类 A 中自定义__str__() 函数，则在调用 str(A()) 时，会自动调用__str__() 函数，并返回相应的结果。在平时使用中，可能经常使用的__init__函数（构造函数）和__del__函数（析构函数），也是魔法函数的一种。

- Python 中以双下划线 (__xx__) 开始和结束的函数（不可自己定义）为魔法函数。
- 调用类实例化的对象的方法时自动调用魔法函数。
- 在自己定义的类中，可以实现之前的内置函数。

## 魔法函数的作用

魔法函数可以为你写的类增加一些额外功能，方便使用者理解。举个简单的例子，我们定义一个“人”的类 People，当中有属性姓名 name、年龄 age。让你需要利用 sorted 函数对一个 People 的数组进行排序，排序规则是按照 name 和 age 同时排序，即 name 不同时比较 name，相同时比较 age。由于 People 类本身不具有比较功能，所以需要自定义，你可以这么定义 People 类：

```python
class People(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        return

    def __str__(self):
        return self.name + ":" + str(self.age)

    def __lt__(self, other):
        return self.name < other.name if self.name != other.name else self.age < other.age

if __name__=="__main__":

    print("\t".join([str(item) for item in sorted([People("abc", 18),
        People("abe", 19), People("abe", 12), People("abc", 17)])]))
```

输出结果：

```
abc:17	abc:18	abe:12	abe:19
```

上个例子中的 `__lt__` 函数即 less than 函数，即当比较两个 People 实例时自动调用。

## 常见的魔法函数

### 1. __ init__()

所有类的超类 object，有一个默认包含 pass 的 __init__() 实现，这个函数会在对象初始化的时候调用，我们可以选择实现，也可以选择不实现，一般建议是实现的，不实现对象属性就不会被初始化。

__init__() 方法可以包含多个参数，但必须包含一个名为 self 的参数，且必须作为第一个参数。也就是说，类的构造方法最少也要有一个 self 参数，仅包含 self 参数的 __init__() 构造方法，又称为类的默认构造方法。例如，以 TheFirstDemo 类为例，添加构造方法的代码如下所示：

```python
class TheFirstDemo:
        # 构造方法
        def __init__(self):
                print("调用构造方法")

        # 下面定义了一个类属性
        pass

        # 下面定义了一个 say 方法
        def say(self, content):
                print(content)

if __name__ == "__main__":
        result = TheFirstDemo()
```

输出结果：

```
调用构造方法
```

在创建 result 这个对象时，隐式调用了我们手动创建的 __init__() 构造方法。

不仅如此，在 __init__() 构造方法中，除了 self 参数外，还可以自定义一些参数，参数之间使用逗号 “,” 进行分割。例如，下面的代码在创建 __init__() 方法时，额外指定了 2 个参数：

```python
class CLanguage:
    '''这是一个学习 Python 定义的一个类'''
    def __init__(self, name, add):
        print(name,"的网址为：",add)

#创建 add 对象，并传递参数给构造函数
add = CLanguage("C 语言中文网","http://c.biancheng.net")
```

输出结果：

```
C 语言中文网 的网址为：http://c.biancheng.net
```

可以看到，虽然构造方法中有 self、name、add 3 个参数，但实际需要传参的仅有 name 和 add，也就是说，self 不需要手动传递参数。

### 2. __ str__()

直接打印对象的实现方法，__str__ 是被 print 函数调用的。打印一个实例化对象时，打印的其实是一个对象的地址。而通过 __str__() 函数就可以帮助我们打印对象中具体的属性值，或者你想得到的东西。

在 Python 中调用 print() 打印实例化对象时会调用__str__()。如果__str__() 中有返回值，就会打印其中的返回值。

```python
class Cat:
    """定义一个猫类"""
 
    def __init__(self, new_name, new_age=20):
        """在创建完对象之后 会自动调用，它完成对象的初始化的功能"""
        self.name = new_name
        self.age = new_age  # 它是一个对象中的属性，在对象中存储，即只要这个对象还存在，那么这个变量就可以使用
        # num = 100  # 它是一个局部变量，当这个函数执行完之后，这个变量的空间就没有了，因此其他方法不能使用这个变量
 
    def __str__(self):
        """返回一个对象的描述信息"""
        # print(num)
        return "名字是：%s , 年龄是：%d" % (self.name, self.age)

# 创建了一个对象
tom = Cat("汤姆", 30)
print(tom)
```

输出结果：

```
名字是：汤姆 , 年龄是：30
```

总结：当使用 print 输出对象的时候，只要自己定义了 __str__(self) 方法，那么就会打印从在这个方法中 return 的数据。__str__ 方法需要返回一个字符串，当做这个对象的描写。

### 3. __ new__()

__new__() 是一种负责创建类实例的静态方法，它无需使用 staticmethod 装饰器修饰，且该方法会优先 __init__() 初始化方法被调用。

一般情况下，覆写 __new__() 的实现将会使用合适的参数调用其超类的 super().__new__()，并在返回之前修改实例。例如：

```python
class demoClass:
    instances_created = 0
    def __new__(cls,*args,**kwargs):
        print("__new__():",cls,args,kwargs)
        instance = super().__new__(cls)
        instance.number = cls.instances_created
        cls.instances_created += 1
        return instance
    def __init__(self,attribute):
        print("__init__():",self,attribute)
        self.attribute = attribute
test1 = demoClass("abc")
test2 = demoClass("xyz")
print(test1.number,test1.instances_created)
print(test2.number,test2.instances_created)
```

输出结果：

```
__new__(): <class '__main__.demoClass'> ('abc',) {}
__init__(): <__main__.demoClass object at 0x0000025650FACF28> abc
__new__(): <class '__main__.demoClass'> ('xyz',) {}
__init__(): <__main__.demoClass object at 0x000002565FFC4CF8> xyz
0 2
1 2
```

__new__() 通常会返回该类的一个实例，但有时也可能会返回其他类的实例，如果发生了这种情况，则会跳过对 __init__() 方法的调用。而在某些情况下（比如需要修改不可变类实例（Python 的某些内置类型）的创建行为），利用这一点会事半功倍。比如：

```python
class nonZero(int):
    def __new__(cls,value):
        return super().__new__(cls,value) if value != 0 else None
    def __init__(self,skipped_value):
        #此例中会跳过此方法
        print("__init__()")
        super().__init__()
print(type(nonZero(-12)))
print(type(nonZero(0)))
```

输出结果：

```
__init__()
<class '__main__.nonZero'>
<class 'NoneType'>
```

那么，什么情况下使用 __new__() 呢？答案很简单，在 __init__() 不够用的时候。

例如，前面例子中对 Python 不可变的内置类型（如 int、str、float 等）进行了子类化，这是因为一旦创建了这样不可变的对象实例，就无法在 __init__() 方法中对其进行修改。

有人可能会认为，__new__() 对执行重要的对象初始化很有用，如果用户忘记使用 super()，可能会漏掉这一初始化。虽然这听上去很合理，但有一个主要的缺点，即如果使用这样的方法，那么即便初始化过程已经是预期的行为，程序员明确跳过初始化步骤也会变得更加困难。不仅如此，它还破坏了 “__init__() 中执行所有初始化工作” 的潜规则。

注意，由于 __new__() 不限于返回同一个类的实例，所以很容易被滥用，不负责任地使用这种方法可能会对代码有害，所以要谨慎使用。一般来说，对于特定问题，最好搜索其他可用的解决方案，最好不要影响对象的创建过程，使其违背程序员的预期。比如说，前面提到的覆写不可变类型初始化的例子，完全可以用工厂方法（一种设计模式）来替代。

### 2.4 __ unicode__()

__ unicode__() 方法是在一个对象上调用 unicode() 时被调用的。因为 Django 的数据库后端会返回 Unicode 字符串给 model 属性，所以我们通常会给自己的 model 写一个__ unicode__() 方法。如果定义了__ unicode__() 方法但是没有定义__ str__() 方法，Django 会自动提供一个__ str__() 方法调用 __ unicode__() 方法，然后把结果转换为 UTF-8 编码的字符串对象，所以在一般情况下，只定义__ unicode__() 方法，让 Django 来处理字符串对象的转换，看一个小栗子：

```
class Demo(object):
        def __init__(self):
                self.a = 1
        def __unicode__(self):
                return f"the value is {self.a}"

print(unicode(Demo()))
```

输出结果：

```
the value is 1
```

在 django 中，虽然没有定义__ str__，但是 django 会将__ unicode__转为了 str，当然你调用 unicode 更加是没有问题的。

### 2.5 __ call__()

该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以 “对象名 ()” 的形式使用。

```
class CLanguage:
    # 定义__call__方法
    def __call__(self,name,add):
        print("调用__call__() 方法",name,add)

clangs = CLanguage()
clangs("C 语言中文网","http://c.biancheng.net")
```

程序执行结果为：

```
调用__call__() 方法 C 语言中文网 http://c.biancheng.net
```

可以看到，通过在 CLanguage 类中实现 __call__() 方法，使的 clangs 实例对象变为了可调用对象。

> Python 中，凡是可以将 () 直接应用到自身并执行，都称为可调用对象。可调用对象包括自定义的函数、Python 内置函数以及本节所讲的类实例对象。

对于可调用对象，实际上 “名称 ()” 可以理解为是 “名称。__call__()” 的简写。仍以上面程序中定义的 clangs 实例对象为例，其最后一行代码还可以改写为如下形式：

```
clangs.__call__("C 语言中文网","http://c.biancheng.net")
```

运行程序会发现，其运行结果和之前完全相同。

用 __call__() 弥补 hasattr() 函数的短板：hasattr() 函数的用法，该函数的功能是查找类的实例对象中是否包含指定名称的属性或者方法，但该函数有一个缺陷，即它无法判断该指定的名称，到底是类属性还是类方法。

要解决这个问题，我们可以借助可调用对象的概念。要知道，类实例对象包含的方法，其实也属于可调用对象，但类属性却不是。举个例子：

```
class CLanguage:
    def __init__ (self):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
    def say(self):
        print("我正在学 Python")
clangs = CLanguage()
if hasattr(clangs,"name"):
    print(hasattr(clangs.name,"__call__"))
print("**********")
if hasattr(clangs,"say"):
    print(hasattr(clangs.say,"__call__"))
```

程序执行结果为：

```
False
**********
True
```

可以看到，由于 name 是类属性，它没有以 __call__ 为名的 __call__() 方法；而 say 是类方法，它是可调用对象，因此它有 __call__() 方法。

### 2.6 __ len__()

在 Python 中，如果你调用 len() 函数试图获取一个对象的长度，实际上，在 len() 函数内部，它自动去调用该对象的__len__() 方法。

```
class Students():
    def __init__(self, *args):
        self.names = args
    def __len__(self):
        return len(self.names)

ss = Students('Bob', 'Alice', 'Tim')
print(len(ss))
```

输出结果：

### 2.7 __repr__()

函数 str() 用于将值转化为适于人阅读的形式，而 repr() 转化为供解释器读取的形式，某对象没有适于人阅读的解释形式的话，str() 会返回与 repr()，所以 print 展示的都是 str 的格式。

我们经常会直接输出类的实例化对象，例如：

```
class CLanguage:
    pass
clangs = CLanguage()
print(clangs)
```

程序运行结果为：

```
<__main__.CLanguage object at 0x000001A7275221D0>
```

通常情况下，直接输出某个实例化对象，本意往往是想了解该对象的基本信息，例如该对象有哪些属性，它们的值各是多少等等。但默认情况下，我们得到的信息只会是 “类名 + object at+ 内存地址”，对我们了解该实例化对象帮助不大。

那么，有没有可能自定义输出实例化对象时的信息呢？答案是肯定，通过重写类的 __repr__() 方法即可。事实上，当我们输出某个实例化对象时，其调用的就是该对象的 __repr__() 方法，输出的是该方法的返回值。

以本节开头的程序为例，执行 print(clangs) 等同于执行 print(clangs.__repr__())，程序的输出结果是一样的（输出的内存地址可能不同）。

和 __init__(self) 的性质一样，Python 中的每个类都包含 __repr__() 方法，因为 object 类包含 __reper__() 方法，而 Python 中所有的类都直接或间接继承自 object 类。

默认情况下，__repr__() 会返回和调用者有关的 “类名 + object at + 内存地址” 信息。当然，我们还可以通过在类中重写这个方法，从而实现当输出实例化对象时，输出我们想要的信息。

举个例子：

```
class CLanguage:
    def __init__(self):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
    def __repr__(self):
        return "CLanguage[]"
clangs = CLanguage()
print(clangs)
```

程序运行结果为：

```
CLanguage[name=C 语言中文网，add=http://c.biancheng.net]
```

由此可见，__repr__() 方法是类的实例化对象用来做 “自我介绍” 的方法，默认情况下，它会返回当前对象的“类名 + object at + 内存地址”，而如果对该方法进行重写，可以为其制作自定义的自我描述信息。

### 2.8 __ setattr__()

在类中对属性进行赋值操作时，python 会自动调用__setattr__() 函数，来实现对属性的赋值。但是重写__setattr__() 函数时要注意防止无限递归的情况出现，一般解决办法有两种，一是用通过 super() 调用__setatrr__() 函数，二是利用字典操作对相应键直接赋值。

简单的说，__setattr__() 在属性赋值时被调用，并且将值存储到实例字典中，这个字典应该是 self 的__dict__属性。即：**在类实例的每个属性进行赋值时，都会首先调用__setattr__() 方法，并在__setattr__() 方法中将属性名和属性值添加到类实例的__dict__属性中**。

**实例属性管理__dict__：**

下面的测试代码中定义了三个实例属性，每个实例属性注册后都 print() 此时的__dict__，代码如下：

```
class AnotherFun:
    def __init__(self):
        self.name = "Liu"
        print(self.__dict__)
        self.age = 12
        print(self.__dict__)
        self.male = True
        print(self.__dict__)
another_fun = AnotherFun()
```

得到的结果显示出，每次实例属性赋值时，都会将属性名和对应值存储到__dict__字典中：

```
{'name': 'Liu'}
{'name': 'Liu', 'age': 12}
{'name': 'Liu', 'age': 12, 'male': True}
```

**__setattr__() 与__dict__：**

由于每次类实例进行属性赋值时都会调用__setattr__()，所以可以重载__setattr__() 方法，来动态的观察每次实例属性赋值时__dict__() 的变化。下面的 Fun 类重载了__setattr__() 方法，并且将实例的属性和属性值作为__dict__的键 - 值对：

```
class Fun:
    def __init__(self):
        self.name = "Liu"
        self.age = 12
        self.male = True
        
    def __setattr__(self, key, value):
        print("*"*50)
        print("setting:{},  with:{}".format(key[], value))
        print("current __dict__ : {}".format(self.__dict__))
        # 属性注册
        self.__dict__[key] = value
fun = Fun()
```

通过在__setattr__() 中将属性名作为 key，并将属性值作为 value，添加到了__dict__中，得到的结果如下：

```
**************************************************
setting:name,  with:Liu
current __dict__ : {}
**************************************************
setting:age,  with:12
current __dict__ : {'name': 'Liu'}
**************************************************
setting:male,  with:True
current __dict__ : {'name': 'Liu', 'age': 12}
```

可以看出，__init__() 中三个属性赋值时，每次都会调用一次__setattr__() 函数。

**重载__setattr__() 必须谨慎：**

由于__setattr__() 负责在__dict__中对属性进行注册，所以自己在重载时必须进行属性注册过程，下面是__setattr__() 不进行属性注册的例子：

```
class NotFun:
    def __init__(self):
        self.name = "Liu"
        self.age = 12
        self.male = True
    
    def __setattr__(self, key, value):
        pass
not_fun = NotFun()
print(not_fun.name)
```

由于__setattr__中并没有将属性注册到__dict__中，所以 not_fun 对象并没有 name 属性，因此最后的 print（not_fun.name）会报出属性不存在的错误：

```
AttributeError                            Traceback (most recent call last)
<ipython-input-21-6158d7aaef71> in <module>()
      8         pass
      9 not_fun = NotFun()
---> 10 print(not_fun.name)

AttributeError: 'NotFun' object has no attribute 'name'
```

所以，重载__setattr__时必须要考虑是否在__dict__中进行属性注册。

**总结：**Python 的实例属性的定义、获取和管理可以通过__setattr__() 和__dict__配合进行，当然还有对应的__getattr__() 方法，本文暂时不做分析。__setattr__() 方法在类的属性赋值时被调用，并通常需要把属性名和属性值存储到 self 的__dict__字典中。

### 2.9 __ getattr__()

当我们访问一个不存在的属性的时候，会抛出异常，提示我们不存在这个属性。而这个异常就是__getattr__方法抛出的，其原因在于他是访问一个不存在的属性的最后落脚点，作为异常抛出的地方提示出错再适合不过了。

看例子，我们找一个存在的属性和不存在的属性：

```
class A(object):
    def __init__(self, value):
        self.value = value

    def __getattr__(self, item):
        print("into __getattr__")
        return "can not find"

a = A(10)
print(a.value)
# 10
print(a.name)
# into __getattr__
# can not find
```

输出结果：

```
into __getattr__
can not find
```

### 2.10 __ getattribute__()

首先理解__getattribute__的用法，先看代码：

```
class Tree(object):
    def __init__(self,name):
        self.name = name
        self.cate = "plant"
    def __getattribute__(self,obj):
        print("哈哈")
        return object.__getattribute__(self,obj)
aa = Tree("大树")
print(aa.name)
```

执行结果是：

```
哈哈
大树
```

为什么会这个结果呢？

__getattribute__是属性访问拦截器，就是当这个类的属性被访问时，会自动调用类的__getattribute__方法。即在上面代码中，当我调用实例对象 aa 的 name 属性时，不会直接打印，而是把 name 的值作为实参传进__getattribute__方法中（参数 obj 是我随便定义的，可任意起名），经过一系列操作后，再把 name 的值返回。Python 中只要定义了继承 object 的类，就默认存在属性拦截器，只不过是拦截后没有进行任何操作，而是直接返回。所以我们可以自己改写__getattribute__方法来实现相关功能，比如查看权限、打印 log 日志等。如下代码，简单理解即可：

```
class Tree(object):
    def __init__(self,name):
        self.name = name
        self.cate = "plant"
    def __getattribute__(self,*args,**kwargs):
        if args[0] == "大树"
            print("log 大树")
            return "我爱大树"
        else:
            return object.__getattribute__(self,*args,**kwargs)
aa = Tree("大树")
print(aa.name)
print(aa.cate)
```

结果是：

```
log 大树
我爱大树
plant
```

**另外，注意注意：**

初学者用__getattribute__方法时，容易栽进这个坑，什么坑呢，直接看代码：

```
class Tree(object):
    def __init__(self,name):
        self.name = name
        self.cate = "plant"
    def __getattribute__(self,obj):
        if obj.endswith("e"):
            return object.__getattribute__(self,obj)
        else:
            return self.call_wind()
    def call_wind(self):
        return "树大招风"
aa = Tree("大树")
print(aa.name)#因为 name 是以 e 结尾，所以返回的还是 name，所以打印出"大树"
print(aa.wind)#这个代码中因为 wind 不是以 e 结尾，#所以返回 self.call_wind 的结果，打印的是"树大招风"
```

**上面的解释正确吗？**

先说结果，关于 print(aa.name) 的解释是正确的，但关于 print(aa.wind) 的解释不对，为什么呢？我们来分析一下，执行 aa.wind 时，先调用__getattribute__方法，经过判断后，它返回的是 self.call_wind()，即 self.call_wind 的执行结果，但当去调用 aa 这个对象的 call_wind 属性时，前提是又要去调用__getattribute__方法，反反复复，没完没了，形成了递归调用且没有退出机制，最终程序就挂了！

### 2.11 __ delattr__()

本函数的作用是删除属性，实现了该函数的类可以用 del 命令来删除属性。

```
class MyClass:
    def __init__(self, work, score):
        self.work = work
        self.score = score
    def __delattr__(self, name):
        print("你正在删除一个属性")
        return super().__delattr__(name)

def main():
    test = MyClass(work="math", score=100)
    # 删除 work 属性
    del test.work
    # work 属性删除，score 属性还在
    print(test.score)
    try:
        print(test.work)
    except AttributeError as reason:
        print(reason)

if __name__ == '__main__':
    main()
```

输出结果：

```
你正在删除一个属性
'MyClass' object has no attribute 'work'
```

### 2.12 __ setitem__()

__setitem__(self,key,value)：该方法应该按一定的方式存储和 key 相关的 value。在设置类实例属性时自动调用的。

```
# -*- coding:utf-8 -*-
 
class A:
    def __init__(self):
        self['B']='BB'
        self['D']='DD'
        
    def __setitem__(self,name,value):
 
        print "__setitem__:Set %s Value %s" %(name,value)
        
if __name__=='__main__':
    X=A()
```

输出结果为：

```
__setitem__:Set B Value BB
__setitem__:Set D Value DD
```

### 2.13 __ getitem__()

Python 的特殊方法__getitem_() 主要作用是可以让对象实现迭代功能。我们通过一个实例来说明。

定义一个 Sentence 类，通过索引提取单词。

```
import re
RE_WORD = re.compile(r'\w+')
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)  # re.findall 函数返回一个字符串列表，里面的元素是正则表达式的全部非重叠匹配
    def __getitem__(self, index):
        return self.words[index]
```

**测试：**

```
>>> s = Sentence('The time has come')
>>> for word in s:
            print(word)
    
    The
    time
    has
    come
>>> s[0]
    'The'
>>> s[1]
    'time'
```

通过测试发现，示例 s 可以正常迭代。但是没有定义 **getitem**() 测试则会报错，TypeError: '***' object is not iterable。

**序列可以迭代：**

我们都知道序列是可以迭代，下面具体说明原因。

解释器需要迭代对象 x 时， 会自动调用 iter(x) 方法。内置的 iter(x) 方法有以下作用：

*   检查对象是否实现了__iter__ 方法，如果实现了就调用它（也就是我们偶尔用到的特殊方法重载），获取一个迭代器。
*   如果没有实现 iter() 方法， 但是实现了 __getitem__方法，Python 会创建一个迭代器，尝试按顺序（从索引 0 开始，可以看到我们刚才是通过 s[0] 取值）获取元素。
*   如果尝试失败，Python 抛出 TypeError 异常，通常会提示 TypeError: '***' object is not iterable。

任何 Python 序列都可迭代的原因是，他们都实现了__getitem__方法。其实，标准的序列也都实现了__iter__方法。

**注意：**从 python3.4 开始，检查对象 x 能否迭代，最准确的方法是： 调用 iter(x) 方法，如果不可迭代，在处理 TypeError 异常。这比使用 isinstance(x,abc.Iterable) 更准确，因为 iter() 方法会考虑到遗留的__getitem__() 方法，而 abc.Iterable 类则不考虑。

### 2.14 __ delitem__()

__delitem__(self,key):

这个方法在对对象的组成部分使用__del__语句的时候被调用，应删除与 key 相关联的值。同样，仅当对象可变的时候，才需要实现这个方法。

```
class Tag:
    def __init__(self):
        self.change={'python':'This is python',
                     'php':'PHP is a good language'}
 
    def __getitem__(self, item):
        print('调用 getitem')
        return self.change[item]
 
    def __setitem__(self, key, value):
        print('调用 setitem')
        self.change[key]=value
 
    def __delitem__(self, key):
        print('调用 delitem')
        del self.change[key]
 
a=Tag()
print(a['php'])
del a['php']
print(a.change)
```

输出结果：

```
调用 getitem
PHP is a good language
调用 delitem
{'python': 'This is python'}
```

### 2.15 __ iter__()

迭代器就是重复地做一些事情，可以简单的理解为循环，在 python 中实现了__iter__方法的对象是可迭代的，实现了 next() 方法的对象是迭代器，这样说起来有点拗口，实际上要想让一个迭代器工作，至少要实现__iter__方法和 next 方法。很多时候使用迭代器完成的工作使用列表也可以完成，但是如果有很多值列表就会占用太多的内存，而且使用迭代器也让我们的程序更加通用、优雅、pythonic。

如果一个类想被用于 for ... in 循环，类似 list 或 tuple 那样，就必须实现一个__iter__() 方法，该方法返回一个迭代对象，然后，Python 的 for 循环就会不断调用该迭代对象的 next() 方法拿到循环的下一个值，直到遇到 StopIteration 错误时退出循环。

**容器（container）：**

容器是用来储存元素的一种数据结构，容器将所有数据保存在内存中，Python 中典型的容器有：list，set，dict，str 等等。

```
class test():
    def __init__(self,data=1):
        self.data = data

    def __iter__(self):
        return self
    def __next__(self):
        if self.data > 5:
            raise StopIteration
        else:
            self.data+=1
            return self.data

for item in test(3):
    print(item)
```

输出结果：

for … in… 这个语句其实做了两件事。第一件事是获得一个可迭代器，即调用了__iter__() 函数。 第二件事是循环的过程，循环调用__next__() 函数。

对于 test 这个类来说，它定义了__iter__和__next__函数，所以是一个可迭代的类，也可以说是一个可迭代的对象（Python 中一切皆对象）。

**迭代器：**

含有__next__() 函数的对象都是一个迭代器，所以 test 也可以说是一个迭代器。如果去掉__itet__() 函数，test 这个类也不会报错。如下代码所示：

```
class test():
    def __init__(self,data=1):
        self.data = data

    def __next__(self):
        if self.data > 5:
            raise StopIteration
        else:
            self.data+=1
            return self.data

t = test(3)   
for i in range(3):
    print(t.__next__())
```

输出结果：

生成器

生成器是一种特殊的迭代器。当调用 fib() 函数时，生成器实例化并返回，这时并不会执行任何代码，生成器处于空闲状态，注意这里 prev, curr = 0, 1 并未执行。然后这个生成器被包含在 list() 中，list 会根据传进来的参数生成一个列表，所以它对 fib() 对象 （一切皆对象，函数也是对象） 调用__next()__方法。

```
def fib(end = 1000):
    prev,curr=0,1
    while curr < end:
        yield curr
        prev,curr=curr,curr+prev

print(list(fib()))
```

输出结果：

```
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
```

### 2.16 __ del__()

__del__() 方法，功能正好和 __init__() 相反，其用来销毁实例化对象。

事实上在编写程序时，如果之前创建的类实例化对象后续不再使用，最好在适当位置手动将其销毁，释放其占用的内存空间（整个过程称为垃圾回收（简称 GC））。

> 大多数情况下，Python 开发者不需要手动进行垃圾回收，因为 Python 有自动的垃圾回收机制（下面会讲），能自动将不需要使用的实例对象进行销毁。

无论是手动销毁，还是 Python 自动帮我们销毁，都会调用 __del__() 方法。举个例子：

```
class CLanguage:
    def __init__(self):
        print("调用 __init__() 方法构造对象")
    def __del__(self):
        print("调用__del__() 销毁对象，释放其空间")
clangs = CLanguage()
del clangs
```

程序运行结果为：

```
调用 __init__() 方法构造对象
调用__del__() 销毁对象，释放其空间
```

但是，读者千万不要误认为，只要为该实例对象调用 __del__() 方法，该对象所占用的内存空间就会被释放。举个例子：

```
class CLanguage:
    def __init__(self):
        print("调用 __init__() 方法构造对象")
    def __del__(self):
        print("调用__del__() 销毁对象，释放其空间")
clangs = CLanguage()
#添加一个引用 clangs 对象的实例对象
cl = clangs
del clangs
print("***********")
```

程序运行结果为：

```
调用 __init__() 方法构造对象
***********
调用__del__() 销毁对象，释放其空间
```

> 注意，最后一行输出信息，是程序执行即将结束时调用 __del__() 方法输出的。

可以看到，当程序中有其它变量（比如这里的 cl）引用该实例对象时，即便手动调用 __del__() 方法，该方法也不会立即执行。这和 Python 的垃圾回收机制的实现有关。

Python 采用自动引用计数（简称 ARC）的方式实现垃圾回收机制。该方法的核心思想是：每个 Python 对象都会配置一个 [计数器](https://www.zhihu.com/search?q=计数器&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType":"answer","sourceId":1682758202})，初始 Python 实例对象的计数器值都为 0，如果有变量引用该实例对象，其计数器的值会加 1，依次类推；反之，每当一个变量取消对该实例对象的引用，计数器会减 1。如果一个 Python 对象的的计数器值为 0，则表明没有变量引用该 Python 对象，即证明程序不再需要它，此时 Python 就会自动调用 __del__() 方法将其回收。

以上面程序中的 clangs 为例，实际上构建 clangs 实例对象的过程分为 2 步，先使用 CLanguage() 调用该类中的 __init__() 方法构造出一个该类的对象（将其称为 C，计数器为 0），并立即用 clangs 这个变量作为所建实例对象的引用（ C 的计数器值 + 1）。在此基础上，又有一个 clang 变量引用 clangs（其实相当于引用 CLanguage()，此时 C 的计数器再 +1 ），这时如果调用 del clangs 语句，只会导致 C 的计数器减 1（值变为 1），因为 C 的计数器值不为 0，因此 C 不会被销毁（不会执行 __del__() 方法）。

如果在上面程序结尾，添加如下语句：

```
del cl
print("-----------")
```

则程序的执行结果为：

```
调用 __init__() 方法构造对象
***********
调用__del__() 销毁对象，释放其空间
-----------
```

可以看到，当执行 del cl 语句时，其应用的对象实例对象 C 的计数器继续 -1（变为 0），对于计数器为 0 的实例对象，Python 会自动将其视为垃圾进行回收。

需要额外说明的是，如果我们重写子类的 __del__() 方法（父类为非 object 的类），则必须显式调用父类的 __del__() 方法，这样才能保证在回收子类对象时，其占用的资源（可能包含继承自父类的部分资源）能被彻底释放。为了说明这一点，这里举一个反例：

```
class CLanguage:
    def __del__(self):
        print("调用父类 __del__() 方法")
class cl(CLanguage):
    def __del__(self):
        print("调用子类 __del__() 方法")
c = cl()
del c
```

程序运行结果为：

```
调用子类 __del__() 方法
```

### 2.17 __dir__(）

dir() 函数，通过此函数可以某个对象拥有的所有的属性名和方法名，该函数会返回一个包含有所有属性名和方法名的有序列表。

举个例子：

```
class CLanguage:
    def __init__ (self,):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
    def say():
        pass
clangs = CLanguage()
print(dir(clangs))
```

程序运行结果为：

```
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'add', 'name', 'say']
```

> 注意，通过 dir() 函数，不仅仅输出本类中新添加的属性名和方法（最后 3 个），还会输出从父类（这里为 object 类）继承得到的属性名和方法名。

值得一提的是，dir() 函数的内部实现，其实是在调用参数对象 __dir__() 方法的基础上，对该方法返回的属性名和方法名做了排序。

所以，除了使用 dir() 函数，我们完全可以自行调用该对象具有的 __dir__() 方法：

```
class CLanguage:
    def __init__ (self,):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
    def say():
        pass
clangs = CLanguage()
print(clangs.__dir__())
```

程序运行结果为：

```
['name', 'add', '__module__', '__init__', 'say', '__dict__', '__weakref__', '__doc__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']
```

显然，使用 __dir__() 方法和 dir() 函数输出的数据是相同，仅仅顺序不同。

### 2.18 __dict__()

在 Python 类的内部，无论是类属性还是实例属性，都是以字典的形式进行存储的，其中属性名作为键，而值作为该键对应的值。

为了方便用户查看类中包含哪些属性，Python 类提供了__dict__ 属性。需要注意的一点是，该属性可以用类名或者类的实例对象来调用，用类名直接调用 __dict__，会输出该由类中所有类属性组成的字典；而使用类的实例对象调用 __dict__，会输出由类中所有实例属性组成的字典。

举个例子：

```
class CLanguage:
    a = 1
    b = 2
    def __init__ (self):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
#通过类名调用__dict__
print(CLanguage.__dict__)
#通过类实例对象调用 __dict__
clangs = CLanguage()
print(clangs.__dict__)
```

程序输出结果为：

```
{'__module__': '__main__', 'a': 1, 'b': 2, '__init__': <function CLanguage.__init__ at 0x0000022C69833E18>, '__dict__': <attribute '__dict__' of 'CLanguage' objects>, '__weakref__': <attribute '__weakref__' of 'CLanguage' objects>, '__doc__': None}
{'name': 'C 语言中文网', 'add': 'http://c.biancheng.net'}
```

不仅如此，对于具有继承关系的父类和子类来说，父类有自己的 __dict__，同样子类也有自己的 __dict__，它不会包含父类的 __dict__。例如：

```
class CLanguage:
    a = 1
    b = 2
    def __init__ (self):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
       
class CL(CLanguage):
    c = 1
    d = 2
    def __init__ (self):
        self.na = "Python 教程"
        self.ad = "http://c.biancheng.net/python"
#父类名调用__dict__
print(CLanguage.__dict__)
#子类名调用__dict__
print(CL.__dict__)
#父类实例对象调用 __dict__
clangs = CLanguage()
print(clangs.__dict__)
#子类实例对象调用 __dict__
cl = CL()
print(cl.__dict__)
```

运行结果为：

```
{'__module__': '__main__', 'a': 1, 'b': 2, '__init__': <function CLanguage.__init__ at 0x000001721A853E18>, '__dict__': <attribute '__dict__' of 'CLanguage' objects>, '__weakref__': <attribute '__weakref__' of 'CLanguage' objects>, '__doc__': None}
{'__module__': '__main__', 'c': 1, 'd': 2, '__init__': <function CL.__init__ at 0x000001721CD15510>, '__doc__': None}
{'name': 'C 语言中文网', 'add': 'http://c.biancheng.net'}
{'na': 'Python 教程', 'ad': 'http://c.biancheng.net/python'}
```

显然，通过子类直接调用的 __dict__ 中，并没有包含父类中的 a 和 b 类属性；同样，通过子类对象调用的 __dict__，也没有包含父类对象拥有的 name 和 add 实例属性。

除此之外，借助由类实例对象调用 __dict__ 属性获取的字典，可以使用字典的方式对其中实例属性的值进行修改，例如：

```
class CLanguage:
    a = "aaa"
    b = 2
    def __init__ (self):
        self.name = "C 语言中文网"
        self.add = "http://c.biancheng.net"
#通过类实例对象调用 __dict__
clangs = CLanguage()
print(clangs.__dict__)
clangs.__dict__['name'] = "Python 教程"
print(clangs.name)
```

程序运行结果为：

```
{'name': 'C 语言中文网', 'add': 'http://c.biancheng.net'}
Python 教程
```

> 注意，无法通过类似的方式修改类变量的值。

### 2.19 __exit__，__enter__

__exit__和__enter__函数是与 with 语句的组合应用的，用于上下文管理。

**1.__enter(self)__：**

负责返回一个值，该返回值将赋值给 as 子句后面的 var_name，通常返回对象自己，即 “self”。函数优先于 with 后面的“代码块”(statements1,statements2,……) 被执行。

**2.__exit__(self, exc_type, exc_val, exc_tb)：**

```
with xxx as var_name：

    # 代码块开始

    statements1

    statements2

    ……

    # 代码块结束

# 代码快后面的语句
statements after code block
```

执行完 with 后面的代码块后自动调用该函数。with 语句后面的 “代码块” 中有异常 （不包括因调用某函数，由被调用函数内部抛出的异常） ，会把异常类型，异常值，异常跟踪信息分别赋值给函数参数 exc_type, exc_val, exc_tb，没有异常的情况下，exc_type, exc_val, exc_tb 值都为 None。另外，如果该函数返回 True、1 类值的 Boolean 真值，那么将忽略“代码块” 中的异常，停止执行 “代码块” 中剩余语句，但是会继续执行 “代码块” 后面的语句；如果函数返回类似 0，False 类的 Boolean 假值、或者没返回值，将抛出 “代码块” 中的异常，那么在没有捕获异常的情况下，中断 “代码块” 及“代码块”之后语句的执行。

**代码：**

```
class User(object):
    def __init__(self, username, password):

        self._username = username
        self._password = password

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, username):
        self._username = username

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, password):
        self._password = password

    def __enter__(self):
        print('before：auto do something before statements body of with executed')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('after：auto do something after statements body of with executed')

if __name__ == '__main__':
    boy = User('faker', 'faker2021')
    print(boy.password)
    print("上下文管理器 with 语句：")
    with User('faker', 'faker2021') as user:
        print(user.password)
    print('---------end-----------')
```

输出结果：

```
faker2021
上下文管理器 with 语句：
before：auto do something before statements body of with executed
faker2021
after：auto do something after statements body of with executed
---------end-----------
```

**更改上述部分代码如下，继续运行：**

```
class User(object):
    def __init__(self, username, password):

        self._username = username
        self._password = password

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, username):
        self._username = username

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, password):
        self._password = password

    def __enter__(self):
        print('before：auto do something before statements body of with executed')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('auto do something after statements body of with executed')

        print('exc_type:', exc_type)
        print('exc_val:', exc_val)
        print('exc_tb:', exc_tb)
        return False

if __name__ == '__main__':
    boy = User('faker', 'faker2021')
    print(boy.password)
    print("上下文管理器 with 语句：")
    with User('faker', 'faker2021') as user:
        print(user.password)
        12/0
        print('after execption')
    print('---------end-----------')
```

输出结果：

```
Traceback (most recent call last):
faker2021
  File "/code/0.py", line 42, in <module>
上下文管理器 with 语句：
    12/0
before：auto do something before statements body of with executed
ZeroDivisionError: division by zero
faker2021
auto do something after statements body of with executed
exc_type: <class 'ZeroDivisionError'>
exc_val: division by zero
exc_tb: <traceback object at 0x0000015F4A62AD48>
```

**在上述的基础上继续修改代码，将__exit__的返回值设置为 True：**

```
def __exit__(self, exc_type, exc_val, exc_tb):
    print('auto do something after statements body of with executed')

    print('exc_type:', exc_type)
    print('exc_val:', exc_val)
    print('exc_tb:', exc_tb)
    return True
```

输出结果：

```
faker2021
上下文管理器 with 语句：
before：auto do something before statements body of with executed
faker2021
auto do something after statements body of with executed
exc_type: <class 'ZeroDivisionError'>
exc_val: division by zero
exc_tb: <traceback object at 0x0000021DBDD3AD48>
---------end-----------
```

注意：

1、抛异常后，代码块中剩余的语句没有再继续运行

2、如果在上述的基础上，把代码中的 12/0 剪切后放到 password(self) 中 ，抛出异常的异常信息也会传递给__exit__函数的

```
@property
def password(self):
    12/0
    return self._password

if __name__ == '__main__':

    print("上下文管理器 with 语句：")
    with User('faker', 'faker2021') as user:
        print(user.password)
    print('---------end-----------')
```

输出结果：

```
上下文管理器 with 语句：
before：auto do something before statements body of with executed
auto do something after statements body of with executed
exc_type: <class 'ZeroDivisionError'>
exc_val: division by zero
exc_tb: <traceback object at 0x000001614FFFAF88>
---------end-----------
```

### __init__方法

`__init__`方法负责对象的初始化，系统执行该方法前，其实该对象已经存在了，要不然初始化什么东西呢？先看例子：

```
# class A(object): 
class A:
    def __init__(self):
        print("__init__ ")
        super(A, self).__init__()

    def __new__(cls):
        print("__new__ ")
        return super(A, cls).__new__(cls)

    def __call__(self):  # 可以定义任意参数
        print('__call__ ')

A()
```

输出

```
__new__
__init__
```

从输出结果来看， `__new__`方法先被调用，返回一个实例对象，接着 `__init__` 被调用。 `__call__`方法并没有被调用，这个我们放到最后说，先来说说前面两个方法，稍微改写成：

```
def __init__(self):
    print("__init__ ")
    print(self)
    super(A, self).__init__()

def __new__(cls):
    print("__new__ ")
    self = super(A, cls).__new__(cls)
    print(self)
    return self
```

输出：

```
__new__ 
<__main__.A object at 0x1007a95f8>
__init__ 
<__main__.A object at 0x1007a95f8>
```

从输出结果来看，`__new__` 方法的返回值就是类的实例对象，这个实例对象会传递给 `__init__` 方法中定义的 self 参数，以便实例对象可以被正确地初始化。

如果 `__new__` 方法不返回值（或者说返回 None）那么 `__init__` 将不会得到调用，这个也说得通，因为实例对象都没创建出来，调用 init 也没什么意义，此外，Python 还规定，`__init__` 只能返回 None 值，否则报错，这个留给大家去试。

`__init__`方法可以用来做一些初始化工作，比如给实例对象的状态进行初始化：

```
def __init__(self, a, b):
    self.a = a
    self.b = b
    super(A, self).__init__()
```

另外，`__init__`方法中除了 self 之外定义的参数，都将与`__new__`方法中除 cls 参数之外的参数是必须保持一致或者等效。

```
class B:
    def __init__(self, *args, **kwargs):
        print("init", args, kwargs)

    def __new__(cls, *args, **kwargs):
        print("new", args, kwargs)
        return super().__new__(cls)

B(1, 2, 3)

# 输出

new (1, 2, 3) {}
init (1, 2, 3) {}
```

### __new__ 方法

一般我们不会去重写该方法，除非你确切知道怎么做，什么时候你会去关心它呢，它作为构造函数用于创建对象，是一个工厂函数，专用于生产实例对象。著名的设计模式之一，[单例模式](https://www.zhihu.com/search?q=单例模式&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType":"answer","sourceId":1842466508})，就可以通过此方法来实现。在自己写框架级的代码时，可能你会用到它，我们也可以从开源代码中找到它的应用场景，例如微型 Web 框架 Bootle 就用到了。

```
class BaseController(object):
    _singleton = None
    def __new__(cls, *a, **k):
        if not cls._singleton:
            cls._singleton = object.__new__(cls, *a, **k)
        return cls._singleton
```

这段代码出自 [https://github.com/bottlepy/bottle/blob/release-0.6/bottle.py](https://github.com/bottlepy/bottle/blob/release-0.6/bottle.py)

这就是通过 `__new__` 方法是实现单例模式的的一种方式，如果实例对象存在了就直接返回该实例即可，如果还没有，那么就先创建一个实例，再返回。当然，实现单例模式的方法不只一种，Python 之禅有说：

> There should be one-- and preferably only one --obvious way to do it.  
> 用一种方法，最好是只有一种方法来做一件事

### __call__ 方法

关于 `__call__` 方法，不得不先提到一个概念，就是_可调用对象（callable）_，我们平时自定义的函数、内置函数和类都属于可调用对象，但凡是可以把一对括号`()`应用到某个对象身上都可称之为可调用对象，判断对象是否为可调用对象可以用函数 `callable`

如果在类中实现了 `__call__` 方法，那么实例对象也将成为一个可调用对象，我们回到最开始的那个例子：

```
a = A()
print(callable(a))  # True
```

`a`是实例对象，同时还是可调用对象，那么我就可以像函数一样调用它。试试：

```
a()  # __call__
```

很神奇不是，实例对象也可以像函数一样作为可调用对象来用，那么，这个特点在什么场景用得上呢？这个要结合类的特性来说，类可以记录数据（属性），而函数不行（闭包某种意义上也可行），利用这种特性可以实现基于类的装饰器，在类里面记录状态，比如，下面这个例子用于记录函数被调用的次数：

```
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@Counter
def foo():
    pass

for i in range(10):
    foo()

print(foo.count)  # 10
```

在 Bottle 中也有 call 方法 的使用案例，另外，[stackoverflow](https://stackoverflow.com/questions/5824881/python-call-special-method-practical-example) 也有一些关于 call 的实践例子，推荐看看，如果你的项目中，需要更加抽象化、框架代码，那么这些高级特性往往能发挥出它作用。

欢迎关注我，编程路上多个朋友

@刘志军！[](https://pic3.zhimg.com/v2-793ddf17d6c7888ed8e2fafea9fd3cc1_xs.jpg?source=1940ef5c) 追远 ·J

好问题！

作为典型的**面向对象**的语言，Python 中 **类** 的**定义**和**使用**是不可或缺的一部分知识。对于有面向对象的经验、对**类**和**实例**的概念已经足够清晰的人，学习 Python 的这套定义规则不过是语法的迁移。但对新手小白而言，要想相对快速地跨过`__init__`这道坎，还是结合一个简单例子来说比较好。

以创建一个 “学生” **类**为例，最简单的语句是

```
class Student():
    pass
```

当然，这样定义的类没有包含任何预定义的数据和功能。除了名字叫 Student 以外，它没有体现出任何 “学生” 应该具有的特点。但它是可用的，上述代码运行过后，通过类似

```
stu_1 = Student()
```

这样的语句，我们可以创建一个 “学生” **实例**，即一个具体的 “学生” 对象。

通过`class`语句定义的类`Student`，就好像一个 **“模具”**，它可以定义作为一个学生应该具有的各种特点（这里暂未具体定义）；

而在类名`Student`后加圆括号`()`，组成一个**类似函数调用**的形式`Student()`，则执行了一个叫做**实例化**的过程，即根据定义好的规则，创建一个包含具体数据的学生对象（实例）。

为了使用创建的学生实例`stu_1`，我们可以继续为它添加或修改属性，比如添加一组成绩`scores` ，由三个整数组成：

```
stu_1.scores = [80, 90, 85]
```

但这样明显存在很多问题，一旦我们需要处理很多学生实例，比如`stu_2`, `stu_3`, `...`，这样不但带来书写上的麻烦，还容易带来错误，万一某些地方`scores`打错了，或者干脆忘记了，相应的学生实例就会缺少正确的`scores`属性。更重要的是，**这样的`scores`属性是暴露出来的，它的使用完全被外面控制着，没有起到 “封装” 的效果，既不方便也不靠谱**。

一个自然的解决方案是允许我们在执行实例化过程`Student()`时**传入一些参数**，以方便且正确地初始化 / 设置一些属性值，那么如何定义这种初始化行为呢？答案就是在类内部定义一个`__init__`函数。这时，`Student`的定义将变成（我们先用一段注释占着`__init__`函数内的位置）。

```
class Student():
    def __init__(self, score1, score2, score3):
        # 相关初始化语句
```

定义`__init__`后，执行实例化的过程须变成`Student(arg1, arg2, arg3)`，**新建的实例本身，连带其中的参数，会一并传给`__init__`函数自动并执行它**。所以**`__init__`函数的参数列表会在开头多出一项，它永远指代新建的那个实例对象**，Python 语法要求这个参数**必须要有**，而名称随意，习惯上就命为`self`。

新建的实例传给`self`后，就可以在`__init__`函数内创建并初始化它的属性了，比如之前的`scores`，就可以写为

```
class Student():
    def __init__(self, score1, score2, score3):
        self.scores = [score1, score2, score3]
```

此时，若再要创建拥有具体成绩的学生实例，就只需

```
stu_1 = Student(80, 90, 85)
```

此时，`stu_1`将已经具有设置好的`scores`属性。并且由于`__init__`规定了实例化时的参数，若传入的参数数目不正确，解释器可以报错提醒。你也可以在其内部添加必要的参数检查，以避免错误或不合理的参数传递。

> 在其他方面，`__init__`就与普通函数无异了。考虑到新手可能对 “函数” 也掌握得很模糊，这里特别指出几个 “无异” 之处：  
> **独立的命名空间**，也就是说**函数内新引入的变量均为局部变量**，新建的实例对象对这个函数来说也只是通过第一参数 self 从外部传入的，故无论设置还是使用它的属性都得利用`self.<属性名>`。如果将上面的初始化语句写成  
> `scores = [score1, score2, score3]`（少了`self.`），  
> 则只是在函数内部创建了一个 scores 变量，它在函数执行完就会消失，对新建的实例没有任何影响；  
> 与此对应，**`self`的属性名和函数内其他名称 （包括参数） 也是不冲突的**，所以你可能经常见到类似这种写法，它正确而且规范。

```
class Student():
    def __init__(self, name, scores):
        # 这里增加了属性 name，并将所有成绩作为一个参数 scores 传入
        # self.name 是 self 的属性，单独的 name 是函数内的局部变量，参数也是局部变量
        self.name = name
        if len(scores) == 3:
            self.scores = scores
        else:
            self.scores = [0] * 3
```

> 从第二参数开始均可设置**变长参数**、**[默认值](https://www.zhihu.com/search?q=默认值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType":"answer","sourceId":767530541}) **等，相应地将允许实例化过程`Student()`中灵活地传入需要数量的参数；  
> 其他……

说到最后，`__init__`还是有个特殊之处，那就是它**不允许有返回值**。如果你的`__init__`过于复杂有可能要提前结束的话，使用**单独的`return`**就好，不要带返回值。

上面代码的执行结果如下

![](https://pic2.zhimg.com/v2-5cd95f173d6f452d90906dac5487337a_r.jpg?source=1940ef5c)

 ![](https://pic2.zhimg.com/536cac09630badb94548a191041cb76c_xs.jpg?source=1940ef5c) cloudream​

刚好在写 Python 的总结写到这块，强行答一波，本人也是 Python 小白：  
以下内容为节选：

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

class 关键字后面跟类的名称就定义了一个类，类的名称可以任意，采用驼峰命名法，也即每个单词的首字母大写，如 Book、Person、WildAnimal 等

这里的`__init__`方法是一个特殊的方法（init 是单词初始化 initialization 的省略形式），在使用类创建对象之后被执行，用于给新创建的对象初始化属性用。  
初始化属性的语句就是`self.name = name`这种了，这一句不太好理解，我们把它改编一下就好理解了：

```
def __init__(self, n, a):
    self.name = n
    self.age = a
```

首先这是一个方法，方法的形参有 self，n 和 a 三个。  
这个 self，表示对象本身，**`谁调用，就表示谁`**（这句话不好理解，先记住，我们后面分析）。  
语法上，类中的方法的第一个参数都是 self，这是和普通方法区别的地方。  
这里`self.name = n`和`self.age = a`表示将外部传来的 n 和 a，赋值给了 self 对象的 name 和 age 属性。  
这里的 n 和 a，其实叫什么都可以，但是会有个问题：一般我们调用方法的时候，想自动提示一下或者查看文档看一下这个方法的参数要求，如果形参名都是 n、a、m、i 这些，会让人摸不着头脑，不知道到底该传入什么样的 [实参](https://www.zhihu.com/search?q=实参&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType":"answer","sourceId":293788116})。因为这里我们传入实参是为了给属性赋值，为了能见名知意，将形参的名字定义的跟属性一致，调用者就知道该怎么传参了。  
所以才有了上面的写法。

再来说创建对象：

```
p = Person('小明', 20)
```

这句话就创建出来了一个具体的人，并且给这个人起了个名字叫小明，指定了小明的年龄为 20，并且将小明这个对象赋值给了变量 p，此时 p 就表示小明这个人（对象）

这就造出了一个人。

既然你是神，当然想造出什么样的人都可以，比如造出一个 200 岁的叫杰拉考的人：

```
p = Person('杰拉考', 200)
```

这句话后面的`Person('杰拉考', 200)`用于创建出了一个对象（人），并且调用了 **init**(self,name,age) 方法完成了该人的属性的初始化，`杰拉考`赋值给了`name`，`200`赋值给了`age`属性。  
那`self`呢？self 不需要传参，上面我们说过，self，表示对象本身，`**谁调用，就表示谁**`，此时的 self 就表示你`Person('杰拉考', 200)`创造出来的那个对象，也即是`p`。  
也即，我们创造出了 p，然后给 p 的属性赋了值，此时 p 就表示拥有属性值之后的那个人。

可以使用点`.`来调用对象的属性，比如输出 p 的名字和年龄，完整代码为：

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
p = Person('杰拉考', 200)
print(p.name)
print(p.age)
```

输出结果：

```
杰拉考
```

接下来我们再在 Person 类中定义一个方法，用于自我介绍：

```
def desc(self):
    print("我叫%s，今年%d 岁" % (self.name, self.age))
```

在类的内部，访问自己的属性和方法，都需要通过 self，self 就是**外部对象在类内部的表示**，此时可以使用 p 调用该方法，完整代码如下：

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def desc(self):
        print("我叫%s，今年%d 岁" % (self.name, self.age))
p = Person('杰拉考', 200)
# 调用自我介绍方法 desc 方法中的 self 就是外部的这个 p
p.desc()
```

输出为：

```
我叫杰拉考，今年 200 岁
```

当前 desc 方法中的 self，就是外部的那个对象 p，如果我再定义了一个对象 p2，那么 p2 调用 desc 时，desc 中的 self 就表示 p2 这个对象。正所谓：**`谁调用，就表示谁`**![](https://pica.zhimg.com/v2-9745ec5ff38dbaed03a21d56f22c647f_xs.jpg?source=1940ef5c) 天天天天白开水

强行装个吧：定义类的时候，若是添加__init__方法，那么在创建类的实例的时候，实例会自动调用这个方法，一般用来对实例的属性进行初使化。比如：

class testClass:

def __init__(self, name, gender): // 定义 __init__方法，这里有三个参数，这个 self 指的是一会创建类的实例的时候这个被创建的实例本身（例中的 testman），你也可以写成其他的东西，比如写成 me 也是可以的，这样的话下面的 self.Name 就要写成 me.Name。

self.Name=name // 通常会写成 self.name=name，这里为了区分前后两个是不同的东东，把前面那个大写了，等号左边的那个 Name（或 name）是实例的属性，后面那个是方法__init__的参数，两个是不同的）

self.Gender=gender // 通常会写成 self.gender=gender

print('hello') // 这个 print('hello') 是为了说明在创建类的实例的时候，__init__方法就立马被调用了。

testman = testClass('neo,'male') // 这里创建了类 testClass 的一个实例 testman, 类中有__init__这个方法，在创建类的实例的时候，就必须要有和方法__init__匹配的参数了，由于 self 指的就是创建的实例本身，self 是不用传入的，所以这里传入两个参数。这条语句一出来，实例 testman 的两个属性 Name，Gender 就被赋值初使化了，其中 Name 是 neo，Gender 是 male。

看图的运行结果。我也刚学，大学接触过编程，献丑了，就匿名了。

![](https://pica.zhimg.com/ffcef790480f7f3475cd4ba90043017f_r.jpg?source=1940ef5c)