# C++基础

## 一、多态

C++ 的多态性用一句话概括就是：在基类的函数前加上 virtual 关键字，在派生类中重写该函数，运行时将会根据对象的实际类型来调用相应的函数。如果对象类型是派生类，就调用派生类的函数；如果对象类型是基类，就调用基类的函数。

1. 用 virtual 关键字申明的函数叫做虚函数，虚函数肯定是类的成员函数。  
2. 存在虚函数的类都有一个一维的虚函数表叫做虚表，类的对象有一个指向虚表开始的虚指针。虚表是和类对应的，虚表指针是和对象对应的。  
3. 多态性是一个接口多种实现，是面向对象的核心，分为类的多态性和函数的多态性。  
4. 多态用虚函数来实现，结合动态绑定。
5. 纯虚函数是虚函数再加上 = 0；  
6. 抽象类是指包括至少一个纯虚函数的类。

纯虚函数：`virtual void fun()=0` 即抽象类必须在子类实现这个函数，即先有名称，没有内容，在派生类实现内容。

我们先看个例子：

```C++
#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
using namespace std;

class Father
{
public:
    void Face()
    {
        cout << "Father's face" << endl;
    }

    void Say()
    {
        cout << "Father say hello" << endl;
    }
};

class Son:public Father
{
public:
    void Say()
    {
        cout << "Son say hello" << endl;
    }
};

void main()
{
    Son son;
    Father *pFather=&son; // 隐式类型转换
    pFather->Say();
}

## 输出
Father say hello
```

在 `main()` 函数中首先定义了一个 `Son` 类的对象 `son`，接着定义了一个指向 Father 类的指针变量 pFather，然后利用该变量调用 `pFather->Say()`。很多人往往将这种情况和 C++ 的多态性搞混淆，认为 `son` 实际上是 `Son` 类的对象，应该是调用 `Son` 类的 Say，输出"Son say hello", 然而结果却不是。

- 从编译的角度来看：

C++ 编译器在编译的时候，要确定每个对象调用的函数（非虚函数）的地址，这称为早期绑定，当我们将 `Son` 类的对象 `son` 的地址赋给 `pFather` 时，C++编译器进行了类型转换，此时 C++ 编译器认为变量 `pFather` 保存的就是 Father 对象的地址，当在 main 函数中执行 pFather->Say(), 调用的当然就是 Father 对象的 Say 函数

从内存角度看：

<div align=center><img src="/assets/C++-2022-04-18-22-04-42.png" alt="C++-2022-04-18-22-04-42" style="zoom:100%;" /></div>

我们构造 `Son` 类的对象时，首先要调用 Father 类的构造函数去构造 Father 类的对象，然后才调用 `Son` 类的构造函数完成自身部分的构造，从而拼接出一个完整的 `Son` 类对象。当我们将 `Son` 类对象转换为 Father 类型时，该对象就被认为是原对象整个内存模型的上半部分，也就是上图中“Father 的对象所占内存”，那么当我们利用类型转换后的对象指针去调用它的方法时，当然也就是调用它所在的内存中的方法，因此，输出“Father Say hello”，也就顺理成章了。

正如很多人那么认为，在上面的代码中，我们知道 `pFather` 实际上指向的是 `Son` 类的对象，我们希望输出的结果是 `son` 类的 Say 方法，那么想到达到这种结果，就要用到虚函数了。

前面输出的结果是因为编译器在编译的时候，就已经确定了对象调用的函数的地址，要解决这个问题就要使用晚绑定，当编译器使用晚绑定时候，就会在运行时再去确定对象的类型以及正确的调用函数，而要让编译器采用晚绑定，就要在基类中声明函数时使用 virtual 关键字，这样的函数我们就称之为虚函数，一旦某个函数在基类中声明为 virtual，那么在所有的派生类中该函数都是 virtual，而不需要再显式地声明为 virtual。

代码稍微改动一下，看一下运行结果

```C++
#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
using namespace std;

class Father
{
public:
    void Face()
    {
        cout << "Father's face" << endl;
    }

    virtual void Say()
    {
        cout << "Father say hello" << endl;
    }
};

class Son:public Father
{
public:
    void Say()
    {
        cout << "Son say hello" << endl;
    }
};

void main()
{
    Son son;
    Father *pFather=&son; // 隐式类型转换
    pFather->Say();
}

## 输出
Son say hello
```

编译器在编译的时候，发现 Father 类中有虚函数，此时编译器会为每个包含虚函数的类创建一个虚表（即 vtable)，该表是一个一维数组，在这个数组中存放每个虚函数的地址。

<div align=center><img src="/assets/C++-2022-04-18-22-06-28.png" alt="C++-2022-04-18-22-06-28" style="zoom:100%;" /></div>

那么如何定位虚表呢？编译器另外还为每个对象提供了一个虚表指针（即 `vptr`)，这个指针指向了对象所属类的虚表，在程序运行时，根据对象的类型去初始化 `vptr`，从而让 `vptr` 正确的指向了所属类的虚表，从而在调用虚函数的时候，能够找到正确的函数，对于第二段代码程序，由于 `pFather` 实际指向的对象类型是 `Son`，因此 `vptr` 指向的 `Son` 类的 `vtable`，当调用 `pFather->Son()` 时，根据虚表中的函数地址找到的就是 `Son` 类的 `Say()` 函数。

正是由于每个对象调用的虚函数都是通过虚表指针来索引的，也就决定了虚表指针的正确初始化是非常重要的，换句话说，在虚表指针没有正确初始化之前，我们不能够去调用虚函数，那么虚表指针是在什么时候，或者什么地方初始化呢？

答案是在构造函数中进行虚表的创建和虚表指针的初始化，在构造子类对象时，要先调用父类的构造函数，此时编译器只“看到了”父类，并不知道后面是否还有继承者，它初始化父类对象的虚表指针，该虚表指针指向父类的虚表，当执行子类的构造函数时，子类对象的虚表指针被初始化，指向自身的虚表。

### 总结（基类有虚函数的）

1. 每一个类都有虚表
2. 虚表可以继承，如果子类没有重写虚函数，那么子类虚表中仍然会有该函数的地址，只不过这个地址指向的是基类的虚函数实现，如果基类有 3 个虚函数，那么基类的虚表中就有三项（虚函数地址），派生类也会虚表，至少有三项，如果重写了相应的虚函数，那么虚表中的地址就会改变，指向自身的虚函数实现，如果派生类有自己的虚函数，那么虚表中就会添加该项。
3. 派生类的虚表中虚地址的排列顺序和基类的虚表中虚函数地址排列顺序相同。

这就是 C++ 中的多态性，当 C++ 编译器在编译的时候，发现 `Father` 类的 `Say()` 函数是虚函数，这个时候 C++ 就会采用晚绑定技术，也就是编译时并不确定具体调用的函数，而是在运行时，依据对象的类型来确认调用的是哪一个函数，这种能力就叫做 `C++` 的多态性，我们没有在 `Say()` 函数前加 `virtual` 关键字时，C++ 编译器就确定了哪个函数被调用，这叫做早期绑定。

C++ 的多态性就是通过晚绑定技术来实现的。

C++ 的多态性用一句话概括就是：在基类的函数前加上 virtual 关键字，在派生类中重写该函数，运行时将会根据对象的实际类型来调用相应的函数，如果对象类型是派生类，就调用派生类的函数，如果对象类型是基类，就调用基类的函数。

虚函数是在基类中定义的，目的是不确定它的派生类的具体行为，例如：

定义一个基类：`class Animal //动物`，它的函数为 `breathe()`

再定义一个类 `class Fish //鱼`。它的函数也为 `breathe()`

再定义一个类 `class Sheep //羊`，它的函数也为 `breathe()`

将 Fish，Sheep 定义成 Animal 的派生类，然而 Fish 与 Sheep 的 breathe 不一样，一个是在水中通过水来呼吸，一个是直接呼吸，所以基类不能确定该如何定义 breathe，所以在基类中只定义了一个 virtual breathe，它是一个空的虚函数，具体的函数在子类中分别定义，程序一般运行时，找到类，如果它有基类，再找到它的基类，最后运行的是基类中的函数，这时，它在基类中找到的是 virtual 标识的函数，它就会再回到子类中找同名函数，派生类也叫子类，基类也叫父类，这就是虚函数的产生，和类的多态性的体现。

这里的多态性是指类的多态性。

函数的多态性是指一个函数被定义成多个不同参数的函数。当你调用这个函数时，就会调用不同的同名函数。

一般情况下（不涉及虚函数），当我们用一个指针/引用调用一个函数的时候，被调用的函数是取决于这个指针/引用的类型。

当设计到多态性的时候，采用了虚函数和动态绑定，此时的调用就不会在编译时候确定而是在运行时确定。不在单独考虑指针/引用的类型而是看指针/引用的对象的类型来判断函数的调用，根据对象中虚指针指向的虚表中的函数的地址来确定调用哪个函数。

现在我们看一个体现 C++ 多态性的例子，看看输出结果：

```C++
#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
using namespace std;

class CA
{
public:
    void f()
    {
        cout << "CA f()" << endl;
    }
    virtual void ff()
    {
        cout << "CA ff()" << endl;
        f();
    }
};

class CB : public CA
{
public :
    virtual void f()
    {
        cout << "CB f()" << endl;
    }
    void ff()
    {
        cout << "CB ff()" << endl;
        f();
        CA::ff();
    }
};
class CC : public CB
{
public:
    virtual void f()
    {
        cout << "C f()" << endl;
    }
};

int main()
{
    CB b;
    CA *ap = &b;
    CC c;
    CB &br = c;
    CB *bp = &c;

    ap->f();
    cout << endl;

    b.f();
    cout << endl;

    br.f();
    cout << endl;

    bp->f();
    cout << endl;

    ap->ff();
    cout << endl;

    bp->ff();
    cout << endl;

    return 0;
}

## 输出
CA f()

CB f()

C f()

C f()

CB ff()
CB f()
CA ff()
CA f()

CB ff()
C f()
CA ff()
CA f()
```

## 二、重载与重写

### 1. 重载（overload）

指函数名相同，但是它的参数表列个数或顺序，类型不同。但是不能靠返回类型来判断。

1. 相同的范围（在同一个作用域中）；
2. 函数名字相同；
3. 参数不同；
4. virtual 关键字可有可无。
5. 返回值可以不同；

### 2. 重写（也称为覆盖 override）

是指派生类重新定义基类的虚函数，特征是：

1. 不在同一个作用域（分别位于派生类与基类）；
2. 函数名字相同；
3. 参数相同；
4. 基类函数必须有 virtual 关键字，不能有 Static 。
5. 返回值相同（或是协变），否则报错；<—-协变这个概念我也是第一次才知道…
6. 重写函数的访问修饰符可以不同。尽管 virtual 是 private 的，派生类中重写改写为 public,protected 也是可以的。

从实现原理上来说：

- 重载：编译器根据函数不同的参数表，对同名函数的名称做修饰，然后这些同名函数就成了不同的函数（至少对于编译器来说是这样的）。如，有两个同名函数：`function func(p:integer):integer;` 和 `function func(p:string):integer;`。那么编译器做过修饰后的函数名称可能是这样的：`int_func`、`str_func`。对于这两个函数的调用，在编译器间就已经确定了，是静态的。也就是说，它们的地址在编译期就绑定了（早绑定），因此，重载和多态无关！
- 重写：和多态真正相关。当子类重新定义了父类的虚函数后，父类指针根据赋给它的不同的子类指针，动态的调用属于子类的该函数，这样的函数调用在编译期间是无法确定的（调用的子类的虚函数的地址无法给出）。因此，这样的函数地址是在运行期绑定的（晚绑定）。

### 3. 重定义（也称隐藏）

1. 不在同一个作用域（分别位于派生类与基类）；
2. 函数名字相同；
3. 返回值可以不同；
4. 参数不同。此时，不论有无 virtual 关键字，基类的函数将被隐藏（注意别与重载以及覆盖混淆）。
5. 参数相同，但是基类函数没有 virtual 关键字。此时，基类的函数被隐藏（注意别与覆盖混淆）。

## 三、虚函数与纯虚函数
纯虚函数就是比如前面声明 virtual 然后后面 =0，就是纯虚函数，纯虚函数的话，子类继承的时候是必须要重新实现的，而虚函数的话没这个规定。

### 不能作为虚函数的

1. 非类的成员函数，即普通函数
    它们没有继承性，即便声明为虚函数，也毫无意义。
2. 构造函数
    首先，构造函数是不可以被继承的，自然就不能声明为虚函数
    其次，构造函数是用来运行初始化的，虚函数是用来实现多态性的。若尚未构造出来，无法实现多态
3. 静态成员函数
    类的静态成员函数是不可以继承的。对于拥有它的类，仅仅有一份代码，由该类的全部对象共享。
4. friend 函数（即友元函数）
    友元函数不属于类的成员函数，不可以被继承。
5. inline 函数
    内联函数在编译时就会展开运行，不具有多态性

### 为什么析构函数可以为虚函数，如果不设为虚函数可能会存在什么问题？

首先析构函数可以为虚函数，而且当要使用基类指针或引用调用子类时，最好将基类的析构函数声明为虚函数，否则可以存在内存泄露的问题。

举例说明：

```C++
class A : public B
A *p = new B; 
delete p;
```

1. 此时，如果类 A 的析构函数不是虚函数，那 `delete p`将会仅仅调用 A 的析构函数，只释放了 B 对象中的 A 部分，而派生出的新的部分未释放掉。
2. 如果类 A 的析构函数是虚函数，`delete p`将会先调用 B 的析构函数，再调用 A 的析构函数，释放 B 对象的所有空间。

```C++
B *p = new B; 
delete p;
```

此时也是先调用 B 的析构函数，再调用 A 的析构函数。

## 四、Static

Static 关键词作用：

1. 作用域隐藏。当一个工程有多个文件的时候，用 Static 修饰的函数或变量只能够在本文件中可见，文件外不可见。 
2. 全局生命周期。用 Static 修饰的变量或函数生命周期是全局的。被 Static 修饰的变量存储在静态数据区。 
3. Static 修饰的变量默认初始化为 0。
4. Static 修饰的变量或函数是属于类的，所有对象只有一份拷贝。 

### Static 与多态

#### Static 与 virtual

1. Static 成员不属于任何类对象或类实例，所以即使给此函数加上 virtual 也是没有任何意义的。
2. 静态与非静态成员函数之间有一个主要的区别。那就是静态成员函数没有`this`指针。

虚函数依靠`vptr`和`vtable`来处理。`vptr`是一个指针，在类的构造函数中创建生成，并且只能用 this 指针来访问它，因为它是类的一个成员，并且`vptr`指向保存虚函数地址的`vtable`. 

对于静态成员函数，它没有`this`指针，所以无法访问`vptr`。这就是为何 Static 函数不能为 virtual。

this 是指向实例化对象本身时候的一个指针，里面存储的是对象本身的地址，通过该地址可以访问内部的成员函数和成员变量。

为什么需要 this？因为 this 作用域是在类的内部，自己声明一个类的时候，还不知道实例化对象的名字，所以用 this 来使用对象变量的自身。在非静态成员函数中，编译器在编译的时候加上 this 作为隐含形参，通过 this 来访问各个成员（即使你没有写上 this 指针）。例如 `a.fun(1)<==等价于==>fun(&a,1)  `

this 的使用：

1. 在类的非静态成员函数中返回对象的本身时候，直接用 return *this（常用于操作符重载和赋值、拷贝等函数）。
2. 传入函数的形参与成员变量名相同时，例如：this->n = n （不能写成 n=n)

## 五、构造函数和析构函数

### 构造函数

类的构造函数是类的一种特殊的成员函数，它会在每次创建类的新对象时执行。

构造函数的名称与类的名称是完全相同的，并且不会返回任何类型，也不会返回 void。构造函数可用于为某些成员变量设置初始值。

下面的实例有助于更好地理解构造函数的概念：

```C++
#include <iostream>
using namespace std;

class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line();  // 这是构造函数
 
   private:
      double length;
};
 
// 成员函数定义，包括构造函数
Line::Line(void)
{
    cout << "Object is being created" << endl;
}
 
void Line::setLength( double len )
{
    length = len;
}
 
double Line::getLength( void )
{
    return length;
}
// 程序的主函数
int main( )
{
   Line line;
 
   // 设置长度
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
 
   return 0;
}

## 输出
Object is being created
Length of line : 6
```

### 析构函数

类的析构函数是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行。

析构函数的名称与类的名称是完全相同的，只是在前面加了个波浪号（~）作为前缀，它不会返回任何值，也不能带有任何参数。析构函数有助于在跳出程序（比如关闭文件、释放内存等）前释放资源。

下面的实例有助于更好地理解析构函数的概念：
 
```C++
#include <iostream>
 
using namespace std;
 
class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line(double len);  // 这是构造函数
 
   private:
      double length;
};
 
// 成员函数定义，包括构造函数
Line::Line( double len)
{
    cout << "Object is being created, length = " << len << endl;
    length = len;
}
 
void Line::setLength( double len )
{
    length = len;
}
 
double Line::getLength( void )
{
    return length;
}
// 程序的主函数
int main( )
{
   Line line(10.0);
 
   // 获取默认设置的长度
   cout << "Length of line : " << line.getLength() <<endl;
   // 再次设置长度
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
 
   return 0;
}

## Output:
Object is being created, length = 10
Length of line : 10
Length of line : 6
```

## 七、深拷贝与浅拷贝

深拷贝和浅拷贝的定义可以简单理解成：如果一个类拥有资源（堆，或者是其它系统资源），当这个类的对象发生复制过程的时候，这个过程就可以叫做深拷贝，反之对象存在资源，但复制过程并未复制资源的情况视为浅拷贝。

## 八、STL

STL 包含了一些容器，然后容器里面的迭代器，以及配合使用的一些算法。常用的 STL 有比如 vector，list，set，map，stack 等等。

- vector 空间是连续的，插入只能从最后插入，如果想要从中间插入，需要拷贝然后重新分配空间。 
- list 空间是不连续的，插入效率会比较高。 
- map，哈希表，里面存放的元素是键值对的形式，如果需要频繁查找的话，可以选用 map，因为它可以根据 key 值查找，效率比较高。 
- stack，栈，最大的特点是后进先出，一般用在符号匹配，比如括号匹配的时候，一般可以用 stack。 
- 队列 queue，它的特点是输出的头只能输出，输入头只能输入，虽然两头都是开口的。

## 九、struct

class 与 struct 区别

1. 默认的继承访问权。class 默认的是 private, struct 默认的是 public。
2. 默认访问权限：struct 默认的数据访问控制是 public 的，而 class 默认的成员变量访问控制是 private 的。
3. “class”这个关键字还用于定义模板参数，就像“type name”。但关建字“struct”不用于定义模板参数
4. class 和 struct 在使用大括号{ }上的区别

关于使用大括号初始化

1. class 和 struct 如果定义了构造函数的话，都不能用大括号进行初始化
2. 如果没有定义构造函数，struct 可以用大括号初始化。
3. 如果没有定义构造函数，且所有成员变量全是 public 的话，class 可以用大括号初始化
