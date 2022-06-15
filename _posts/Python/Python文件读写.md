
# 文件读写

## 文件的打开

- 文件打开的通用格式

```python
with open("文件路径", "打开模式", encoding = "操作文件的字符编码") as f:
    "对文件进行相应的读写操作"
```

使用 with 块的好处：执行完毕后，自动对文件进行 close 操作。

```python
with open("测试文件.txt", "r", encoding = "gbk") as f: # 第一步：打开文件
    text = f.read()  # 第二步：读取文件
    print(text)
```

- "r": 只读模式，如文件不存在，报错
- "w": 覆盖写模式，如文件不存在，则创建；如文件存在，则完全覆盖原文件
- "x": 创建写模式，如文件不存在，则创建；如文件存在，报错
- "a": 追加写模式，如文件不存在，则创建；如文件存在，则在原文件后追加内容
- "b": 二进制文件模式，不能单独使用，需要配合使用如"rb"，"wb"，"ab"，**该模式不需指定 encoding**
- "t": 文本文件模式，默认值，需配合使用 如"rt"，"wt"，"at"，一般省略，简写成如"r"，"w"，"a"
- "+": 与"r","w","x","a"配合使用，在原功能基础上，增加读写功能
- **打开模式缺省，默认为只读模式**

### 字符编码

- **万国码 utf-8**：包含全世界所有国家需要用到的字符
- **中文编码 gbk**：专门解决中文编码问题
- **windows 系统下，如果缺省，则默认为 gbk（所在区域的编码）**
- 为清楚起见，除了处理二进制文件，**建议不要缺省 encoding**

## 文件的读取

### 读取整个内容 f.read()

```python
with open("三国演义.txt", "r", encoding="gbk") as f: # 第一步：打开文件，"r" 可缺省，为清晰起见，最好写上
    text = f.read() # 第二步：读取文件
    print(text)
```

解码模式必须匹配，如果文件是 gbk，则必须用 encoding="gbk" 解码；如果文件是 utf-8 encoding="utf-8" 解码，否则会报错。

### 逐行进行读取 f.readline()

```python
with open("三国演义.txt", "r", encoding="gbk") as f:     
    for i in range(3):
        text = f.readline() # 每次只读取一行
        print(text)
```

```python
with open("三国演义.txt", "r", encoding="gbk") as f:     
    while True:
        text = f.readline()
        if not text:
            # print(text is "") 
            break
        else:
            # print(text == "\n") # 空行不为空，而是 "\n"
            print(text, end="")  # 保留原文的换行，使 print() 的换行不起作用，否则会有原文换行和 print 换行，两次换行
```

### 读入所有行，以每行为元素形成一个列表 f.readlines()

```python
with open("三国演义.txt", "r", encoding="gbk") as f:
    text = f.readlines() # 注意每行末尾有换行符
    print(text)

Out:
['临江仙·滚滚长江东逝水、n', '滚滚长江东逝水，浪花淘尽英雄。\n', '是非成败转头空。\n', '\n', '青山依旧在，几度夕阳红。\n', '白发渔樵江渚上，惯看秋月春风。\n', '一壶浊酒喜相逢。\n', '古今多少事，都付笑谈中。\n']
```

```python
with open("三国演义片头曲_gbk.txt", "r", encoding="gbk") as f:
    for text in f.readlines():
        print(text) # 不想换行则用 print(text, end="")   

Out:
临江仙·滚滚长江东逝水

滚滚长江东逝水，浪花淘尽英雄。

是非成败转头空。

青山依旧在，几度夕阳红。

白发渔樵江渚上，惯看秋月春风。

一壶浊酒喜相逢。

古今多少事，都付笑谈中。 
```

文件比较大时，`read()` 和 `readlines()` 占用内存过大，不建议使用，`readline()` 用起来又不太方便。可以直接对对象 f 进行迭代。

```python
with open("三国演义片头曲_gbk.txt", "r", encoding="gbk") as f:     
    for text in f:         # f 本身就是一个可迭代对象，每次迭代读取一行内容 
        print(text)
```

### 图片：二进制文件

```python
with open("test.png", "rb") as f:
    print(len(f.readlines()))
```

## 文件的写入

### 向文件写入一个字符串或字节流（二进制） f.write()

```python
with open("恋曲 1980.txt", "w", encoding="utf-8") as f:                      
        f.write("你曾经对我说\n")        # 文件不存在则立刻创建一个
        f.write("你永远爱着我\n")        # 如需换行，末尾加换行符\n
        f.write("爱情这东西我明白\n")
        f.write("但永远是什么\n")
```

如果文件存在，新写入内容会覆盖掉原内容，一定要注意！！！

### 追加模式"a"

`with open("恋曲 1980.txt", "a", encoding="utf-8") as f:`

### 将一个元素为字符串的列表整体写入文件 f.writelines()

```python
ls = ["春天刮着风", "秋天下着雨", "春风秋雨多少海誓山盟随风远去"]
with open("恋曲 1980.txt", "w", encoding="utf-8") as f:
    f.writelines(ls)

ls = ["春天刮着风\n", "秋天下着雨\n", "春风秋雨多少海誓山盟随风远去\n"]
with open("恋曲 1980.txt", "w", encoding="utf-8") as f:
    f.writelines(ls)
```

## 既读又写

### "r+" 

- 如果文件名不存在，则报错
- 指针在开始
- 要把指针移到末尾才能开始写，否则会覆盖前面内容

```python
with open("浪淘沙_北戴河.txt", "r+", encoding="gbk") as f:
#     for line in f:
#         print(line)   # 全部读一遍后，指针到达结尾
    f.seek(0,2) # 或者可以将指针移到末尾f.seek(偏移字节数,位置（0：开始；1：当前位置；2：结尾）)
    text = ["萧瑟秋风今又是，\n", "换了人间。\n"]
    f.writelines(text)
```

### "w+"

- 若文件不存在，则创建
- 若文件存在，会立刻清空原内容！！！

```python
with open("浪淘沙_北戴河.txt", "w+", encoding="gbk") as f:
    text = ["萧瑟秋风今又是，\n", "换了人间。\n"]  # 清空原内容
    f.writelines(text)     # 写入新内容，指针在最后
    f.seek(0,0)            # 指针移到开始
    print(f.read())        # 读取内容
```

### "a+"

- 若文件不存在，则创建
- 指针在末尾，添加新内容，不会清空原内容

```python
with open("浪淘沙_北戴河.txt", "a+", encoding="gbk") as f:
    f.seek(0,0)            # 指针移到开始
    print(f.read())        # 读取内容
```

## 数据的存储与读取

### cvs 模式

由逗号将数据分开的字符序列，可以由 excel 打开。

- 读取

```python
with open("成绩.csv", "r", encoding="gbk") as f:
    ls = []
    for line in f:                              # 逐行读取
        ls.append(line.strip("\n").split(","))  # 去掉每行的换行符，然后用“,”进行分割
for res in ls:
    print(res)
```

- 写入

```python
ls = [['编号', '数学成绩', '语文成绩'], ['1', '100', '98'], ['2', '96', '99'], ['3', '97', '95']]
with open("score.csv", "w", encoding="gbk") as f:   # encoding="utf-8"中文出现乱码
    for row in ls:                          # 逐行写入
        f.write(",".join(row)+"\n")         # 用逗号组合成字符串形式，末尾加换行符
```

**也可以借助csv模块完成上述操作。**

### json 模式

常被用来存储字典类型

- 写入 `dump()`

```python
import json

scores = {"Petter":{"math":96 , "physics": 98},
        "Paul":{"math":92 , "physics": 99},
        "Mary":{"math":98 , "physics": 97}}
with open("score.json", "w", encoding="utf-8") as f:             # 写入整个对象 
        # indent 表示字符串换行+缩进 ensure_ascii=False 显示中文
        json.dump(scores, f, indent=4, ensure_ascii=False)   
```

- 读取 `load()`

```python
with open("score.json", "r", encoding="utf-8") as f:      
        scores = json.load(f)           # 加载整个对象
        for k,v in scores.items():
            print(k,v)
```