# 正则表达式

Python 通过标准库中的 re 模块来支持正则表达式操作。

关于正则表达式的相关知识，可以阅读一篇非常有名的博客叫 [《正则表达式 30 分钟入门教程》](https://deerchao.net/tutorials/regex/regex.htm)，下面的表格是对正则表达式中的一些基本符号进行的扼要总结。

| 符号 | 解释 | 示例 | 说明 |
| --- | --- | --- | --- |
| . | 匹配任意字符 | b.t | 可以匹配 bat / but / b#t / b1t 等 |
| \\w | 匹配字母/数字/下划线 | b\\wt | 可以匹配 bat / b1t / b_t 等<br>但不能匹配 b#t |
| \\s | 匹配空白字符（包括、r、\n、\t 等）| love\\syou | 可以匹配 love you |
| \\d | 匹配数字| \\d\\d | 可以匹配 01 / 23 / 99 等 |
| \\b | 匹配单词的边界 | \\bThe\\b | |
| ^ | 匹配字符串的开始 | ^The | 可以匹配 The 开头的字符串 |
| $ | 匹配字符串的结束 | .exe$  | 可以匹配 .exe 结尾的字符串 |
| \\W | 匹配非字母/数字/下划线 | b\\Wt | 可以匹配 b#t / b@t 等<br>但不能匹配 but / b1t / b_t 等 |
| \\S | 匹配非空白字符 | love\\Syou | 可以匹配 love#you 等<br>但不能匹配 love you |
| \\D | 匹配非数字 | \\d\\D | 可以匹配 9a / 3# / 0F 等 |
| \\B | 匹配非单词边界 | \\Bio\\B | |
| [ ] | 匹配来自字符集的任意单一字符 | [aeiou] | 可以匹配任一元音字母字符 |
| [^] | 匹配不在字符集中的任意单一字符 | [^aeiou] | 可以匹配任一非元音字母字符 |
| * | 匹配 0 次或多次 | \\w* | |
| + | 匹配 1 次或多次 | \\w+ | |
| ? | 匹配 0 次或 1 次 | \\w? | |
| {N}| 匹配 N 次 | \\w{3} | |
| {M,} | 匹配至少 M 次 | \\w{3,} | |
| {M,N} | 匹配至少 M 次至多 N 次 | \\w{3,6}| |
| \| | 分支 | foo\|bar | 可以匹配 foo 或者 bar |
| (?#) | 注释 | | |
| (exp) | 匹配 exp 并捕获到自动命名的组中 | | |
| (?&lt;name&gt;exp) | 匹配 exp 并捕获到名为 name 的组中 | | |
| (?:exp) | 匹配 exp 但是不捕获匹配的文本 | | |
| (?=exp) | 匹配 exp 前面的位置 | \\b\\w+(?=ing) | 可以匹配 I'm dancing 中的 danc |
| (?<=exp) | 匹配 exp 后面的位置 | (?<=\\bdanc)\\w+\\b | 可以匹配 I love dancing and reading 中的第一个 ing |
| (?!exp) | 匹配后面不是 exp 的位置 | | |
| (?<!exp) | 匹配前面不是 exp 的位置 | | |
| *? | 重复任意次，但尽可能少重复 | a.\*b<br>a.\*?b | 将正则表达式应用于 aabab，前者会匹配整个字符串 aabab，后者会匹配 aab 和 ab 两个字符串 |
| +? | 重复 1 次或多次，但尽可能少重复 | | |
| ?? | 重复 0 次或 1 次，但尽可能少重复 | | |
| {M,N}? | 重复 M 到 N 次，但尽可能少重复 | | |
| {M,}? | 重复 M 次以上，但尽可能少重复 | | |

**说明：** 如果需要匹配的字符是正则表达式中的特殊字符，那么可以使用 `\` 进行转义处理，例如想匹配小数点可以写成 `\.` 就可以了，因为直接写 `.` 会匹配任意字符；同理，想匹配圆括号必须写成 `\(` 和 `\)`，否则圆括号被视为正则表达式中的分组。

### Python 对正则表达式的支持

Python 提供了 re 模块来支持正则表达式相关操作，下面是 re 模块中的核心函数。

| 函数 | 说明 |
| --- | --- |
| compile(pattern, flags=0) | 编译正则表达式返回正则表达式对象 |
| match(pattern, string, flags=0) | 用正则表达式匹配字符串，成功返回匹配对象，否则返回 None |
| search(pattern, string, flags=0) | 搜索字符串中第一次出现正则表达式的模式，成功返回匹配对象，否则返回 None |
| split(pattern, string, maxsplit=0, flags=0) | 用正则表达式指定的模式分隔符拆分字符串，返回列表 |
| sub(pattern, repl, string, count=0, flags=0) | 用指定的字符串替换原字符串中与正则表达式匹配的模式，可以用 count 指定替换的次数 |
| fullmatch(pattern, string, flags=0)| match 函数的完全匹配（从字符串开头到结尾）版本|
| findall(pattern, string, flags=0) | 查找字符串所有与正则表达式匹配的模式，返回字符串的列表 |
| finditer(pattern, string, flags=0) | 查找字符串所有与正则表达式匹配的模式，返回一个迭代器|
| purge() | 清除隐式编译的正则表达式的缓存 |
| re.I / re.IGNORECASE | 忽略大小写匹配标记 |
| re.M / re.MULTILINE| 多行匹配标记 |

**说明：** 上面提到的 re 模块中的这些函数，实际开发中也可以用正则表达式对象的方法替代对这些函数的使用，如果一个正则表达式需要重复的使用，那么先通过 compile 函数编译正则表达式并创建出正则表达式对象无疑是更为明智的选择。


## 例：验证输入用户名和 QQ 号是否有效并给出对应的提示信息。

```Python
"""
验证输入用户名和 QQ 号是否有效并给出对应的提示信息
要求：用户名必须由字母、数字或下划线构成且长度在 6~20 个字符之间，QQ 号是 5~12 的数字且首位不能为 0
"""
import re

def main():
    username = input('请输入用户名：')
    qq = input('请输入 QQ 号：')
    # match 函数的第一个参数是正则表达式字符串或正则表达式对象
    # 第二个参数是要跟正则表达式做匹配的字符串对象
    m1 = re.match(r'^[0-9a-zA-Z_]{6,20}$', username)
    if not m1:
        print('请输入有效的用户名。')
    m2 = re.match(r'^[1-9]\d{4,11}$', qq)
    if not m2:
        print('请输入有效的 QQ 号。')
    if m1 and m2:
        print('你输入的信息是有效的！')

if __name__ == '__main__':
    main()
```

**提示：** 上面在书写正则表达式时使用了“原始字符串”的写法（在字符串前面加上了 r），所谓“原始字符串”就是字符串中的每个字符都是它原始的意义，说得更直接一点就是字符串中没有所谓的转义字符。因为正则表达式中有很多元字符和需要进行转义的地方，如果不使用原始字符串就需要将反斜杠写作 `\\\`，例如表示数字的 `\d` 得书写成 `\\\d`，这样不仅写起来不方便，阅读的时候也会很吃力。

## 例 2：从一段文字中提取出国内手机号码。

```Python
import re

def main():
    # 创建正则表达式对象 使用了前瞻和回顾来保证手机号前后不应该出现数字
    pattern = re.compile(r'(?<=\D)1[34578]\d{9}(?=\D)')
    sentence = '''
    重要的事情说 8130123456789 遍，我的手机号是 13512346789 这个靓号，
    不是 15600998765，也是 110 或 119，王大锤的手机号才是 15600998765。
    '''
    # 查找所有匹配并保存到一个列表中
    mylist = re.findall(pattern, sentence)
    print(mylist)
    print('--------华丽的分隔线--------')
    # 通过迭代器取出匹配对象并获得匹配的内容
    for temp in pattern.finditer(sentence):
        print(temp.group())
    print('--------华丽的分隔线--------')
    # 通过 search 函数指定搜索位置找出所有匹配
    m = pattern.search(sentence)
    while m:
        print(m.group())
        m = pattern.search(sentence, m.end())

if __name__ == '__main__':
    main()
```

**说明：** 上面匹配国内手机号的正则表达式并不够好，因为像 14 开头的号码只有 145 或 147，而上面的正则表达式并没有考虑这种情况，要匹配国内手机号，更好的正则表达式的写法是：`(?<=\D)(1[38]\d{9}|14[57]\d{8}|15[0-35-9]\d{8}|17[678]\d{8})(?=\D)`，国内最近好像有 19 和 16 开头的手机号了，但是这个暂时不在我们考虑之列。

## 例 3：替换字符串中的不良内容

```Python
import re

def main():
    sentence = '你丫是傻叉吗？我操你大爷的。Fuck you.'
    purified = re.sub('[操肏艹]|fuck|shit|傻[比屄逼叉缺吊屌]|煞笔',
      '*', sentence, flags=re.IGNORECASE)
    print(purified)  # 你丫是*吗？我*你大爷的。* you.

if __name__ == '__main__':
    main()
```

**说明：** re 模块的正则表达式相关函数中都有一个 flags 参数，它代表了正则表达式的匹配标记，可以通过该标记来指定匹配时是否忽略大小写、是否进行多行匹配、是否显示调试信息等。如果需要为 flags 参数指定多个值，可以使用 [按位或运算符](http://www.runoob.com/python/python-operators.html#ysf5) 进行叠加，如`flags=re.I | re.M`。

## 例 4：拆分长字符串

```Python
import re

def main():
    poem = '窗前明月光，疑是地上霜。举头望明月，低头思故乡。'
    sentence_list = re.split(r'[，。, .]', poem)
    while '' in sentence_list:
        sentence_list.remove('')
    print(sentence_list)  # ['窗前明月光', '疑是地上霜', '举头望明月', '低头思故乡']

if __name__ == '__main__':
    main()
```
