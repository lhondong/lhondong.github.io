常见的几种字符编码：

- GBK
  - 用途：主要用于简体中文字符的编码。
  - 字节长度：使用 1 个字节表示 ASCII 字符，使用 2 个字节表示中文字符。
  - 优点：对中文字符支持良好，兼容 ASCII，适合纯中文环境。
  - 缺点：不支持所有的 Unicode 字符，国际化支持有限。
- Unicode
  - 用途：是一个字符集，而不是具体的编码方式，它为全球所有书写系统的字符分配了唯一的编码点。
  - 字节长度：不定义字节长度，具体实现依赖于编码方案（如 UTF-8、UTF-16、UTF-32）。
  - 优点：提供统一的字符表示，支持全球所有语言。
  - 缺点：需要结合具体的编码方案来使用。
- UTF-8
  - 用途：一种可变长度的 Unicode 编码方案。
  - 字节长度：使用 1 到 4 个字节，1 个字节用于 ASCII 字符，2 到 4 个字节用于其他字符。
  - 优点：兼容 ASCII，节省存储空间，适合网络传输和以 ASCII 为主的文本。
  - 缺点：对于某些字符（尤其是东亚字符），占用空间比固定长度编码略大。
- UTF-16
  - 用途：一种 Unicode 编码方案，常用于内部处理和存储。
  - 字节长度：通常使用 2 个字节，但对于某些补充字符（超出基本多文种平面），使用 4 个字节。
  - 优点：在处理大量非 ASCII 字符（如中文、日文）时空间效率较高。
  - 缺点：不兼容 ASCII，可能导致字节序问题（需要字节序标记 BOM）。
- UTF-32
  - 用途：一种固定长度的 Unicode 编码方案。
  - 字节长度：每个字符固定使用 4 个字节。
  - 优点：编码简单，字符直接映射到编码点，适合需要快速随机访问字符的应用。
  - 缺点：占用空间较大，不适合存储和传输。
- GBK 适合单一中文环境，但国际化支持不足。
- UTF-8 因其兼容性和空间效率，广泛用于 Web 和跨平台应用。
- UTF-16 在处理大量非 ASCII 字符时有优势，常用于 Windows 及一些编程环境。
- UTF-32 因其简单性适合某些内部处理，但存储效率较低。

## BPE（Byte-Pair-Encoding）

最初用于数据压缩时，处理的是字节级别的数据。在计算机中，1Byte=8bit，对于二进制文件或某些编码格式的文本文件来说，处理的基本单位就是字节。然而，在 NLP 领域应用 BPE 时，我们通常处理的是文本数据，而文本通常由字符构成。在这种情况下，BPE 操作的基本单位就变成了**字符对**，而不是字节对。不过，由于 BPE 这个名字已经沿用了很长时间，即使在处理字符时，技术名称仍然保留为 Byte-Pair Encoding。

### BPE 基本实现流程

Byte-Pair Encoding (BPE) 是一种用于数据压缩和文本处理中子词分割的方法。通过反复合并频繁出现的字符对（或子词对）来减少数据的尺寸。以下是 BPE 的基本流程，并通过一个简单的例子来说明：

### BPE 流程

1. **初始化**：首先计算语料库中使用的唯一单词集（在规范化和预标记步骤之后），然后通过提取用于书写这些单词的所有符号来构建词汇表。
2. **统计词 pair 的频率**：计算所有相邻字符对的出现频率。
3. **合并频繁对&更新词汇表和语料库**：找到出现频率最高的字符对，并将其合并为一个新的字符，并更新词汇表和语料库。
4. **重复 2-3**：重复步骤 2 和 3。
5. **停止条件**：直到达到想要的 vocabulary size。

### 例子

假设我们有一个简单的语料：["hug", "pug", "pun", "bun", "hugs"]

1. **初始化**：
   - base vocabulary（初始化词汇表）: ["b", "g", "h", "n", "p", "s", "u"]
   - 假设 Corpus（语料）里面的词频如下：
     ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
     即("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
2. **统计词 pair 的频率**：
   - 统计所有相邻字符对的频率：
     该词 pair("h", "u")出现在单词"hug"和"hugs"中，即在语料库中总共出现了 15 次。
     但是("u", "g")出现在"hug"、"pug"和 中"hugs"，即在语料库中总共出现了 20 次。
3. **合并频繁对&更新词汇表和语料库**：
   - 因此，分词器学习到的第一个合并规则是("u", "g") -> "ug"
   - 更新词汇表和语料库：
     Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
     Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
4. **重复 2-3**:
   - 根据更新的 Corpus 重新统计词 pair 的频率，发现("u", "n")出现了 16 次
   - 因此，分词器学习到的第一个合并规则是("u", "n") -> "un"
   - 更新词汇表和语料库：
     Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
     Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
5. **停止条件**:
   - 直到达到想要的 vocabulary size。

通过这种方式，BPE 逐步将文本压缩为较少的符号表示，同时保留文本的信息结构。在自然语言处理中，BPE 被广泛用于生成子词单元，以处理词汇表外词（OOV）的问题，并在神经网络模型中提高文本表示的效率。

## BBPE（Byte-Byte-Pair-Encoding）

一种 BPE 的衍生物，和 BPE 的区别如下：

- BPE 的最小词汇是字符级别，而 BBPE 的最小词汇是**字节级别**
- BBPE 的流程和 BPE 一样，不同之处在于实现的粒度

```python
from collections import defaultdict

sentences = [
    "我",
    "喜欢",
    "吃",
    "苹果",
    "他",
    "不",
    "喜欢",
    "吃",
    "苹果派",
    "I like to eat apples",
    "She has a cute cat",
    "you are very cute",
    "give you a hug",
]

# 构建初始词汇表，包含一个字节的256个表示
initial_vocab = [bytes([byte]) for byte in range(256)]
vocab = initial_vocab.copy()
print("initial_vocab:", initial_vocab)

# 构建频率统计
def build_stats(sentences):
    """
    input：句子list
    output：{字节级字符1&字节级字符2&...&字节级字符n: 词频...}
    """
    stats = defaultdict(int)
    for sentence in sentences:
        symbols = sentence.split()
        for symbol in symbols:
            stats[symbol.encode("utf-8")] += 1
    return stats
stats = build_stats(sentences)
print("stats: ", stats)

# 这里[byte for byte in word]就自动将utf8编码的word -> 转化为数字了
# splits = {b'\xe6\x88\x91': [230, 136, 145], b'\xe5\x96\x9c\xe6\xac\xa2': [229, 150, 156, 230, 172, 162],...}
splits = {word: [byte for byte in word] for word in stats.keys()}
print("splits: ", splits)

# 计算
def compute_pair_freqs(splits):
    """
    input：{b'\xe6\x88\x91': [230, 136, 145], b'\xe5\x96\x9c\xe6\xac\xa2': [229, 150, 156, 230, 172, 162],...}
    output：{(230, 136): 1, (136, 145): 1,...}
    """
    pair_freqs = defaultdict(int)
    for word, freq in stats.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)
print("pair_freqs: ", pair_freqs)

# 合并
def merge_pair(pair, splits):
    merged_byte = bytes(pair)
    for word in stats:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i:i+2] == list(pair):  # 检查分割中是否有这对字节
                split = split[:i] + [merged_byte] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

vocab_size = 500
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ()
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    print("best_pair: ", best_pair)
    splits = merge_pair(best_pair, splits)
    print("merge_splites: ", splits)
    print()
    merged_byte = bytes(best_pair)
    break

print("vocab:", vocab)
```

中间过程中，\x9c 和\xe6 会被合并到一起。和 BPE 一样，持续合并，直到满足要求数量。

## WPE（Word-Piece-Encoding）

在 BERT 中常用

|               | 自下而上 WordPiece（BPE 风格）        | 自上而下 WordPiece（BERT 风格）            |
| ------------- | ------------------------------------- | ------------------------------------------ |
| 起点          | 字母表（单个字符）                    | 完整单词（需预先分词）                     |
| 核心过程      | 合并高频相邻词片段对                  | 拆分低频词                                 |
| 方向          | 从小单元（字符）向上构建成更大子词/词 | 从大单元（词）向下分解成更小子词           |
| 类似算法      | BPE                                   | BERT 原始实现                              |
| 是否适用 CJK  | 是（从字符开始，不依赖预分词）        | 否（依赖预分词，而 CJK 无天然词边界）      |
| 处理 CJK 方案 | 直接应用                              | 需先使用专用分词器（如 Jieba）分词，再应用 |

假设单词“不可能"在语料库中频繁出现。

- 自下而上（BPE 风格）：从单字 “不”，“可”，“能” 开始。算法可能会发现 “可” 和 “能” 经常连在一起出现，于是将它们合并成子词 “可能”。接着，它可能发现 “不” 和 “可能” 也经常连用， 于是将它们合井成 “不可能” 。最终调汇表可能包含 “不”，“可能”，“不可能” 等片段。
- 自上而下（BERT 风格）：从完整单词 “不可能” 开始。算法检查这个词是否足够常见（超过频率阈值）。如果不够常见（或者出于优化目标考虑），尝试将其拆分。如可能拆分成前缀 “不” 和后缀 “##可能”（## 常表示非词首子词）。接着检查 “可能”，如果它本身是常见词，就保留；如果不够常见，可能进一步拆分成 “可” 和 “##能”。最终词汇表可能包含 “不”，“可能”，“可”， “##能” 等片段。
