# PyTorch 数据处理

- 数据收集：Img，Label
- 数据划分：train valid test
- 数据读取：Dataloader
  - Sampler -> Index
  - Dataset -> Img, Label
- 数据预处理：transforms

## 1. Dataloader

#### torch.utils.data.DataLoader 

功能：构建可迭代的数据装载器

- dataset: Dataset 类，决定数据从哪读取及如何读取
- batchsize: 批大小
- num_works: 是否多进程读取数据
- shuffle: 每个 epoch 是否乱序
- drop_last: 当样本数不能被 batchsize 整除时，是否舍弃最后一批数据

关于 Epoch，Iteration，Batchsize

- Epoch: 所有训练样本都已输入到模型中，称为一个 Epoch 
- Iteration: 一批样本输入到模型中，称之为一个 Iteration 
- Batchsize: 批大小，决定一个 Epoch 有多少个 Iteration 

如样本总数：80，Batchsize:8，则 1 Epoch = 10 Iteration

如样本总数：87， Batchsize:8

- 1 Epoch = 10 Iteration if drop_last = True 
- 1 Epoch = 11 Iteration if drop_last = False

```python
# 构建 MyDataset 实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建 DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
```

## 2. DataSet

#### torch.utils.data.Dataset

功能：Dataset 抽象类，所有自定义的 Dataset 需要继承它，并且复写 `__getitem__()`

- getitem : 接收一个索引，返回一个样本

```python
class Dataset(object):
    def __init__(self, other):
        pass

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)
```

### DataLoader, DataSet, Sampler 之间的关系

DataLoader.next 的源码，只选取了 num_works 为 0 的情况（num_works 简单理解就是能够并行化地读取数据）。

```python
class DataLoader(object):
	...
	
    def __next__(self):
        if self.num_workers == 0:  
            indices = next(self.sample_iter)  # Sampler
            batch = self.collate_fn([self.dataset[i] for i in indices]) # Dataset
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch
```

可以假设我们的数据是一组图像，每一张图像对应一个 index，那么如果我们要读取数据就只需要对应的 index 即可，即上面代码中的 indices，而选取 index 的方式有多种，有按顺序的，也有乱序的，所以这个工作需要 Sampler 完成，DataLoader 和 Sampler 在这里产生关系。

那么 Dataset 和 DataLoader 在什么时候产生关系呢？没错就是下面一行。我们已经拿到了 indices，那么下一步我们只需要根据 index 对数据进行读取即可了。

再下面的 if 语句的作用简单理解就是，如果 pin_memory=True, 那么 Pytorch 会采取一系列操作把数据拷贝到 GPU，总之就是为了加速。

综上可以知道 DataLoader，Sampler 和 Dataset 三者关系如下：

<div align=center><img src="/assets/PyTorch数据处理-2022-05-22-21-18-39.png" alt="PyTorch数据处理-2022-05-22-21-18-39" style="zoom:50%;" /></div>

<div align=center><img src="/assets/PyTorch数据处理-2022-05-22-17-06-20.png" alt="PyTorch数据处理-2022-05-22-17-06-20" style="zoom:50%;" /></div>

### Sampler

#### 参数传递

要更加细致地理解 Sampler 原理，需要先阅读一下 DataLoader 的源代码，如下：

```python
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None)
```

可以看到初始化参数里有两种 sampler：`sampler` 和 `batch_sampler`，都默认为 `None`。前者的作用是生成一系列的 index，而 batch_sampler 则是将 sampler 生成的 indices 打包分组，得到一个又一个 batch 的 index。例如下面示例中，`BatchSampler` 将 `SequentialSampler` 生成的 index 按照指定的 batch size 分组。

```python
>>>in : list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
>>>out: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
```

Pytorch 中已经实现的 `Sampler` 有如下几种：

- `SequentialSampler`
- `RandomSampler`
- `WeightedSampler`
- `SubsetRandomSampler`

需要注意的是 DataLoader 的部分初始化参数之间存在互斥关系，这个可以阅读 [源码](https://github.com/pytorch/pytorch/blob/0b868b19063645afed59d6d49aff1e43d1665b88/torch/utils/data/dataloader.py#L157-L182) 更深地理解，这里只做总结：

- 如果自定义了 `batch_sampler`, 那么这些参数都必须使用默认值：`batch_size`, `shuffle`,`sampler`,`drop_last`.
- 如果自定义了 `sampler`，那么 `shuffle` 需要设置为 `False`
- 如果 `sampler` 和 `batch_sampler` 都为 `None`, 那么 `batch_sampler` 使用 Pytorch 已经实现好的 `BatchSampler`, 而 `sampler` 分两种情况：
    - 若 `shuffle=True`, 则 `sampler=RandomSampler(dataset)`
    - 若 `shuffle=False`, 则 `sampler=SequentialSampler(dataset)`

#### 如何自定义 Sampler 和 BatchSampler？

仔细查看源代码其实可以发现，所有采样器其实都继承自同一个父类，即 `Sampler`, 其代码定义如下：

```python
class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError
		
    def __len__(self):
        return len(self.data_source)
```

所以你要做的就是定义好 `__iter__(self)` 函数，不过要注意的是该函数的返回值需要是可迭代的。例如 `SequentialSampler` 返回的是 `iter(range(len(self.data_source)))`。

另外 `BatchSampler` 与其他 Sampler 的主要区别是它需要将 Sampler 作为参数进行打包，进而每次迭代返回以 batch size 为大小的 index 列表。也就是说在后面的读取数据过程中使用的都是 batch sampler。

### Dataset

Dataset 定义方式如下：

```python
class Dataset(object):
	def __init__(self):
		...
		
	def __getitem__(self, index):
		return ...
	
	def __len__(self):
		return ...
```

上面三个方法是最基本的，其中 `__getitem__` 是最主要的方法，它规定了如何读取数据。但是它又不同于一般的方法，因为它是 python built-in 方法，其主要作用是能让该类可以像 list 一样通过索引值对数据进行访问。假如你定义好了一个 dataset，那么你可以直接通过 `dataset[0]` 来访问第一个数据。在此之前我一直没弄清楚 `__getitem__` 是什么作用，所以一直不知道该怎么进入到这个函数进行调试。现在如果你想对 `__getitem__` 方法进行调试，你可以写一个 for 循环遍历 dataset 来进行调试了，而不用构建 dataloader 等一大堆东西了，建议学会使用 `ipdb`这 个库，非常实用！

另外，其实我们通过最前面的 Dataloader 的 `__next__` 函数可以看到 DataLoader 对数据的读取其实就是用了 for 循环来遍历数据。

```python
class DataLoader(object): 
    ... 
     
    def __next__(self): 
        if self.num_workers == 0:   
            indices = next(self.sample_iter)  
            batch = self.collate_fn([self.dataset[i] for i in indices]) # this line 
            if self.pin_memory: 
                batch = _utils.pin_memory.pin_memory_batch(batch) 
            return batch
```

我们仔细看可以发现，前面还有一个 `self.collate_fn` 方法，在介绍前我们需要知道每个参数的意义：

- `indices`: 表示每一个 iteration，sampler 返回的 indices，即一个 batch size 大小的索引列表
- `self.dataset[i]`: 前面已经介绍了，这里就是对第 i 个数据进行读取操作，一般来说 `self.dataset[i]=(img, label)`

看到这不难猜出 `collate_fn` 的作用就是将一个 batch 的数据进行合并操作。默认的 `collate_fn` 是将 img 和 label 分别合并成 imgs 和 labels，所以如果你的 `__getitem__` 方法只是返回 `img, label`, 那么你可以使用默认的 `collate_fn` 方法，但是如果你每次读取的数据有 `img, box, label` 等等，那么你就需要自定义 `collate_fn` 来将对应的数据合并成一个 batch 数据，这样方便后续的训练步骤。

#### 读哪些数据？

- Sampler 输出的 Index

#### 从哪读数据？

- Dataset 中的 data_dir

#### 怎么读数据？

- Dataset 中的 getitem

## Transforms

torchvision: 计算机视觉工具包

- torchvision.transforms: 常用的图像预处理方法
- torchvision.datasets: 常用数据集的 dataset 实现，MNIST，CIFAR-10，ImageNet 等
- torchvision.model: 常用的模型预训练，AlexNet，VGG， ResNet，GoogLeNet 等

torchvision.transforms : 常用的图像预处理方法

- 数据中心化
- 数据标准化
- 缩放
- 裁剪
- 旋转
- 翻转
- 填充
- 噪声添加
- 灰度变换
- 线性变换
- 仿射变换
- 亮度、饱和度及对比度变换

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

## 数据标准化

### transforms.Normalize

功能：逐 channel 的对图像进行标准化 

output = (input - mean) / std

- mean: 各通道的均值
- std: 各通道的标准差
- inplace: 是否原地操作

`transforms.Normalize(mean, std, inplace=False)`

## 数据预处理

## 数据增强（22 种模块）