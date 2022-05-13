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

功能：逐channel的对图像进行标准化 

output = (input - mean) / std

- mean: 各通道的均值
- std: 各通道的标准差
- inplace: 是否原地操作

`transforms.Normalize(mean, std, inplace=False)`

## 数据预处理

## 数据增强（22 种模块）