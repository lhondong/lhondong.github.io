# PyTorch 数据处理

- 数据收集：Img，Label
- 数据划分：train valid test
- 数据读取：Dataloader
  - Sampler -> Index
  - Dataset -> Img, Label
- 数据预处理：transforms

## 1. Dataloader

##### torch.utils.data.DataLoader 

功能：构建可迭代的数据装载器

- dataset: Dataset 类，决定数据从哪读取及如何读取
- batchsize : 批大小
- num_works: 是否多进程读取数据
- shuffle: 每个 epoch 是否乱序
- drop_last: 当样本数不能被 batchsize 整除时，是否舍弃最后一批数据

关于 Epoch，Iteration，Batchsize

- Epoch: 所有训练样本都已输入到模型中，称为一个 Epoch 

- Iteration: 一批样本输入到模型中，称之为一个 Iteration 

- Batchsize: 批大小，决定一个 Epoch 有多少个 Iteration 

如样本总数：80，Batchsize:8，则 1 Epoch = 10 Iteration

如样本总数：87， Batchsize:8

1 Epoch = 10 Iteration if drop_last = True 

1 Epoch = 11 Iteration if drop_last = False

## 2. DataSet

##### torch.utils.data.Dataset

功能：Dataset 抽象类，所有自定义的 Dataset 需要继承它，并且复写 `__getitem__()`

- getitem : 接收一个索引，返回一个样本

## Transforms

## 数据标准化

## 数据预处理

## 数据增强（22 种模块）