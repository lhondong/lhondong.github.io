
# PyTorch Distributed Data Parallel

DistributedDataParallel（DDP）是一个支持多机多卡、分布式训练的深度学习工程方法。PyTorch 现已原生支持 DDP，可以直接通过 torch.distributed 使用。

## 快速运行

### 依赖

PyTorch(gpu)>=1.5，python>=3.6

### 环境准备

推荐使用官方打好的 PyTorch docker，避免乱七八糟的环境问题影响心情。

```shell
# Dockerfile
# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.03-py3
```

### 代码

单 GPU 代码

```python
## main.py文件
import torch

# 构造模型
model = nn.Linear(10, 10).to(local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()

## Bash运行
python main.py
```

加入 DDP 的代码

```python
## main.py文件
import torch
# 新增：
import torch.distributed as dist

# 新增：从外面得到local_rank参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()


## Bash运行
# 改变：使用torch.distributed.launch启动DDP模式，
#   其会给main.py一个local_rank的参数。这就是之前需要"新增:从外面得到local_rank参数"的原因
python -m torch.distributed.launch --nproc_per_node 4 main.py
```

## DDP 的基本原理

### 大白话原理

假如我们有 N 张显卡，

1. （缓解 GIL 限制）在 DDP 模式下，会有 N 个进程被启动，每个进程在一张卡上加载一个模型，这些模型的参数在数值上是相同的。
2. （Ring-Reduce 加速）在模型训练时，各个进程通过一种叫 Ring-Reduce 的方法与其他进程通讯，交换各自的梯度，从而获得所有进程的梯度；
3. （实际上就是 Data Parallelism）各个进程用平均后的梯度更新自己的参数，因为各个进程的初始参数、更新梯度是一致的，所以更新后的参数也是完全相同的。

### 与 DP 模式的不同

那么，DDP 对比 Data Parallel（DP）模式有什么不同呢？

DP 模式是很早就出现的、单机多卡的、参数服务器架构的多卡训练模式，在 PyTorch，即是：

```python
model = torch.nn.DataParallel(model)
```

在 DP 模式中，总共只有一个进程（受到 GIL 很强限制）。master 节点相当于参数服务器，其会向其他卡广播其参数；在梯度反向传播后，各卡将梯度集中到 master 节点，master 节点对搜集来的参数进行平均后更新参数，再将参数统一发送到其他卡上。这种参数更新方式，会导致 master 节点的计算任务、通讯量很重，从而导致网络阻塞，降低训练速度。

但是 DP 也有优点，优点就是代码实现简单。要速度还是要方便，看官可以自行选用噢。

### DDP 为什么能加速？

### Python GIL

Python GIL 的存在使得，一个 python 进程只能利用一个 CPU 核心，不适合用于计算密集型的任务。使用多进程，才能有效率利用多核的计算资源。而 DDP 启动多进程训练，一定程度地突破了这个限制。

### Ring-Reduce 梯度合并

Ring-Reduce 是一种分布式程序的通讯方法。

- 因为提高通讯效率，Ring-Reduce 比 DP 的 parameter server 快。
- 其避免了 master 阶段的通讯阻塞现象，n 个进程的耗时是 O(n)。
- 详细的介绍：[ring allreduce 和 tree allreduce 的具体区别是什么？](https://www.zhihu.com/question/57799212/answer/612786337)

**简单说明**

- 各进程独立计算梯度。
- 每个进程将梯度依次传递给下一个进程，之后再把从上一个进程拿到的梯度传递给下一个进程。循环 n 次（进程数量）之后，所有进程就可以得到全部的梯度了。
- **每个进程只跟自己上下游两个进程进行通讯，极大地缓解了参数服务器的通讯阻塞现象！**

### 并行计算

统一来讲，神经网络中的并行有以下三种形式：

1. Data Parallelism
   - 这是最常见的形式，通俗来讲，就是增大 batch size。
   - **平时我们看到的多卡并行就属于这种。比如 DP、DDP 都是。这能让我们方便地利用多卡计算资源。**
   - 能加速。
2. Model Parallelism
   - 把模型放在不同 GPU 上，计算是并行的。
   - 有可能是加速的，看通讯效率。
3. Workload Partitioning
   - 把模型放在不同 GPU 上，但计算是串行的。
   - 不能加速。

### 如何在 PyTorch 中使用 DDP

DDP 有不同的使用模式。**DDP 的官方最佳实践是，每一张卡对应一个单独的 GPU 模型（也就是一个进程），在下面介绍中，都会默认遵循这个 pattern。**  

举个例子：我有两台机子，每台 8 张显卡，那就是 2x8=16 个进程，并行数是 16。

但是，我们也是可以给每个进程分配多张卡的。总的来说，分为以下三种情况：

1. 每个进程一张卡。这是 DDP 的最佳使用方法。
2. 每个进程多张卡，复制模式。一个模型复制在不同卡上面，每个进程都实质等同于 DP 模式。这样做是能跑得通的，但是，速度不如上一种方法，一般不采用。
3. 每个进程多张卡，并行模式。一个模型的不同部分分布在不同的卡上面。例如，网络的前半部分在 0 号卡上，后半部分在 1 号卡上。这种场景，一般是因为我们的模型非常大，大到一张卡都塞不下 batch size = 1 的一个模型。

下面介绍一些 PyTorch 分布式编程的基础概念。

## 基本概念

在 16 张显卡，16 的并行数下，DDP 会同时启动 16 个进程。下面介绍一些分布式的概念。

**group**

即进程组。默认情况下，只有一个组。这个可以先不管，一直用默认的就行。

**world size**

表示全局的并行数，简单来讲，就是 2x8=16。

```python
# 获取world size，在不同进程里都是一样的，得到16
torch.distributed.get_world_size()
```

**rank**

表现当前进程的序号，用于进程间通讯。对于 16 的 world sizel 来说，就是 0,1,2,…,15。 
注意：rank=0 的进程就是 master 进程。

```python
# 获取rank，每个进程都有自己的序号，各不相同
torch.distributed.get_rank()
```

**local_rank**

又一个序号。这是每台机子上的进程的序号。机器一上有 0,1,2,3,4,5,6,7，机器二上也有 0,1,2,3,4,5,6,7

```python
# 获取local_rank。一般情况下，你需要用这个local_rank来手动设置当前模型是跑在当前机器的哪块GPU上面的。
torch.distributed.local_rank()
```

## 如何在 PyTorch 中使用 DDP：详细流程

### 精髓

DDP 的使用非常简单，因为它不需要修改你网络的配置。其精髓只有一句话

```python
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

原本的 model 就是你的 PyTorch 模型，新得到的 model，就是你的 DDP 模型。  
最重要的是，后续的模型关于前向传播、后向传播的用法，和原来完全一致！DDP 把分布式训练的细节都隐藏起来了，不需要暴露给用户，非常优雅！

### 准备工作

但是，在套 `model = DDP(model)` 之前，我们还是需要做一番准备功夫，把环境准备好的。  

这里需要注意的是，我们的程序虽然会在 16 个进程上跑起来，但是它们跑的是同一份代码，所以在写程序的时候要处理好不同进程的关系。

```python
## main.py文件
import torch
import argparse

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 新增2：从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数，后面还会介绍。所以不用考虑太多，照着抄就是了。
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增3：DDP backend初始化
#   a.根据local_rank来设定当前使用哪块GPU
torch.cuda.set_device(local_rank)
#   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
dist.init_process_group(backend='nccl')

# 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
#       如果要加载模型，也必须在这里做哦。
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# 可能的load模型...

# 新增5：之后才是初始化DDP模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### 前向与后向传播

有一个很重要的概念，就是数据的并行化。我们知道，DDP 同时起了很多个进程，但是他们用的是同一份数据，那么就会有数据上的冗余性。也就是说，你平时一个 epoch 如果是一万份数据，现在就要变成 1*16=16 万份数据了。

那么，我们需要使用一个特殊的 sampler，来使得各个进程上的数据各不相同，进而让一个 epoch 还是 1 万份数据。幸福的是，DDP 也帮我们做好了！

```python
my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
# 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
#       sampler的原理，后面也会介绍。
train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
# 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)


for epoch in range(num_epochs):
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.step()
```

### 其他需要注意的地方

- 保存参数

```python
# 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
#    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
# 2. 我只需要在进程0上保存一次就行了，避免多次保存重复的东西。
if dist.get_rank() == 0:
    torch.save(model.module, "saved_model.ckpt")
```

理论上，在没有 buffer 参数（如 BN）的情况下，DDP 性能和单卡 Gradient Accumulation 性能是完全一致的。

- 并行度为 8 的 DDP 等于 Gradient Accumulation Step 为 8 的单卡
- 速度上，DDP 当然比 Graident Accumulation 的单卡快；
- 如果要对齐性能，需要确保喂进去的数据，在 DDP 下和在单卡 Gradient Accumulation 下是一致的。
- 这个说起来简单，但对于复杂模型，可能是相当困难的。

### 调用方式

像我们在 QuickStart 里面看到的，DDP 模型下，python 源代码的调用方式和原来的不一样了。现在，需要用 `torch.distributed.launch` 来启动训练。

在这里，我们给出分布式训练的**重要参数**：

- --nnodes：有多少台机器？
- --node_rank：当前是哪台机器？
- --nproc_per_node：每台机器有多少个进程？
- 高级参数（多机模式才会用到）
  - 通讯的 address
  - 通讯的 port

### 实现方式

我们需要在每一台机子（总共 m 台）上都运行一次 `torch.distributed.launch`，每个 `torch.distributed.launch` 会启动 n 个进程，并给每个进程一个 `--local_rank=i` 的参数，这就是之前需要 "新增: 从外面得到 local_rank 参数" 的原因。这样我们就得到 n×m 个进程，world_size=n×m。

**单机模式**

```python
## Bash运行
# 假设我们只在一台机器上运行，可用卡数是8
python -m torch.distributed.launch --nproc_per_node 8 main.py
```

**多机模式**

复习一下，master 进程就是 rank=0 的进程。  在使用多机模式前，需要介绍两个参数：

- `--master_address`：通讯的 address
  - 也就是 master 进程的网络地址
  - 默认是：127.0.0.1，只能用于单机。
- `--master_port`：通讯的 port
  - 也就是 master 进程的一个端口，要先确认这个端口没有被其他程序占用了哦。一般情况下用默认的就行
  - 默认是：29500

```python
## Bash运行
# 假设我们在2台机器上运行，每台可用卡数是8
#    机器1：
python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 \
  --master_adderss $my_address --master_port $my_port main.py
#    机器2：
python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 \
  --master_adderss $my_address --master_port $my_port main.py
```

**小技巧**

```python
# 假设我们只用4,5,6,7号卡
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 main.py
# 假如我们还有另外一个实验要跑，也就是同时跑两个不同实验。
#    这时，为避免master_port冲突，我们需要指定一个新的。这里我随便敲了一个。
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 53453 main.py
```

### mp.spawn 调用方式

PyTorch 引入了 torch.multiprocessing.spawn，可以使得单卡、DDP 下的外部调用一致，即不用使用 torch.distributed.launch。 python main.py 一句话搞定 DDP 模式。

mp.spawn 的文档：[代码文档](https://pytorch.org/docs/stable/_modules/torch/multiprocessing/spawn.html#spawn)

下面给一个简单的 demo：

```python
def demo_fn(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # lots of code.
    ...

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

**mp.spawn 与 launch 各有利弊，请按照自己的情况选用。**  

按照经验，如果算法程序是提供给别人用的，那么 mp.spawn 更方便，因为不用解释 launch 的用法；但是如果是自己使用，launch 更有利，因为你的内部程序会更简单，支持单卡、多卡 DDP 模式也更简单。

## 总结

最后让我们来总结一下所有的代码，这份是一份能直接跑的代码，**推荐收藏！**

```python
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```