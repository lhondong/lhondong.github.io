
## 1. Data Preprocess

- Imagenet1k, 1k 个分类，每个分类大概 1300 张图片，一共一百万张图片。

### 1.1 Image2tensor

- RGB 3 channels
- PIL.Image.open + convert("RGB") 或者 torchvision.datasets.ImageFolder
- shape: (C, H, W), dtype: uint8 (unsigned integer 8 bit)
- 黑色 #000000，白色 #FFFFFF

### 1.2 Augmentation 数据增广

- Crop/Resize/Flip(500×500 to 224×224)

### 1.3 Convert

- torchvision.transform.ToPILImage
- torchvision.transform.PILToTensor()
- 将整数转换为 [0, 1] 浮点数

### 1.4 Normalize 归一化

- (image-mean)/std, global-level
- ImageNet1k: mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]

## 2. Model

### 2.1 Encoder

- Image2patch2embedding
- add position embedding (sin/cos)
- random masking (shuffle)
- class token
- Transformer Block (ViT-base/ViT-large/ViT-huge)

### 2.2 Decoder

- Projection_layer
- unshuffle
- Position embedding
- Transformer Blocks (shallow)
- Regression layer
- MSE loss function (norm pixel)

### 2.3 Forward functions

- Forward Encoder
- Forward Decoder
- Forward Loss(MSE loss)

## 3. training

### 3.1 Dataset

- torchvision.datasets.ImageFolder
- (x, y) 元组构成的生成器

### 3.2 Data_loader

- 从 Dataset 中取一个个元素，以特定的方式拼成一个个 mini batch

### 3.3 Model

- 实例化 model

### 3.4 Optimizer

- 实例化优化器

### 3.5 Load_model

- model.state_dict()
- optimizer.state_dict()
- epoch

### 3.6 train_one_epoch

### 3.7 save_model

- model.state_dict()
- optimizer.state_dict()
- epoch/loss
- config

## 4. Finetuning

- strong augmentation
- build encoder + BN + MLP classifier head
- interplate position embedding (预训练阶段与 finetune 阶段 patch 大小不同)
- load pre-trained model (**strict=False** 仅加载相同的层)
- update all parameter
- AdamW optimizer
- **label smoothing** Cross Entropy loss

## 5. Linear probing

- weak augmentation
- build encoder + BN(no affine) + MLP classifier head
- interpolate position embedding
- load pre-trained model (strict=False)
- only update parameters of MLP classifier head
- LARS optimizer
- Cross Entropy loss

## 6. Evaluation

- Regression 预训练阶段
  - MSE loss
  - PIL.show() 查看图片
- 分类任务
  - with torch.no_grad() -> efficient
  - model.eval() -> training=False, accurate BN/dropout
  - top1/top5