# UNETR-2D 训练指南

本指南将帮助你从头开始训练 UNETR-2D 医学图像分割模型。

## 📋 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [开始训练](#开始训练)
4. [监控训练](#监控训练)
5. [使用训练好的模型](#使用训练好的模型)
6. [常见问题](#常见问题)

---

## 🔧 环境准备

### 已安装的依赖
```bash
torch==2.8.0
torchvision==0.23.0
Pillow>=11.3.0
numpy>=2.0.2
matplotlib>=3.9.4
tqdm>=4.65.0
```

如果还没安装，运行：
```bash
pip install -r requirements.txt
```

---

## 📁 数据准备

### 1. 创建数据结构

运行以下命令自动创建所需目录：
```bash
python prepare_data.py --action prepare
```

这将创建以下结构：
```
./data/
├── train/
│   ├── images/     # 放置训练图像
│   └── masks/      # 放置对应的分割掩码
└── val/
    ├── images/     # 放置验证图像
    └── masks/      # 放置对应的分割掩码
```

### 2. 准备你的数据

**重要要求：**
- 每张图像必须有对应名称的掩码文件
- 支持的格式：PNG, JPG, JPEG, BMP
- 图像应为 RGB 格式（3 通道）
- 掩码应为灰度图（单通道），目标区域为白色 (255)，背景为黑色 (0)

**示例：**
```
data/train/images/image001.png  →  data/train/masks/image001.png
data/train/images/image002.png  →  data/train/masks/image002.png
...
```

### 3. 检查数据集

准备好数据后，运行：
```bash
python prepare_data.py --action check
```

这会验证：
- ✓ 所有目录是否存在
- ✓ 图像和掩码数量是否匹配
- ✓ 文件名是否正确对应

---

## 🚀 开始训练

### 基础训练

直接运行默认配置：
```bash
python train.py
```

### 自定义配置

编辑 [`train.py`](train.py)底部的`config`字典：

```python
config = {
    # 数据路径
    'train_images_dir': './data/train/images',
    'train_masks_dir': './data/train/masks',
    'val_images_dir': './data/val/images',
    'val_masks_dir': './data/val/masks',
    
    # 模型参数
    'image_size': 256,        # 输入图像尺寸
    'patch_size': 16,         # Patch 大小
    
    # 训练参数
    'batch_size': 4,          # 批次大小
    'epochs': 100,            # 训练轮数
    'learning_rate': 1e-4,    # 学习率
    'augment': True,          # 数据增强
    'early_stopping_patience': 20,  # 早停耐心值
    
    # 输出目录
    'output_dir': './outputs',
}
```

### 推荐配置

根据你的硬件调整：

**GPU用户（显存充足）：**
```python
'batch_size': 8  # 或更大
```

**CPU 用户或小显存 GPU：**
```python
'batch_size': 2  # 减小批次
'image_size': 128  # 降低分辨率
```

**小数据集 (< 100 样本):**
```python
'epochs': 200
'early_stopping_patience': 30
'learning_rate': 5e-5
```

---

## 📊 监控训练

### 实时输出

训练时会看到类似输出：
```
Epoch 1/100 completed:
  Train Loss: 0.7234
  Val Loss: 0.6891
  Val Dice: 0.4521
  Time: 45.2s
  LR: 0.000100
```

### 可视化历史

每 10 个 epoch 会自动保存训练曲线到：
```
./outputs/training_history.png
```

包含：
- 训练损失和验证损失曲线
- 验证集 Dice 分数变化

### Checkpoint 保存

每个 epoch 都会保存 checkpoint：
```
./outputs/checkpoints/checkpoint_ep{epoch}.pth.tar
```

最佳模型会单独保存：
```
./outputs/model_best.pth.tar
```

---

## 🎯 使用训练好的模型

Load trained weights for inference:

```python
import torch
from model import UNETR

# 初始化模型
model_config = {
    "image_height": 256,
    "image_width": 256,
    "num_channels": 3,
    "patch_height": 16,
    "patch_width": 16,
    "num_patches": 256,
    "num_layers": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "dropout_rate": 0.3
}

model = UNETR(model_config)

# 加载训练好的权重
checkpoint = torch.load('./outputs/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print("Model loaded successfully!")
print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")
```

然后在 [`model.py`](d:\AAA_test\UNETR-2D-Medical-Image-Segmentation\model.py) 中使用即可。

---

## ❓ 常见问题

### Q1: CUDA Out of Memory
**解决方案：**
- 减小 `batch_size`
- 降低`image_size`
- 使用梯度累积

### Q2: 训练不收敛
**可能原因：**
- 学习率太高 → 尝试 `1e-5`
- 数据质量问题 → 检查标注准确性
- 数据量太少 → 增加数据增强或使用迁移学习

### Q3: Dice 分数很低 (< 0.5)
**建议：**
- 增加训练轮数
- 检查数据和掩码是否对齐
- 确认掩码是二值的（目标和背景）
- 尝试不同的损失函数权重

### Q4: 如何加速训练？
**方法：**
- 使用 GPU（强烈推荐）
- 启用混合精度训练（AMP）
- 增加`num_workers`（多进程数据加载）
- 减小模型尺寸（减少层数或隐藏维度）

---

## 📈 性能基准

参考指标（因数据集而异）：

| 数据类型 | 样本数 | Epochs | Expected Dice |
|---------|--------|--------|---------------|
| 脑部 MRI | 500+ | 100 | 0.85+ |
| 肝脏 CT | 200+ | 150 | 0.90+ |
| 肺部 X-ray | 1000+ | 80 | 0.88+ |
| 小型数据集 | 50-100 | 200 | 0.70-0.80 |

---

## 💡 进阶技巧

### 1. 数据增强扩展

在 `MedicalImageDataset`类中可以添加更多增强：
- 随机旋转（任意角度）
- 弹性形变
- 亮度和对比度调整
- 高斯噪声

### 2. 学习率调度器

当前使用的是 ReduceLROnPlateau，也可以尝试：
- CosineAnnealingLR
- OneCycleLR
- Warmup + Decay

### 3. 损失函数实验

可以尝试：
- Focal Loss（处理类别不平衡）
- Tversky Loss
- Combo Loss（不同权重的组合）

---

## 🆘 需要帮助？

遇到问题时，请提供：
1. 错误信息的完整截图
2. 你的数据集规模
3. 使用的硬件配置（CPU/GPU、内存）
4. 当前的配置文件

祝训练顺利！🎉
