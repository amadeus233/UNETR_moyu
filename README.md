# UNETR-2D Medical Image Segmentation

基于 Vision Transformer 的医学图像分割模型 - 完整训练框架

## 🚀 特性 Features

- ✅ **完整训练流程** - 支持混合精度、早停、学习率调度
- ✅ **数据准备工具** - 自动组织数据集结构
- ✅ **模块化设计** - 清晰的代码结构和注释
- ✅ **详细文档** - 中英双语使用说明

## 📦 快速开始 Quick Start

### 1. 环境配置 Environment Setup

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 测试模型 Test Model

```bash
python test_complete.py
```

### 3. 准备数据集 Prepare Dataset

```bash
# 检查数据结构
python prepare_data.py --action check

# 如需整理自己的数据
python prepare_data.py --action organize --source <your_data_path>
```

### 4. 开始训练 Start Training

```bash
python train.py
```

## 🏗️ 项目结构 Project Structure

```
UNETR-2D-Medical-Image-Segmentation/
├── model.py              # UNETR-2D 模型架构
├── train.py              # 训练脚本（含验证、保存checkpoint）
├── prepare_data.py       # 数据准备与验证工具
├── test_complete.py      # 完整测试脚本
├── requirements.txt      # Python 依赖列表
├── data/                 # 示例数据目录
│   ├── train/           # 训练集 images/masks
│   └── val/             # 验证集 images/masks
└── outputs/             # 训练输出（tensorboard logs, checkpoints）
```

## ⚙️ 模型配置 Configuration

默认参数可在 [`model.py`](model.py) 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `img_size` | 256 | 输入图像尺寸 |
| `patch_size` | 16×16 | Patch大小 |
| `hidden_dim` | 768 | Transformer隐藏层维度 |
| `num_layers` | 12 | Transformer层数 |
| `num_heads` | 12 | 注意力头数 |

## 📈 训练技巧 Tips

1. **显存优化**: 启用 AMP 混合精度训练（已默认开启）
2. **防止过拟合**: 使用数据增强（旋转、翻转等）
3. **学习率调整**: CosineAnnealingLR 自动调度
4. **早停策略**: patience=10 epochs无改善则停止

详见 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)

## 🛠️ 常见问题 FAQ

**Q: CUDA out of memory?**  
A: 减小 `batch_size` 或使用梯度累积

**Q: 如何加载预训练权重？**  
A: `checkpoint = torch.load('model_best.pth.tar')`

**Q: 支持 GPU 吗？**  
A: 支持！CUDA 可用时自动切换

## 📄 License

MIT License

---

💡 **提示**: 本项目为教学和研究目的提供的基础实现。生产环境请根据具体任务调优。
