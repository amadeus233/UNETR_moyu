"""
Training Script For UNETR-2D Medical Image Segmentation Project
Main Program To Learn How Segment Organs From Scanned Images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import UNETR


class MedicalImageDataset(Dataset):
    """Custom Made Dataset Class For Handle Medical Image Segmentation Task"""
    
    def __init__(self, images_dir, masks_dir=None, transform=None, augment=False):
        """
        Initialize Parameters:
            images_dir: Folder Path Contain Input Training Images
            masks_dir: Folder Path Contain Ground Truth Masks (If None Means Only Inference)
            transform: What Kind Of Changes Apply To Images
            augment: True Or False Use Data Augmentation Technique
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        
        # Get List Of All Image Files Name - Sort Them In Order
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        if masks_dir and os.path.exists(masks_dir):
            self.mask_files = sorted([f for f in os.listdir(masks_dir) 
                                      if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            assert len(self.image_files) == len(self.mask_files), \
                "Number Of Images And Masks Must Be Same Otherwise Error"
        else:
            self.mask_files = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load Image From Disk - Open File Using Pillow Library
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Load Mask If Available - Grayscale Black White Image
        if self.mask_files:
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            mask = Image.open(mask_path).convert("L")  # Change To Gray Scale Mode
        else:
            mask = None
        
        # Data Augmentation Strategy - Make More Training Samples Artificially
        if self.augment and mask is not None:
            # Random Horizontal Flip - Mirror Left Right With 50% Chance
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random Vertical Flip - Mirror Up Down With 50% Chance
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Random Rotation By 90 Degree Steps - Rotate K Times Quarter Circle
            k = np.random.randint(0, 4)
            image = image.rotate(k * 90, expand=False)
            mask = mask.rotate(k * 90, expand=False)
        
        # Apply Transformation Functions - Resize Normalize Convert To Tensor
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)
        
        if mask is not None:
            return image, mask
        else:
            return image, self.image_files[idx]


def dice_loss(pred, target, smooth=1e-6):
    """Calculate Dice Loss Function - Measure Overlap Between Prediction And Real Answer"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    """Combine Two Loss Types Together - Binary Cross Entropy Plus Dice Loss Weighted Sum"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train Model For One Complete Epoch - Loop Through All Batches Once"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate Performance On Unseen Data - No Gradient Calculation During Testing Phase"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Calculate Dice Coefficient Score - Evaluate Quality Metric
        preds = torch.sigmoid(outputs)
        intersection = (preds * masks).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice = (2.0 * intersection) / (union + 1e-6)
        
        total_loss += loss.item()
        total_dice += dice.mean().item()
    
    num_batches = len(dataloader)
    return total_loss / num_batches, total_dice / num_batches


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    torch.save(state, filename)
    if is_best:
        best_filename = 'model_best.pth.tar'
        torch.save(state['state_dict'], best_filename)
        print(f"=> Saved best model to {best_filename}")


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score
    if history.get('val_dice'):
        axes[1].plot(history['val_dice'], label='Validation Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Performance')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def train(config):
    """Main Training Function Entry Point - Start Whole Training Process From Here"""
    
    # Setup compute device choice cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary folder directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    
    # Define data transformation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    # Create dataset objects for loading data
    print("Loading datasets...")
    train_dataset = MedicalImageDataset(
        images_dir=config['train_images_dir'],
        masks_dir=config['train_masks_dir'],
        transform=train_transform,
        augment=config.get('augment', True)
    )
    
    val_dataset = MedicalImageDataset(
        images_dir=config['val_images_dir'],
        masks_dir=config['val_masks_dir'],
        transform=val_transform,
        augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloader iterators for batch processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize neural network model structure
    print("Initializing model...")
    model_config = {
        "image_height": config['image_size'],
        "image_width": config['image_size'],
        "num_channels": 3,
        "patch_height": config['patch_size'],
        "patch_width": config['patch_size'],
        "num_patches": (config['image_size'] ** 2) // (config['patch_size'] ** 2),
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "dropout_rate": 0.3
    }
    
    model = UNETR(model_config).to(device)
    
    # Count how many trainable parameters exist in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Define loss function and optimization algorithm
    criterion = lambda pred, target: combined_loss(pred, target, 
                                                    bce_weight=0.5, 
                                                    dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop process iterate over epochs
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_dice = 0.0
    early_stop_counter = 0
    
    print(f"\nStarting training for {config['epochs']} epochs...\n")
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train one complete pass through all training data
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate performance on held-out test set
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Update learning rate based on validation performance
        scheduler.step(val_loss)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{config['epochs']} completed:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {val_dice:.4f}")
        print(f"  Time: {elapsed_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save checkpoint file to disk
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_val_dice,
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best, 
           os.path.join(config['output_dir'], 'checkpoints', f'checkpoint_ep{epoch}.pth.tar'))
        
        # Early stopping mechanism stop if no improvement
        if early_stop_counter >= config.get('early_stopping_patience', 20):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Plot history visualization every 10 epochs
        if epoch % 10 == 0 or epoch == config['epochs']:
            plot_training_history(history, 
                                os.path.join(config['output_dir'], 'training_history.png'))
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    print(f"Final model saved to: {os.path.join(config['output_dir'], 'model_best.pth.tar')}")
    
    return model, history


if __name__ == "__main__":
    # Configuration Settings Dictionary
    config = {
        # Data paths folder locations
        'train_images_dir': './data/train/images',
        'train_masks_dir': './data/train/masks',
        'val_images_dir': './data/val/images',
        'val_masks_dir': './data/val/masks',
        
        # Model params architecture settings
        'image_size': 256,
        'patch_size': 16,
        
        # Training params hyperparameters
        'batch_size': 4,
        'epochs': 100,
        'learning_rate': 1e-4,
        'augment': True,
        'early_stopping_patience': 20,
        
        # Output directory where save results
        'output_dir': './outputs',
    }
    
    # Check if data directories exist before starting
    if not all(os.path.exists(path) for path in [
        config['train_images_dir'], config['train_masks_dir'],
        config['val_images_dir'], config['val_masks_dir']
    ]):
        print("="*60)
        print("ERROR: Data directories not found!")
        print("="*60)
        print("\nTo train the model, please organize your data as follows:")
        print("""
        ./data/
          ├── train/
          │   ├── images/     # Training images (*.png, *.jpg)
          │   └── masks/      # Corresponding segmentation masks
          └── val/
              ├── images/     # Validation images
              └── masks/      # Corresponding masks
        """)
        print("Each image should have a corresponding mask with the same filename.")
        print("="*60)
        exit(1)
    
    # Run training procedure
    model, history = train(config)
