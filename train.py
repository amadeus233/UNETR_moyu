"""
Training Script For UNETR-2D Medical Image Segmentation Project
Main Program To Learn How Segment Organs From Scanned Images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import re
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import UNETR


class MedicalImageDataset(Dataset):
    """Custom Made Dataset Class For Handle Medical Image Segmentation Task"""
    
    def __init__(self, images_dir, masks_dir=None, transform=None, augment=False, binary_mode=False):
        """
        Initialize Parameters:
            images_dir: Folder Path Contain Input Training Images
            masks_dir: Folder Path Contain Ground Truth Masks (If None Means Only Inference)
            transform: What Kind Of Changes Apply To Images
            augment: True Or False Use Data Augmentation Technique
            binary_mode: Convert Multi-Class Mask To Binary (0=Background, 1=Foreground)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        self.binary_mode = binary_mode
        
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
        
        # Data augmentation strategy - ALL DISABLED per fix plan
        # Skip rotation, crop, deformation, normalization variations
        # Only basic flip for minimal variation if absolutely needed
        if self.augment and mask is not None:
            # Minimal augmentation: horizontal flip only
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Verify image and mask dimensions match exactly
        if mask is not None:
            if image.size != mask.size:
                print(f"WARNING: Image {img_path} size {image.size} != mask {mask_path} size {mask.size}")
                # Force resize mask to match image dimensions using NEAREST interpolation
                mask = mask.resize((image.width, image.height), Image.NEAREST)
        
        # Apply transformation functions - resize normalize convert to tensor
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)
                
                # Convert multi-class mask to binary if needed - KEY FIX: any non-zero is foreground
                if self.binary_mode:
                    # CRITICAL: All non-zero values become 1 (handles multi-label masks)
                    mask = (mask > 0).float()  # Changed from 0.5 to 0 to catch all foreground
                
                mask = mask.squeeze(0)  # Remove channel dimension: [1,H,W] -> [H,W]
        else:
            # Even without transform, convert to tensor
            image = transforms.ToTensor()(image)
            if mask is not None:
                mask = transforms.ToTensor()(mask)
                # CRITICAL FIX: Binarize mask - any non-zero pixel becomes foreground (1)
                if self.binary_mode:
                    mask = (mask > 0).float()
                    mask = (mask > 0.5).float()
                mask = mask.squeeze(0)
        
        if mask is not None:
            return image, mask
        else:
            return image, self.image_files[idx]


def dice_loss(pred, target, smooth=1e-6):
    """Calculate Dice Loss Function - Measure Overlap Between Prediction And Real Answer
    
    Args:
        pred: Model predictions with shape [B, 1, H, W]
        target: Ground truth masks with shape [B, H, W] or [B, 1, H, W]
    """
    # Ensure target has same shape as pred: [B, H, W] -> [B, 1, H, W]
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
    """Combine Two Loss Types Together - Binary Cross Entropy Plus Dice Loss Weighted Sum
    
    Args:
        pred: Model predictions with shape [B, 1, H, W]
        target: Ground truth masks with shape [B, H, W] or [B, 1, H, W]
        pos_weight: Weight for positive class in BCE (auto-calculated if None to address class imbalance)
    """
    # Ensure target has same shape as pred: [B, H, W] -> [B, 1, H, W]
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    
    # Auto-calculate pos_weight per batch if not provided - KEY FIX for class imbalance
    if pos_weight is None:
        pos_pixels = (target > 0).float().sum()
        neg_pixels = (target == 0).float().sum()
        batch_pos_weight = (neg_pixels / (pos_pixels + 1e-8)).clamp(max=500)  # Cap at 500
    else:
        batch_pos_weight = torch.tensor([pos_weight]).to(pred.device)
    
    # Use weighted BCE to penalize missing foreground more heavily
    bce = nn.BCEWithLogitsLoss(pos_weight=batch_pos_weight)(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    """Train Model For One Complete Epoch - Loop Through All Batches Once"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Step scheduler per batch if using ReduceLROnPlateau (pass metric later)
        # Scheduler will be stepped after epoch with val_dice metric
        
        total_loss += loss.item()
        
        # Calculate Dice for monitoring
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            masks_for_dice = masks.unsqueeze(1) if masks.dim() == 3 else masks
            intersection = (preds * masks_for_dice).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks_for_dice.sum(dim=(2, 3))
            dice = (2.0 * intersection) / (union + 1e-6)
            total_dice += dice.mean().item()
        
        avg_loss = total_loss / (batch_idx + 1)
        avg_dice = total_dice / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'dice': f'{avg_dice:.4f}'})
    
    return avg_loss, avg_dice


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
        
        # Ensure masks has channel dimension: [B, H, W] -> [B, 1, H, W]
        masks_for_dice = masks.unsqueeze(1) if masks.dim() == 3 else masks
        
        intersection = (preds * masks_for_dice).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + masks_for_dice.sum(dim=(2, 3))
        dice = (2.0 * intersection) / (union + 1e-6)
        
        total_loss += loss.item()
        total_dice += dice.mean().item()
    
    num_batches = len(dataloader)
    return total_loss / num_batches, total_dice / num_batches


def cleanup_old_checkpoints(output_dir, current_epoch, keep_last_n=5):
    """Remove old checkpoint files, keeping only the last n checkpoints"""
    checkpoint_pattern = os.path.join(output_dir, 'checkpoint_ep*.pth.tar')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by epoch number and remove oldest ones
    def get_epoch(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'checkpoint_ep(\d+)\.pth\.tar', basename)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_epoch)
    
    # Remove oldest checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            os.remove(checkpoint)
            print(f"  Cleaned up old checkpoint: {os.path.basename(checkpoint)}")
        except Exception as e:
            print(f"  Warning: Could not remove {checkpoint}: {e}")


def cleanup_old_checkpoints(output_dir, current_epoch, keep_last_n=5):
    """Remove old checkpoint files, keeping only the last n checkpoints"""
    checkpoint_pattern = os.path.join(output_dir, 'checkpoint_ep*.pth.tar')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by epoch number and remove oldest ones
    def get_epoch(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'checkpoint_ep(\d+)\.pth\.tar', basename)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_epoch)
    
    # Remove oldest checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            os.remove(checkpoint)
            print(f"  Cleaned up old checkpoint: {os.path.basename(checkpoint)}")
        except Exception as e:
            print(f"  Warning: Could not remove {checkpoint}: {e}")


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    """Save checkpoint to specified output directory"""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, 'model_best.pth.tar')
        torch.save(state['state_dict'], best_filepath)
        print(f"=> Saved best model to {best_filepath}")


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
        augment=config.get('augment', True),
        binary_mode=config.get('binary_mode', False)
    )
    
    val_dataset = MedicalImageDataset(
        images_dir=config['val_images_dir'],
        masks_dir=config['val_masks_dir'],
        transform=val_transform,
        augment=False,
        binary_mode=config.get('binary_mode', False)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloader iterators for batch processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Keep 0 for Windows compatibility
        pin_memory=False  # Disable for CPU training (avoids warning)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Also disable for validation
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
        "num_heads": 12,
        "mlp_dim": 3072,
        "dropout_rate": 0.3
    }
    
    model = UNETR(model_config).to(device)
    
    # Count how many trainable parameters exist in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # CRITICAL FIX: Initialize final convolutional layer bias to negative value
    # This addresses extreme class imbalance (foreground < 1%)
    # Default bias=0 causes sigmoid(0)=0.5, predicting 50% foreground
    # Negative bias makes model initially favor background
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels == 1:
            # Set bias so sigmoid(bias) ≈ 0.01 (1% foreground prior)
            # sigmoid(-4.6) ≈ 0.01
            module.bias.data.fill_(-4.6)
            print(f"Initialized {name} bias to -4.6 (prior: 1% foreground)")
            break
    
    # Define loss function and optimization algorithm
    # CRITICAL: pos_weight=500 to address extreme class imbalance (0.3-0.5% foreground)
    pos_weight_value = 500.0 if config.get('binary_mode', False) else 1.0
    print(f"\n{'='*60}")
    print(f"CONFIGURATION SUMMARY:")
    print(f"{'='*60}")
    print(f"Mode: {'Binary Segmentation' if config.get('binary_mode') else 'Multi-class'}")
    print(f"Pos Weight: {pos_weight_value}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Output Dir: {config['output_dir']}")
    print(f"{'='*60}\n")
    
    # CRITICAL: pos_weight=None means auto-calculate per batch based on class imbalance
    criterion = lambda pred, target: combined_loss(pred, target, 
                                                    bce_weight=0.5, 
                                                    dice_weight=0.5,
                                                    pos_weight=None)  # Auto-calculate!
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop process iterate over epochs
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_dice = 0.0
    early_stop_counter = 0
    start_epoch = 1
    
    # Check for existing checkpoint to resume training
    checkpoint_dir = config['output_dir']
    latest_checkpoint = None
    
    # Find the latest checkpoint file
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_ep') and f.endswith('.pth.tar')]
        if checkpoint_files:
            # Extract epoch numbers and find the latest
            epochs = []
            for f in checkpoint_files:
                try:
                    epoch_num = int(f.replace('checkpoint_ep', '').replace('.pth.tar', ''))
                    epochs.append((epoch_num, f))
                except ValueError:
                    continue
            
            if epochs:
                epochs.sort(reverse=True)
                latest_epoch, latest_checkpoint = epochs[0]
                print(f"\n{'='*60}")
                print(f"FOUND CHECKPOINT: {latest_checkpoint} (Epoch {latest_epoch})")
                print(f"{'='*60}")
                
                # Ask user if they want to resume
                response = input(f"\nResume training from epoch {latest_epoch}? (y/n): ").strip().lower()
                
                if response == 'y':
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    print(f"\nLoading checkpoint: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Restore model weights
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"✓ Restored model weights from epoch {latest_epoch}")
                    
                    # Restore optimizer state
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"✓ Restored optimizer state")
                    
                    # Restore best dice score
                    best_val_dice = checkpoint.get('best_dice', 0.0)
                    print(f"✓ Best Dice score: {best_val_dice:.4f}")
                    
                    # Set start epoch
                    start_epoch = latest_epoch + 1
                    print(f"\n✓ Training will resume from epoch {start_epoch}")
                    print(f"{'='*60}\n")
                    
                    # Load training history from previous epochs
                    history_file = os.path.join(checkpoint_dir, 'training_history.npz')
                    if os.path.exists(history_file):
                        hist_data = np.load(history_file)
                        history['train_loss'] = list(hist_data['train_loss'])
                        history['val_loss'] = list(hist_data['val_loss'])
                        history['val_dice'] = list(hist_data['val_dice'])
                        print(f"✓ Loaded training history ({len(history['train_loss'])} epochs)")
                    else:
                        print(f"! Warning: No history file found, starting fresh history")
                else:
                    print(f"\nStarting fresh training from epoch 1...\n")
            else:
                print(f"\nNo valid checkpoints found. Starting fresh training...\n")
        else:
            print(f"\nNo checkpoints found. Starting fresh training...\n")
    else:
        print(f"\nCreating new output directory: {checkpoint_dir}\n")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Show which images are being used - CRITICAL for transparency
    print(f"\n{'='*70}")
    print(f"Training Dataset Summary:")
    print(f"  Training images: {len(train_dataset)} samples")
    print(f"  Validation images: {len(val_dataset)} samples")
    print(f"\nTraining images (first 10):")
    for i, img_name in enumerate(train_dataset.image_files[:10], 1):
        mask_name = train_dataset.mask_files[i-1] if train_dataset.mask_files else "N/A"
        print(f"  [{i:2d}] Image: {img_name:25s} -> Mask: {mask_name}")
    if len(train_dataset.image_files) > 10:
        print(f"  ... and {len(train_dataset.image_files) - 10} more images")
    print(f"{'='*70}\n")
    
    print(f"Starting training from epoch {start_epoch} to epoch {config['epochs']}...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        start_time = time.time()
        
        # Train one complete pass through all training data
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate performance on held-out test set
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Update learning rate based on validation Dice (maximize Dice, not minimize loss)
        scheduler.step(val_dice)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{config['epochs']} completed:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"  Time: {elapsed_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save checkpoint file to disk
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save regular checkpoint
        checkpoint_filename = f'checkpoint_ep{epoch}.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_val_dice,
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best, config['output_dir'], checkpoint_filename)
        
        # Cleanup old checkpoints to save disk space (keep last 5)
        if epoch % 10 == 0:  # Only cleanup every 10 epochs to reduce I/O
            cleanup_old_checkpoints(config['output_dir'], epoch, keep_last_n=5)
        
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
        # Data paths folder locations - USING FULL DATASET
        'train_images_dir': './data/train/images',
        'train_masks_dir': './data/train/masks',
        'val_images_dir': './data/val/images',
        'val_masks_dir': './data/val/masks',
        
        # Model params architecture settings
        'image_size': 256,
        'patch_size': 16,
        
        # Training params hyperparameters
        'batch_size': 4,
        'epochs': 100,  # Full training with 100 epochs
        'learning_rate': 1e-3,  # Standard learning rate for good convergence
        'augment': False,  # DISABLED: No augmentation for cleaner training
        'early_stopping_patience': 15,  # Patient early stopping
        'binary_mode': True,  # Binary segmentation for tumor detection
        
        # Output directory where save results
        'output_dir': './outputs/full_dataset_v1',
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
