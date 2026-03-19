"""
Data preparation script for UNETR-2D training
Helps organize your medical image dataset into the correct structure
"""

import os
import shutil
from pathlib import Path


def prepare_data_structure(base_dir='./data'):
    """Create the required directory structure"""
    
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'masks'),
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'val', 'masks'),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print("\n✓ Data directory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training images in: ./data/train/images/")
    print("2. Place corresponding masks in: ./data/train/masks/")
    print("3. Place validation images in: ./data/val/images/")
    print("4. Place corresponding masks in: ./data/val/masks/")
    print("\nImportant: Each image must have a mask with the SAME filename!")
    print("Example: image001.png -> mask001.png")


def check_dataset(data_dir='./data'):
    """Check if dataset is properly organized"""
    
    print("Checking dataset organization...\n")
    
    splits = ['train', 'val']
    all_good = True
    
    for split in splits:
        images_dir = os.path.join(data_dir, split, 'images')
        masks_dir = os.path.join(data_dir, split, 'masks')
        
        print(f"\n{split.upper()} SET:")
        print("-" * 50)
        
        if not os.path.exists(images_dir):
            print(f"❌ Missing: {images_dir}")
            all_good = False
            continue
            
        if not os.path.exists(masks_dir):
            print(f"❌ Missing: {masks_dir}")
            all_good = False
            continue
        
        images = sorted([f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        masks = sorted([f for f in os.listdir(masks_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print(f"Images found: {len(images)}")
        print(f"Masks found: {len(masks)}")
        
        if len(images) != len(masks):
            print(f"⚠️  WARNING: Number of images ({len(images)}) != masks ({len(masks)})")
            all_good = False
        
        # Check matching filenames
        mismatches = []
        for img_file in images:
            if img_file not in masks:
                mismatches.append(img_file)
        
        if mismatches:
            print(f"⚠️  WARNING: {len(mismatches)} images without matching masks:")
            for m in mismatches[:5]:  # Show first 5
                print(f"   - {m}")
            if len(mismatches) > 5:
                print(f"   ... and {len(mismatches) - 5} more")
            all_good = False
        else:
            print("✓ All images have corresponding masks")
    
    print("\n" + "="*60)
    if all_good and len(images) > 0:
        print("✅ Dataset looks good! Ready for training.")
    else:
        print("⚠️  Please fix the issues above before training.")
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare/check dataset for UNETR-2D training')
    parser.add_argument('--action', choices=['prepare', 'check'], default='check',
                       help='Action to perform: prepare (create dirs) or check (validate data)')
    parser.add_argument('--data-dir', default='./data',
                       help='Base directory for data (default: ./data)')
    
    args = parser.parse_args()
    
    if args.action == 'prepare':
        prepare_data_structure(args.data_dir)
    elif args.action == 'check':
        check_dataset(args.data_dir)
