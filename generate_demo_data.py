"""
Generate Synthetic Demo Dataset - Create Fake Images And Masks For Testing Pipeline
This Script Makes Some Random Circle Pictures To Test If Everything Works Correctly
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_dataset():
    """Create Some Fake Medical Images With Circles As Organs"""
    
    # Make Directory Structure - Prepare Folders For Data
    dirs = [
        './data/train/images',
        './data/train/masks', 
        './data/val/images',
        './data/val/masks'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    print("\nGenerating synthetic images...")
    
    # Generate Training Samples - Make 20 Fake Images For Practice
    for i in range(20):
        img, mask = generate_sample()
        
        # Save Image File - Store Picture On Disk
        img.save(f'./data/train/images/sample_{i:03d}.png')
        mask.save(f'./data/train/masks/sample_{i:03d}.png')
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/20 training samples...")
    
    # Generate Validation Samples - Make 5 Fake Images For Checking During Training  
    print("\nGenerating validation samples...")
    for i in range(5):
        img, mask = generate_sample()
        
        img.save(f'./data/val/images/sample_{i:03d}.png')
        mask.save(f'./data/val/masks/sample_{i:03d}.png')
    
    print("\n✅ Done! Created synthetic dataset:")
    print("  - 20 training samples in ./data/train/")
    print("  - 5 validation samples in ./data/val/")
    print("\nNow you can run: python prepare_data.py --action check")
    print("Then run: python train.py to start learning!")


def generate_sample():
    """Make One Random Image With Colored Circles Inside"""
    
    # Create Blank Canvas - White Background Picture
    size = 256
    img_array = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Add Some Noise - Make Picture Look More Real Like Medical Scan
    noise = np.random.normal(0, 10, (size, size, 3))
    img_array = np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create Empty Mask - Black Background For Labels
    mask_array = np.zeros((size, size), dtype=np.uint8)
    
    # Draw Random Circles As "Organs" - Different Sizes And Positions
    draw_img = Image.fromarray(img_array)
    draw_mask = Image.fromarray(mask_array)
    
    num_circles = random.randint(2, 5)  # How Many Circles To Draw
    
    for _ in range(num_circles):
        # Pick Random Color For This Organ/Tissue
        color = tuple(random.randint(50, 200) for _ in range(3))
        
        # Pick Random Position And Size
        center_x = random.randint(30, size - 30)
        center_y = random.randint(30, size - 30)
        radius = random.randint(15, 40)
        
        # Draw Circle On Image - This Is The Visible Part
        draw = ImageDraw.Draw(draw_img)
        ellipse_bbox = [center_x - radius, center_y - radius, 
                       center_x + radius, center_y + radius]
        draw.ellipse(ellipse_bbox, fill=color)
        
        # Draw Same Circle On Mask - This Is The Label/Ground Truth
        # Value 1 Means "Organ Present Here"
        draw_m = ImageDraw.Draw(draw_mask)
        draw_m.ellipse(ellipse_bbox, fill=1)
    
    return draw_img.convert('RGB'), draw_mask


if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC DATASET GENERATOR - FAKE MEDICAL IMAGE MAKER")
    print("=" * 60)
    print()
    create_synthetic_dataset()
