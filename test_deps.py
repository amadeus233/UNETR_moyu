#!/usr/bin/env python3
"""Test script to verify all dependencies are working correctly."""

def test_dependencies():
    """Test importing and basic functionality of all required packages."""
    print("=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    # Test torch
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        # Basic tensor operation
        x = torch.tensor([1, 2, 3])
        assert x.shape == (3,), "Tensor shape mismatch"
    except Exception as e:
        print(f"✗ torch failed: {e}")
        return False
    
    # Test torchvision
    try:
        import torchvision
        print(f"✓ torchvision: {torchvision.__version__}")
    except Exception as e:
        print(f"✗ torchvision failed: {e}")
        return False
    
    # Test PIL/Pillow
    try:
        from PIL import Image
        print(f"✓ Pillow: OK")
        # Create a simple image
        img = Image.new('RGB', (100, 100), color='red')
        assert img.size == (100, 100), "Image size mismatch"
    except Exception as e:
        print(f"✗ Pillow failed: {e}")
        return False
    
    # Test numpy
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,), "Array shape mismatch"
    except Exception as e:
        print(f"✗ numpy failed: {e}")
        return False
    
    # Test matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print(f"✓ matplotlib: {matplotlib.__version__}")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
    except Exception as e:
        print(f"✗ matplotlib failed: {e}")
        return False
    
    print("=" * 60)
    print("All dependencies tested successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_dependencies()
    exit(0 if success else 1)
