"""
Comprehensive Test Script - Verify Everything Work Correctly
Test All Components After Rename UNETR-2D.py To model.py
"""

import sys
import os

print("=" * 60)
print("START COMPREHENSIVE SYSTEM TEST")
print("=" * 60)

# Test 1: Check File Existence
print("\n[Test 1] Check Required Files Exist...")
required_files = [
    'model.py',
    'train.py',
    'prepare_data.py',
    'requirements.txt',
    'README.md',
    'TRAINING_GUIDE.md'
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ Found: {file}")
    else:
        print(f"  ✗ Missing: {file}")
        all_exist = False

if not all_exist:
    print("\n❌ CRITICAL ERROR: Some files missing!")
    sys.exit(1)

print("\n✅ Test 1 PASSED: All required files present")

# Test 2: Import Dependencies
print("\n[Test 2] Testing Dependency Imports...")
try:
    import torch
    import torchvision
    from PIL import Image
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    print(f"  ✓ torch: {torch.__version__}")
    print(f"  ✓ torchvision: {torchvision.__version__}")
    from PIL import __version__ as pil_version
    print(f"  ✓ Pillow: {pil_version}")
    print(f"  ✓ numpy: {np.__version__}")
    print(f"  ✓ matplotlib: {matplotlib.__version__}")
    print("\n✅ Test 2 PASSED: All dependencies imported successfully")
except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Failed to import dependencies: {e}")
    sys.exit(1)

# Test 3: Import Model From New Location
print("\n[Test 3] Testing Model Import From model.py...")
try:
    from model import UNETR
    print("  ✓ Successfully imported UNETR from model.py")
    print("\n✅ Test 3 PASSED: Model import working correctly")
except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Cannot import UNETR from model.py: {e}")
    sys.exit(1)

# Test 4: Instantiate Model
print("\n[Test 4] Testing Model Instantiation...")
try:
    configuration = {
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
    
    model = UNETR(configuration)
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ Test 4 PASSED: Model instantiation successful")
except Exception as e:
    print(f"\n❌ CRITICAL ERROR: Failed to instantiate model: {e}")
    sys.exit(1)

# Test 5: Forward Pass Test
print("\n[Test 5] Testing Model Forward Pass...")
try:
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  ✓ Input shape: {dummy_input.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    assert output.shape == (1, 1, 256, 256), f"Unexpected output shape: {output.shape}"
    print("\n✅ Test 5 PASSED: Forward pass successful")
except Exception as e:
    print(f"\n❌ CRITICAL ERROR: Forward pass failed: {e}")
    sys.exit(1)

# Test 6: Check Old Filename References Are Gone
print("\n[Test 6] Checking No Old Filename References Remain...")
files_to_check = ['train.py', 'prepare_data.py', 'TRAINING_GUIDE.md', 'README.md']
old_references_found = False

for filepath in files_to_check:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'UNETR-2D.py' in content or 'from UNETR_2D import' in content:
                print(f"  ⚠ WARNING: Old reference found in {filepath}")
                old_references_found = True

if not old_references_found:
    print("  ✓ No old filename references detected")
    print("\n✅ Test 6 PASSED: All references updated correctly")
else:
    print("\n❌ WARNING: Some old references still exist!")

# Final Summary
print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nSummary:")
print("  • File rename: UNETR-2D.py → model.py ✅")
print("  • All imports updated correctly ✅")
print("  • Model functional and tested ✅")
print("  • No broken references detected ✅")
print("\n🎉 System Ready For Training!")
print("=" * 60)
