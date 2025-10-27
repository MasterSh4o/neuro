#!/usr/bin/env python3
"""
Debug script to identify dataset path and file issues.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def get_platform_paths():
    """
    Get platform-appropriate default paths.
    """
    if os.name == 'nt':  # Windows
        return {
            'home': str(Path.home()),
            'workspace': str(Path.home() / 'workspace' / 'NeuroInterf'),
            'datasets': str(Path.home() / 'datasets'),
            'datasphere_datasets': str(Path.home() / 'workspace' / 'datasets' / 'InterfDataset'),
            'problem_path': str(Path.home() / 'workspace' / 'datasphere' / 'datasets' / 'InterfDataset')
        }
    else:  # Unix/Linux/macOS
        return {
            'home': str(Path.home()),
            'workspace': '/home/jupyter/work',
            'datasets': '/home/jupyter/datasets',
            'datasphere_datasets': '/home/jupyter/work/datasets/InterfDataset',
            'problem_path': '/home/jupyter/datasphere/datasets/InterfDataset'
        }

def check_dataset_path(root_path):
    """
    Comprehensive check of dataset directory.
    """
    print(f"🔍 Checking dataset path: {root_path}")
    print(f"   Absolute path: {os.path.abspath(root_path)}")
    print(f"   Exists: {os.path.exists(root_path)}")
    print(f"   Is directory: {os.path.isdir(root_path) if os.path.exists(root_path) else False}")

    if not os.path.exists(root_path):
        print(f"❌ ERROR: Dataset path does not exist!")
        return False

    if not os.path.isdir(root_path):
        print(f"❌ ERROR: Path is not a directory!")
        return False

    # Check directory contents
    try:
        contents = os.listdir(root_path)
        print(f"   Directory contents ({len(contents)} items):")

        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = []

        for item in contents[:20]:  # Show first 20 items
            item_path = os.path.join(root_path, item)
            print(f"     - {item} {'(dir)' if os.path.isdir(item_path) else '(file)'}")

            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in image_extensions:
                    image_files.append(item)

        print(f"   Found {len(image_files)} image files (first scan)")

        # Recursive search for images
        print(f"\n🔍 Recursive search for images...")
        image_patterns = ['**/*.png', '**/*.jpg', '**/*.tif', '**/*.bmp']
        found_images = []

        for pattern in image_patterns:
            import glob
            full_pattern = os.path.join(root_path, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            found_images.extend(matches)
            print(f"   Pattern '{pattern}': {len(matches)} files")

            if matches and len(matches) <= 5:
                for match in matches:
                    print(f"     - {os.path.relpath(match, root_path)}")

        # Remove duplicates
        found_images = list(set(found_images))
        print(f"\n   Total unique images found: {len(found_images)}")

        if found_images:
            # Test loading first image
            test_image = found_images[0]
            print(f"\n🧪 Testing image loading: {os.path.basename(test_image)}")
            try:
                img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    print(f"   ✅ Image loaded successfully: {img.shape}")
                    print(f"   Data type: {img.dtype}, Range: [{img.min()}, {img.max()}]")
                else:
                    print(f"   ❌ Failed to load image")
                    return False
            except Exception as e:
                print(f"   ❌ Error loading image: {e}")
                return False

        return len(found_images) > 0

    except Exception as e:
        print(f"❌ ERROR accessing directory: {e}")
        return False

def suggest_alternative_paths(base_path):
    """
    Suggest alternative dataset paths.
    """
    print(f"\n💡 Suggested alternative paths:")

    # Get platform-appropriate paths
    platform_paths = get_platform_paths()

    # Common dataset locations
    suggestions = [
        os.path.join(base_path, "datasets"),
        os.path.join(base_path, "data"),
        os.path.join(base_path, "InterfDataset"),
        os.path.join(base_path, "interferograms"),
        platform_paths['datasets'],
        platform_paths['datasphere_datasets'],
        os.path.join(os.getcwd(), "datasets"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "InterfDataset"),
    ]

    for suggestion in suggestions:
        exists = os.path.exists(suggestion)
        is_dir = os.path.isdir(suggestion) if exists else False
        print(f"   {suggestion}: {'✅' if exists and is_dir else '❌'}")

def create_sample_data(output_dir="sample_dataset", num_images=5):
    """
    Create sample interferogram data for testing.
    """
    print(f"\n📁 Creating sample dataset: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # Create sample interferogram
        size = (512, 512)
        y, x = np.meshgrid(np.linspace(0, size[0]-1, size[0]),
                          np.linspace(0, size[1]-1, size[1]), indexing='ij')

        # Generate interference pattern
        fringe_spacing = 40
        pattern = 128 + 50 * np.sin(2 * np.pi * x / fringe_spacing) * np.cos(2 * np.pi * y / fringe_spacing)

        # Add noise
        noise = np.random.normal(0, 5, size)
        pattern = np.clip(pattern + noise, 0, 255)

        # Save with Korsch naming convention
        filename = f"mirror1_{i}_{i+1}_{i+2}_{i+3}_{i+4}_mirror2_{i+5}_{i+6}_{i+7}_{i+8}_{i+9}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, pattern.astype(np.uint8))
        print(f"   Created: {filename}")

    print(f"✅ Sample dataset created with {num_images} images")
    return output_dir

def check_config_file(config_path):
    """
    Check dataset configuration in YAML file.
    """
    print(f"\n📄 Checking config file: {config_path}")

    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return None

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✅ Config file loaded successfully")

        # Extract data configuration
        data_config = config.get('data', {})
        root_path = data_config.get('root', '')
        img_glob = data_config.get('img_glob', '')

        print(f"   Dataset root: {root_path}")
        print(f"   Image glob: {img_glob}")
        print(f"   Image size: {data_config.get('image_size', 'Not specified')}")

        # Check reference config
        ref_config = config.get('reference_interferogram', {})
        if ref_config.get('enabled', False):
            ref_path = ref_config.get('path', '')
            print(f"   Reference interferogram: {ref_path}")

            if os.path.exists(ref_path):
                print(f"   ✅ Reference file exists")
            else:
                print(f"   ❌ Reference file NOT found")

        return config

    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return None

def main():
    """
    Main diagnostic function.
    """
    print("🔍 Dataset Path Diagnostics")
    print("=" * 50)

    # Get platform-appropriate paths
    platform_paths = get_platform_paths()
    problem_path = platform_paths['problem_path']

    # Check the problematic path
    success = check_dataset_path(problem_path)

    if not success:
        print(f"\n❌ Dataset not found at: {problem_path}")

        # Suggest alternatives
        base_dir = platform_paths['home']
        suggest_alternative_paths(base_dir)

        # Check current working directory
        cwd = os.getcwd()
        print(f"\n📍 Current working directory: {cwd}")
        suggest_alternative_paths(cwd)

        # Offer to create sample data
        create_sample_data("sample_interferograms", num_images=5)

        print(f"\n🔧 SOLUTIONS:")
        print(f"1. Create/dowload dataset to one of the suggested paths")
        print(f"2. Update config file with correct path")
        print(f"3. Use sample dataset for testing: python src/train.py --config configs/debug_config.yaml")

        # Create debug config
        debug_config = {
            'seed': 42,
            'device': 'cuda',
            'data': {
                'root': os.path.abspath('sample_interferograms'),
                'img_glob': '**/*.png',
                'image_size': 256,
                'korsch_mode': True,
                'bits_per_number': 5,
                'augmentations': {'enabled': False},
                'loader': {'workers': 4, 'pin_memory': True}
            },
            'model': {
                'out_dim': 50,  # 2 mirrors × 5 params × 5 bits
                'bits_per_parameter': 5,
                'base_channels': 32,
                'width_multipliers': [2, 4, 8],
                'block_repeats': [1, 1, 2]
            },
            'train': {
                'epochs': 5,
                'batch_size': 8,
                'lr': 0.001
            },
            'split': {
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'test_ratio': 0.0
            },
            'logging': {
                'out_dir': 'runs/debug_training',
                'project_name': 'debug_dataset'
            }
        }

        import yaml
        with open('debug_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(debug_config, f, default_flow_style=False, allow_unicode=True)

        print(f"4. Debug config created: debug_config.yaml")

    # Check for config files
    config_files = [
        "configs/korsch_reference.yaml",
        "configs/default.yaml",
        "debug_config.yaml"
    ]

    print(f"\n📋 Checking configuration files:")
    for config_file in config_files:
        check_config_file(config_file)

if __name__ == "__main__":
    main()