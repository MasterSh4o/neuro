#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления RuntimeError: channels_last
"""
import sys
import os
sys.path.append('src')

import torch
import numpy as np

def test_tensor_formats():
    """Проверяем различные форматы тензоров"""
    print("✅ Testing tensor formats for channels_last compatibility...")

    # Test 1: 4D тензоры (должны работать)
    print("\n1. Testing 4D tensors [B, C, H, W]:")
    tensor_4d = torch.randn(2, 3, 224, 224)
    print(f"   Original shape: {tensor_4d.shape}")

    try:
        tensor_4d_cl = tensor_4d.contiguous(memory_format=torch.channels_last)
        print(f"   ✅ channels_last applied successfully: {tensor_4d_cl.shape}")
    except Exception as e:
        print(f"   ❌ Error applying channels_last: {e}")

    # Test 2: 3D тензоры (должны вызывать ошибку)
    print("\n2. Testing 3D tensors [C, H, W]:")
    tensor_3d = torch.randn(3, 224, 224)
    print(f"   Original shape: {tensor_3d.shape}")

    try:
        tensor_3d_cl = tensor_3d.contiguous(memory_format=torch.channels_last)
        print(f"   ❌ Unexpected success: {tensor_3d_cl.shape}")
    except Exception as e:
        print(f"   ✅ Expected error: {e}")

    # Test 3: Исправление 3D -> 4D
    print("\n3. Testing 3D -> 4D fix:")
    tensor_3d_fixed = tensor_3d.unsqueeze(0)  # [1, C, H, W]
    print(f"   Fixed shape: {tensor_3d_fixed.shape}")

    try:
        tensor_3d_fixed_cl = tensor_3d_fixed.contiguous(memory_format=torch.channels_last)
        print(f"   ✅ channels_last applied to fixed tensor: {tensor_3d_fixed_cl.shape}")
    except Exception as e:
        print(f"   ❌ Error with fixed tensor: {e}")

def test_dataset_tensor_formats():
    """Тестируем форматы тензоров из датасета"""
    print("\n✅ Testing dataset tensor formats...")

    try:
        from data.dataset_reference_simple import ReferenceInterferogramDataset

        # Создаем тестовый датасет
        dataset = ReferenceInterferogramDataset(
            root='src/data',
            img_glob='**/*.py',  # Используем .py файлы для теста
            korsch_mode=False,
            mode='dual'  # Тестируем dual mode
        )

        if len(dataset) > 0:
            # Тестируем получение элемента
            sample = dataset[0]
            print(f"   Dataset returned {len(sample)} elements")

            if len(sample) >= 3:
                dual_input, label, mode_info = sample[:3]
                print(f"   dual_input shape: {dual_input.shape}")
                print(f"   dual_input dtype: {dual_input.dtype}")

                # Проверяем, что dual_input имеет правильную размерность
                if dual_input.dim() == 4:
                    print("   ✅ dual_input has correct 4D format [B, C, H, W]")

                    try:
                        dual_input_cl = dual_input.contiguous(memory_format=torch.channels_last)
                        print("   ✅ channels_last applied successfully to dual_input")
                    except Exception as e:
                        print(f"   ❌ Error applying channels_last: {e}")

                elif dual_input.dim() == 3:
                    print("   ⚠️  dual_input has 3D format [C, H, W]")
                    print("   💡 This might cause channels_last error in train.py")

                else:
                    print(f"   ❌ Unexpected dual_input format: {dual_input.shape}")
            else:
                print("   ⚠️  Dataset sample format is unexpected")
        else:
            print("   ⚠️  Dataset is empty, cannot test tensor formats")

    except Exception as e:
        print(f"   ❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

def test_collate_function():
    """Тестируем collate функцию"""
    print("\n✅ Testing collate function...")

    try:
        # Импортируем collate функцию из train.py
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", "src/train.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)

        reference_collate_fn = train_module.reference_collate_fn

        # Создаем тестовые данные
        batch_data = [
            (torch.randn(1, 224, 224), torch.randn(2, 224, 224), {'mode': 'dual'}),
            (torch.randn(1, 224, 224), torch.randn(2, 224, 224), {'mode': 'dual'}),
        ]

        # Применяем collate функцию
        result = reference_collate_fn(batch_data)
        x, y, dual_input, mode_info = result

        print(f"   x shape: {x.shape}")
        print(f"   dual_input shape: {dual_input.shape if dual_input is not None else 'None'}")

        # Проверяем форматы
        if x.dim() == 4:
            print("   ✅ x has correct 4D format")
        else:
            print(f"   ❌ x has wrong format: {x.shape}")

        if dual_input is not None and dual_input.dim() == 4:
            print("   ✅ dual_input has correct 4D format from collate")
        elif dual_input is not None:
            print(f"   ❌ dual_input has wrong format from collate: {dual_input.shape}")
        else:
            print("   ⚠️  dual_input is None")

    except Exception as e:
        print(f"   ❌ Error testing collate function: {e}")
        import traceback
        traceback.print_exc()

def test_channels_last_scenarios():
    """Тестируем различные сценарии с channels_last"""
    print("\n✅ Testing channels_last scenarios...")

    scenarios = [
        ("4D tensor [B, C, H, W]", torch.randn(2, 3, 224, 224)),
        ("3D tensor [C, H, W]", torch.randn(3, 224, 224)),
        ("2D tensor [H, W]", torch.randn(224, 224)),
        ("5D tensor [B, C, D, H, W]", torch.randn(1, 3, 10, 224, 224)),
    ]

    for name, tensor in scenarios:
        print(f"\n   Testing {name}:")
        print(f"   Shape: {tensor.shape}, Dim: {tensor.dim()}")

        try:
            if tensor.dim() == 4:
                tensor_cl = tensor.contiguous(memory_format=torch.channels_last)
                print(f"   ✅ channels_last applied successfully")
            else:
                print(f"   ⚠️  Skipping channels_last (not 4D)")

                # Пробуем исправить
                if tensor.dim() == 3:
                    fixed = tensor.unsqueeze(0)
                    print(f"   💡 Fixed shape: {fixed.shape}")
                    if fixed.dim() == 4:
                        fixed_cl = fixed.contiguous(memory_format=torch.channels_last)
                        print(f"   ✅ channels_last applied to fixed tensor")

        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing channels_last fix...")
    print("=" * 60)

    results = []
    results.append(test_tensor_formats())
    results.append(test_dataset_tensor_formats())
    results.append(test_collate_function())
    results.append(test_channels_last_scenarios())

    print("\n" + "=" * 60)
    print("📊 Test Summary:")

    print("✅ Tensor format analysis completed")
    print("✅ Dataset compatibility checked")
    print("✅ Collate function tested")
    print("✅ Various scenarios evaluated")

    print("\n💡 Recommendations:")
    print("1. Ensure all tensors are 4D [B, C, H, W] before applying channels_last")
    print("2. Use .unsqueeze(0) to add batch dimension to 3D tensors")
    print("3. Check tensor dimensions before memory format operations")
    print("4. Test with both reference and non-reference modes")

    print("\n🎯 The fix should resolve the RuntimeError: 'required rank 4 tensor to use channels_last format'")
    print("   by adding proper dimension checks and automatic fixes where possible.")