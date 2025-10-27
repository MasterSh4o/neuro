#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления ошибки AttributeError: 'ReferenceInterferogramDataset' object has no attribute 'K'
"""
import sys
import os
sys.path.append('src')

def test_dataset_K_attribute():
    """Проверяем, что у датасета есть атрибут K"""
    try:
        from data.dataset_reference_simple import ReferenceInterferogramDataset

        # Создаем временный тестовый датасет
        dataset = ReferenceInterferogramDataset(
            root='src/data',
            img_glob='**/*.py',  # Ищем Python файлы вместо изображений
            korsch_mode=False,
            bits_per_number=7
        )

        print(f"✅ Dataset created successfully")
        print(f"✅ Files found: {len(dataset.files)}")
        print(f"✅ Labels shape: {dataset.labels.shape}")
        print(f"✅ K attribute: {dataset.K}")
        print(f"✅ K type: {type(dataset.K)}")

        # Проверяем, что K - это целое число
        assert isinstance(dataset.K, int), f"K should be int, got {type(dataset.K)}"
        assert dataset.K > 0, f"K should be positive, got {dataset.K}"

        # Проверяем, что K соответствует размерности меток
        assert dataset.K == dataset.labels.shape[1], f"K ({dataset.K}) should equal labels.shape[1] ({dataset.labels.shape[1]})"

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_K_attribute()
    sys.exit(0 if success else 1)