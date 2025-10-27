#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления TypeError.
Проверяет поддержку различных форматов параметра image_size.
"""

import sys
import traceback
from pathlib import Path

# Добавляем src в путь
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_reference_processor():
    """Тестирование ReferenceProcessor с разными типами target_size."""
    print("🧪 Тестирование ReferenceProcessor...")

    try:
        from utils.reference_processor import ReferenceProcessor

        # Тест с int
        print("  ✅ Тест с target_size=512 (int)")
        try:
            # Создадим временное изображение для теста
            import cv2
            import numpy as np

            test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            test_path = "temp_test_image.png"
            cv2.imwrite(test_path, test_img)

            processor_int = ReferenceProcessor(test_path, target_size=512)
            assert processor_int.target_size == (512, 512), f"Ожидалось (512, 512), получено {processor_int.target_size}"
            print("    ✅ int формат работает корректно")

            # Тест с tuple
            print("  ✅ Тест с target_size=(256, 256) (tuple)")
            processor_tuple = ReferenceProcessor(test_path, target_size=(256, 256))
            assert processor_tuple.target_size == (256, 256), f"Ожидалось (256, 256), получено {processor_tuple.target_size}"
            print("    ✅ tuple формат работает корректно")

            # Тест с list
            print("  ✅ Тест с target_size=[128, 128] (list)")
            processor_list = ReferenceProcessor(test_path, target_size=[128, 128])
            assert processor_list.target_size == (128, 128), f"Ожидалось (128, 128), получено {processor_list.target_size}"
            print("    ✅ list формат работает корректно")

            # Тест с валидацией ошибок
            print("  ✅ Тест валидации ошибок")
            try:
                ReferenceProcessor(test_path, target_size=-1)
                assert False, "Должна была быть ошибка ValueError"
            except ValueError as e:
                print(f"    ✅ Отрицательный размер корректно отклонен: {e}")

            try:
                ReferenceProcessor(test_path, target_size=(512, 512, 256))
                assert False, "Должна была быть ошибка ValueError"
            except ValueError as e:
                print(f"    ✅ Некорректный размер tuple корректно отклонен: {e}")

            # Удаляем временное изображение
            import os
            os.remove(test_path)

        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"  ❌ Не удалось импортировать ReferenceProcessor: {e}")
        return False

    print("  ✅ ReferenceProcessor тесты пройдены успешно!")
    return True

def test_dataset_reference_simple():
    """Тестирование ReferenceInterferogramDataset с разными типами image_size."""
    print("\n🧪 Тестирование ReferenceInterferogramDataset...")

    try:
        from data.dataset_reference_simple import ReferenceInterferogramDataset

        # Создадим временную директорию с тестовыми изображениями
        import os
        import cv2
        import numpy as np

        test_dir = "temp_test_dataset"
        os.makedirs(test_dir, exist_ok=True)

        # Создадим тестовые изображения
        for i in range(3):
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            filename = f"mirror1_{i}_{i+1}_{i+2}_{i+3}_{i+4}_mirror2_{i+5}_{i+6}_{i+7}_{i+8}_{i+9}.png"
            cv2.imwrite(os.path.join(test_dir, filename), img)

        # Тест с int
        print("  ✅ Тест с image_size=128 (int)")
        dataset_int = ReferenceInterferogramDataset(
            root=test_dir,
            image_size=128,
            korsch_mode=True,
            bits_per_number=5
        )
        assert dataset_int.image_size == (128, 128), f"Ожидалось (128, 128), получено {dataset_int.image_size}"
        print("    ✅ int формат работает корректно")

        # Тест с tuple
        print("  ✅ Тест с image_size=(64, 64) (tuple)")
        dataset_tuple = ReferenceInterferogramDataset(
            root=test_dir,
            image_size=(64, 64),
            korsch_mode=True,
            bits_per_number=5
        )
        assert dataset_tuple.image_size == (64, 64), f"Ожидалось (64, 64), получено {dataset_tuple.image_size}"
        print("    ✅ tuple формат работает корректно")

        # Тест с list
        print("  ✅ Тест с image_size=[32, 32] (list)")
        dataset_list = ReferenceInterferogramDataset(
            root=test_dir,
            image_size=[32, 32],
            korsch_mode=True,
            bits_per_number=5
        )
        assert dataset_list.image_size == (32, 32), f"Ожидалось (32, 32), получено {dataset_list.image_size}"
        print("    ✅ list формат работает корректно")

        # Тест валидации ошибок
        print("  ✅ Тест валидации ошибок")
        try:
            ReferenceInterferogramDataset(test_dir, image_size=-1)
            assert False, "Должна была быть ошибка ValueError"
        except ValueError as e:
            print(f"    ✅ Отрицательный размер корректно отклонен: {e}")

        # Удаляем тестовые данные
        import shutil
        shutil.rmtree(test_dir)

    except ImportError as e:
        print(f"  ❌ Не удалось импортировать ReferenceInterferogramDataset: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        traceback.print_exc()
        return False

    print("  ✅ ReferenceInterferogramDataset тесты пройдены успешно!")
    return True

def main():
    """Главная функция для запуска тестов."""
    print("🚀 Запуск тестов исправления TypeError...")
    print("=" * 50)

    results = []

    # Запускаем тесты
    results.append(test_reference_processor())
    results.append(test_dataset_reference_simple())

    print("\n" + "=" * 50)
    print("📊 Результаты тестов:")

    if all(results):
        print("🎉 Все тесты пройдены успешно! Ошибка TypeError исправлена.")
        return 0
    else:
        print("❌ Некоторые тесты не пройдены. Проверьте ошибки выше.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)