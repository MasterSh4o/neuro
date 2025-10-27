#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления RuntimeError в датасете.
Проверяет правильность поиска файлов и обработки ошибок.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Добавляем src в путь
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def create_test_dataset_with_files():
    """Создать тестовый датасет с файлами для проверки."""
    import cv2
    import numpy as np

    # Создаем временную директорию
    test_dir = tempfile.mkdtemp(prefix="test_neuro_dataset_")
    print(f"📁 Создан тестовый датасет: {test_dir}")

    # Создаем несколько тестовых изображений
    test_images = [
        "mirror1_0_1_2_3_4_mirror2_5_6_7_8_9.png",
        "mirror1_10_11_12_13_14_mirror2_15_16_17_18_19.png",
        "test_image_1.jpg",
        "sample.tif"
    ]

    for filename in test_images:
        # Создаем простое тестовое изображение
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        filepath = os.path.join(test_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"   ✅ Создан файл: {filename}")

    return test_dir

def test_dataset_discovery():
    """Тестирование обнаружения файлов датасетом."""
    print("\n🧪 Тестирование обнаружения файлов...")

    try:
        from data.dataset_reference_simple import ReferenceInterferogramDataset

        # Создаем тестовый датасет
        test_dir = create_test_dataset_with_files()

        try:
            # Тест с базовыми шаблонами
            print("  ✅ Тест с базовыми шаблонами")
            dataset1 = ReferenceInterferogramDataset(
                root=test_dir,
                img_glob="**/*.png,**/*.jpg,**/*.tif",
                image_size=128,
                korsch_mode=True,
                bits_per_number=5
            )
            print(f"    ✅ Найдено файлов: {len(dataset1.files)}")
            assert len(dataset1.files) >= 3, f"Ожидалось >=3 файлов, получено {len(dataset1.files)}"

            # Тест с конкретным шаблоном
            print("  ✅ Тест с конкретным шаблоном PNG")
            dataset2 = ReferenceInterferogramDataset(
                root=test_dir,
                img_glob="**/*.png",
                image_size=128,
                korsch_mode=True,
                bits_per_number=5
            )
            print(f"    ✅ Найдено PNG файлов: {len(dataset2.files)}")
            assert len(dataset2.files) >= 2, f"Ожидалось >=2 PNG файлов, получено {len(dataset2.files)}"

            # Тест с неверным шаблоном (должен вызвать ошибку)
            print("  ✅ Тест с неверным шаблоном")
            try:
                dataset3 = ReferenceInterferogramDataset(
                    root=test_dir,
                    img_glob="**/*.nonexistent",
                    image_size=128,
                    korsch_mode=True,
                    bits_per_number=5
                )
                assert False, "Должна была быть ошибка RuntimeError"
            except RuntimeError as e:
                if "No images found" in str(e):
                    print(f"    ✅ Правильная обработка ошибки: {e}")
                else:
                    raise e

            print("  ✅ Все тесты обнаружения файлов пройдены!")
            return True

        finally:
            # Очищаем тестовую директорию
            shutil.rmtree(test_dir)
            print(f"   🧹 Тестовая директория удалена: {test_dir}")

    except ImportError as e:
        print(f"  ❌ Не удалось импортировать ReferenceInterferogramDataset: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagnostic_utilities():
    """Тестирование утилит диагностики."""
    print("\n🧪 Тестирование утилит диагностики...")

    try:
        from utils.dataset_diagnostic import DatasetDiagnostic, diagnose_dataset_issue

        # Создаем тестовый датасет
        test_dir = create_test_dataset_with_files()

        try:
            # Тест диагностики
            print("  ✅ Тест базовой диагностики")
            result = DatasetDiagnostic.diagnose_dataset_path(
                test_dir,
                patterns=["**/*.png", "**/*.jpg"],
                verbose=False
            )

            assert result['exists'], "Директория должна существовать"
            assert result['is_directory'], "Путь должен быть директорией"
            assert len(result['image_files_found']) >= 3, "Должны быть найдены изображения"

            print(f"    ✅ Найдено изображений: {len(result['image_files_found'])}")

            # Тест функции diagnose_dataset_issue
            print("  ✅ Тест функции diagnose_dataset_issue")
            diagnose_dataset_issue(test_dir, ["**/*.png"])

            print("  ✅ Все тесты диагностики пройдены!")
            return True

        finally:
            # Очищаем тестовую директорию
            shutil.rmtree(test_dir)

    except ImportError as e:
        print(f"  ❌ Не удалось импортировать утилиты диагностики: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Тестирование обработки ошибок."""
    print("\n🧪 Тестирование обработки ошибок...")

    try:
        from data.dataset_reference_simple import ReferenceInterferogramDataset

        # Тест с несуществующей директорией
        print("  ✅ Тест с несуществующей директорией")
        try:
            dataset = ReferenceInterferogramDataset(
                root="/nonexistent/dataset/path",
                img_glob="**/*.png",
                image_size=128,
                korsch_mode=True,
                bits_per_number=5
            )
            assert False, "Должна была быть ошибка FileNotFoundError"
        except FileNotFoundError as e:
            print(f"    ✅ Правильная ошибка: {e}")

        # Тест с файлом вместо директории
        print("  ✅ Тест с файлом вместо директории")
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            dataset = ReferenceInterferogramDataset(
                root=tmp_path,
                img_glob="**/*.png",
                image_size=128,
                korsch_mode=True,
                bits_per_number=5
            )
            assert False, "Должна была быть ошибка NotADirectoryError"
        except NotADirectoryError as e:
            print(f"    ✅ Правильная ошибка: {e}")
        finally:
            os.unlink(tmp_path)

        print("  ✅ Все тесты обработки ошибок пройдены!")
        return True

    except ImportError as e:
        print(f"  ❌ Не удалось импортировать ReferenceInterferogramDataset: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция для запуска тестов."""
    print("🚀 Запуск тестов исправления RuntimeError в датасете...")
    print("=" * 60)

    results = []

    # Запускаем тесты
    results.append(test_dataset_discovery())
    results.append(test_diagnostic_utilities())
    results.append(test_error_handling())

    print("\n" + "=" * 60)
    print("📊 Результаты тестов:")

    if all(results):
        print("🎉 Все тесты пройдены успешно!")
        print("✅ Ошибка RuntimeError исправлена")
        print("✅ Поиск файлов работает корректно")
        print("✅ Обработка ошибок улучшена")
        print("✅ Диагностика работает")
        return 0
    else:
        print("❌ Некоторые тесты не пройдены. Проверьте ошибки выше.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)