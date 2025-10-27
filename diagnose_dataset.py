#!/usr/bin/env python3
"""
Скрипт для быстрой диагностики проблем с датасетом.
Использование: python diagnose_dataset.py <path_to_dataset>
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Главная функция."""
    if len(sys.argv) < 2:
        print("Использование: python diagnose_dataset.py <path_to_dataset>")
        print("Пример: python diagnose_dataset.py /home/jupyter/work/datasets/InterfDataset")
        print("\nИли используйте встроенные тесты:")
        print("python diagnose_dataset.py --test")
        return 1

    if sys.argv[1] == "--test":
        # Запуск тестов
        print("🧪 Запуск тестов диагностики...")

        from utils.dataset_diagnostic import DatasetDiagnostic

        # Создать тестовый датасет
        test_dir = "test_dataset_diagnostic"
        DatasetDiagnostic.create_sample_dataset(test_dir, num_images=3)

        # Протестировать диагностику
        result = DatasetDiagnostic.diagnose_dataset_path(test_dir, verbose=True)

        # Очистить
        import shutil
        shutil.rmtree(test_dir)

        print("✅ Тесты пройдены успешно!")
        return 0

    dataset_path = sys.argv[1]

    try:
        from utils.dataset_diagnostic import diagnose_dataset_issue
        diagnose_dataset_issue(dataset_path)
        return 0

    except ImportError as e:
        print(f"❌ Не удалось импортировать модуль диагностики: {e}")
        print("Убедитесь, что файл src/utils/dataset_diagnostic.py существует")
        return 1
    except Exception as e:
        print(f"❌ Ошибка при диагностике: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)