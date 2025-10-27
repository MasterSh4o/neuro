"""
Dataset Diagnostic Utilities

Утилиты для диагностики проблем с загрузкой датасетов.
"""

import os
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path


class DatasetDiagnostic:
    """Класс для диагностики проблем с датасетами."""

    @staticmethod
    def diagnose_dataset_path(
        root_path: str,
        patterns: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Диагностика пути к датасету и шаблонов поиска.

        Args:
            root_path: Путь к директории датасета
            patterns: Список шаблонов поиска (опционально)
            verbose: Вывод подробной информации

        Returns:
            Dict[str, Any]: Результат диагностики
        """
        result = {
            'root_path': root_path,
            'exists': False,
            'is_directory': False,
            'patterns': patterns or ['**/*.png', '**/*.jpg', '**/*.tif', '**/*.bmp'],
            'files_found': [],
            'directories_found': [],
            'image_files_found': [],
            'issues': [],
            'suggestions': []
        }

        if verbose:
            print(f"🔍 Диагностика датасета: {root_path}")

        # Проверка существования пути
        if not os.path.exists(root_path):
            result['issues'].append(f"Путь не существует: {root_path}")
            result['suggestions'].append("Проверьте правильность пути к датасету")
            if verbose:
                print(f"❌ Путь не существует: {root_path}")
            return result

        result['exists'] = True

        # Проверка, что это директория
        if not os.path.isdir(root_path):
            result['issues'].append(f"Путь не является директорией: {root_path}")
            result['suggestions'].append("Укажите путь к директории, а не к файлу")
            if verbose:
                print(f"❌ Путь не является директорией: {root_path}")
            return result

        result['is_directory'] = True

        if verbose:
            print(f"✅ Директория существует: {root_path}")

        # Анализ содержимого директории
        try:
            contents = os.listdir(root_path)
            result['total_items'] = len(contents)

            if verbose:
                print(f"📁 Всего элементов в директории: {len(contents)}")

            # Классификация содержимого
            for item in contents:
                item_path = os.path.join(root_path, item)
                if os.path.isdir(item_path):
                    result['directories_found'].append(item)
                else:
                    result['files_found'].append(item)

            # Поиск файлов изображений
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
            for item in result['files_found']:
                if any(item.lower().endswith(ext) for ext in image_extensions):
                    result['image_files_found'].append(item)

            if verbose:
                print(f"📄 Найдено файлов: {len(result['files_found'])}")
                print(f"📁 Найдено поддиректорий: {len(result['directories_found'])}")
                print(f"🖼️  Найдено изображений: {len(result['image_files_found'])}")

                # Показать первые несколько элементов каждого типа
                if result['image_files_found']:
                    print(f"   Примеры изображений: {result['image_files_found'][:3]}")
                if result['directories_found']:
                    print(f"   Поддиректории: {result['directories_found'][:3]}")

        except PermissionError:
            result['issues'].append(f"Нет доступа к директории: {root_path}")
            result['suggestions'].append("Проверьте права доступа к директории")
            if verbose:
                print(f"❌ Нет доступа к директории: {root_path}")
            return result

        # Тестирование шаблонов поиска
        if patterns:
            if verbose:
                print(f"\n🔍 Тестирование шаблонов поиска:")

            for pattern in patterns:
                search_path = os.path.join(root_path, pattern)
                matches = glob.glob(search_path, recursive=True)

                pattern_result = {
                    'pattern': pattern,
                    'search_path': search_path,
                    'matches_count': len(matches),
                    'matches': matches[:5]  # Первые 5 совпадений
                }

                result.setdefault('pattern_results', []).append(pattern_result)

                if verbose:
                    print(f"   📋 Шаблон: {pattern}")
                    print(f"      Поиск: {search_path}")
                    print(f"      Найдено: {len(matches)} файлов")
                    if matches:
                        print(f"      Примеры: {[os.path.basename(m) for m in matches[:3]]}")

        # Формирование предложений
        if not result['image_files_found']:
            if result['directories_found']:
                result['suggestions'].append(
                    "Изображения могут находиться в поддиректориях. Используйте рекурсивные шаблоны: '**/*.png'"
                )
            else:
                result['suggestions'].append(
                    "В директории нет файлов изображений. Проверьте содержимое и добавьте изображения."
                )

        if result['image_files_found'] and not any(
            pr['matches_count'] > 0 for pr in result.get('pattern_results', [])
        ):
            # Есть изображения, но шаблоны не находят
            sample_image = result['image_files_found'][0]
            name_parts = sample_image.split('.')
            if len(name_parts) >= 2:
                suggested_pattern = f"{name_parts[0]}*.{name_parts[1]}"
                result['suggestions'].append(
                    f"Попробуйте шаблон: '{suggested_pattern}' или более общий: '*.{name_parts[1]}'"
                )

        return result

    @staticmethod
    def create_sample_dataset(
        output_dir: str,
        num_images: int = 5,
        image_size: tuple = (256, 256),
        pattern_prefix: str = "mirror"
    ) -> str:
        """
        Создать пример датасета для тестирования.

        Args:
            output_dir: Директория для создания датасета
            num_images: Количество изображений для создания
            image_size: Размер изображений (height, width)
            pattern_prefix: Префикс для имен файлов

        Returns:
            str: Путь к созданной директории
        """
        import cv2
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        print(f"📁 Создание тестового датасета: {output_dir}")

        for i in range(num_images):
            # Создание тестовой интерферограммы
            height, width = image_size
            y, x = np.meshgrid(np.linspace(0, height-1, height),
                              np.linspace(0, width-1, width), indexing='ij')

            # Генерация интерференционного паттерна
            fringe_spacing = 40
            pattern = 128 + 50 * np.sin(2 * np.pi * x / fringe_spacing) * np.cos(2 * np.pi * y / fringe_spacing)

            # Добавление шума
            noise = np.random.normal(0, 5, (height, width))
            pattern = np.clip(pattern + noise, 0, 255)

            # Создание имени файла в формате Korsch
            filename = f"{pattern_prefix}1_{i}_{i+1}_{i+2}_{i+3}_{i+4}_{pattern_prefix}2_{i+5}_{i+6}_{i+7}_{i+8}_{i+9}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, pattern.astype(np.uint8))

            print(f"   ✅ Создан файл: {filename}")

        print(f"✅ Тестовый датасет создан: {output_dir}")
        print(f"   Файлов создано: {num_images}")
        print(f"   Размер изображений: {image_size}")

        return output_dir

    @staticmethod
    def suggest_config(
        root_path: str,
        found_files: List[str],
        current_patterns: List[str]
    ) -> Dict[str, Any]:
        """
        Предложить конфигурацию на основе найденных файлов.

        Args:
            root_path: Путь к датасету
            found_files: Список найденных файлов
            current_patterns: Текущие шаблоны поиска

        Returns:
            Dict[str, Any]: Предложения по конфигурации
        """
        suggestions = {
            'img_glob': current_patterns,
            'image_size': 512,
            'notes': []
        }

        if found_files:
            # Анализ имен файлов для определения формата
            sample_file = found_files[0]
            if '_' in sample_file:
                suggestions['notes'].append(
                    "Файлы содержат '_' в имени, возможно используется формат Korsch"
                )
                suggestions['korsch_mode'] = True

            # Анализ расширений
            extensions = set()
            for file in found_files:
                if '.' in file:
                    extensions.add(file.split('.')[-1].lower())

            if len(extensions) == 1:
                ext = list(extensions)[0]
                suggestions['img_glob'] = [f"**/*.{ext}"]
                suggestions['notes'].append(f"Все файлы имеют расширение .{ext}")

        # Если файлы не найдены, предложить создать тестовый датасет
        if not found_files:
            suggestions['notes'].append(
                "Файлы не найдены. Создайте тестовый датасет с помощью DatasetDiagnostic.create_sample_dataset()"
            )
            # Предложить путь для тестового датасета
            parent_dir = os.path.dirname(root_path)
            test_dataset_path = os.path.join(parent_dir, "test_dataset")
            suggestions['test_dataset_path'] = test_dataset_path

        return suggestions


def diagnose_dataset_issue(root_path: str, patterns: Optional[List[str]] = None) -> None:
    """
    Быстрая диагностика проблем с датасетом.

    Args:
        root_path: Путь к датасету
        patterns: Шаблоны поиска
    """
    if patterns is None:
        patterns = ['**/*.png', '**/*.jpg', '**/*.tif', '**/*.bmp']

    print("🚀 Диагностика датасета NeuroInterf")
    print("=" * 50)

    # Диагностика
    diagnostic = DatasetDiagnostic()
    result = diagnostic.diagnose_dataset_path(root_path, patterns, verbose=True)

    print("\n" + "=" * 50)
    print("📊 Результаты диагностики:")

    if result['issues']:
        print("❌ Обнаружены проблемы:")
        for issue in result['issues']:
            print(f"   • {issue}")

    if result['suggestions']:
        print("\n💡 Рекомендации:")
        for suggestion in result['suggestions']:
            print(f"   • {suggestion}")

    # Анализ шаблонов
    if 'pattern_results' in result:
        print("\n📋 Анализ шаблонов поиска:")
        for pr in result['pattern_results']:
            status = "✅" if pr['matches_count'] > 0 else "❌"
            print(f"   {status} {pr['pattern']}: {pr['matches_count']} файлов")

    print("\n" + "=" * 50)

    # Если есть проблемы, предложить решение
    if result['issues'] or not any(pr['matches_count'] > 0 for pr in result.get('pattern_results', [])):
        print("🔧 Предлагаемое решение:")

        if not result['exists']:
            print("1. Проверьте путь к датасету")
            print("2. Создайте тестовый датасет:")
            print(f"   DatasetDiagnostic.create_sample_dataset('{root_path}')")

        elif result['exists'] and not result['image_files_found']:
            print("1. В директории нет изображений")
            print("2. Создайте тестовый датасет:")
            parent_dir = os.path.dirname(root_path)
            test_path = os.path.join(parent_dir, "test_dataset")
            print(f"   DatasetDiagnostic.create_sample_dataset('{test_path}')")

        else:
            print("1. Измените шаблоны поиска в конфигурации")
            if result['image_files_found']:
                sample = result['image_files_found'][0]
                ext = sample.split('.')[-1]
                print(f"   img_glob: ['**/*.{ext}']")
            else:
                print("   img_glob: ['*.png', '*.jpg']")