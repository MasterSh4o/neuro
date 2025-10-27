#!/usr/bin/env python3
"""
Utility to fix filenames with spaces and special characters.
Переименовывает файлы с интерферограммами в стандартный формат.
"""

import os
import re
import shutil
import argparse
from pathlib import Path


def extract_korsch_parameters(filename):
    """
    Извлечение параметров Корша из имени файла.

    Returns:
        tuple: (mirror1_params, mirror2_params) or None if not found
    """
    # Удаляем расширение
    name = Path(filename).stem

    # Различные паттерны для поиска параметров
    patterns = [
        # Стандартный формат: mirror1_10_20_30_40_50_mirror2_-10_-20_-30_-40_-50
        r"mirror1[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*mirror2[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)",

        # Формат с индексами: mirror1_ax_10_ay_20_...
        r"mirror1[^a-z]*ax[^-\d]*([-\d]+)[^a-z]*ay[^-\d]*([-\d]+)[^a-z]*lx[^-\d]*([-\d]+)[^a-z]*ly[^-\d]*([-\d]+)[^a-z]*lz[^-\d]*([-\d]+)[^-\d]*mirror2[^a-z]*ax[^-\d]*([-\d]+)[^a-z]*ay[^-\d]*([-\d]+)[^a-z]*lx[^-\d]*([-\d]+)[^a-z]*ly[^-\d]*([-\d]+)[^a-z]*lz[^-\d]*([-\d]+)",

        # Универсальный поиск 10 чисел
        r"([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)[^-\d]*([-\d]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            numbers = list(map(int, match.groups()))
            if len(numbers) >= 10:
                return numbers[:10]  # Берем первые 10 чисел

    return None


def generate_standard_filename(params, extension=".png"):
    """
    Генерация стандартного имени файла из параметров.

    Args:
        params: Список из 10 чисел
        extension: Расширение файла

    Returns:
        str: Стандартное имя файла
    """
    if len(params) != 10:
        raise ValueError(f"Expected 10 parameters, got {len(params)}")

    mirror1 = params[:5]
    mirror2 = params[5:]

    filename = f"mirror1_{'_'.join(map(str, mirror1))}_mirror2_{'_'.join(map(str, mirror2))}{extension}"
    return filename


def clean_filename(filename):
    """
    Очистка имени файла от специальных символов.

    Args:
        filename: Имя файла

    Returns:
        str: Очищенное имя файла
    """
    # Заменяем пробелы на подчеркивания
    cleaned = re.sub(r'\s+', '_', filename.strip())

    # Удаляем специальные символы кроме подчеркиваний, цифр, букв
    cleaned = re.sub(r'[^\w\-_.]', '', cleaned)

    # Заменяем множественные подчеркивания на одно
    cleaned = re.sub(r'_+', '_', cleaned)

    # Удаляем подчеркивания в начале и конце
    cleaned = cleaned.strip('_')

    return cleaned


def process_directory(directory, dry_run=False, backup=True):
    """
    Обработка директории с файлами интерферограмм.

    Args:
        directory: Путь к директории
        dry_run: Только показать переименования без выполнения
        backup: Создавать резервную копию
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return

    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return

    print(f"🔍 Processing directory: {directory}")

    # Создаем резервную копию если нужно
    if backup and not dry_run:
        backup_dir = directory.parent / f"{directory.name}_backup"
        if backup_dir.exists():
            print(f"⚠️  Backup directory already exists: {backup_dir}")
        else:
            try:
                shutil.copytree(directory, backup_dir)
                print(f"✅ Created backup: {backup_dir}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                return

    # Получаем все файлы изображений
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [f for f in directory.iterdir()
                  if f.suffix.lower() in image_extensions and f.is_file()]

    print(f"📁 Found {len(image_files)} image files")

    # Счетчики
    renamed_count = 0
    error_count = 0
    skipped_count = 0

    print("\n🔄 Processing files:")
    print("-" * 80)

    for file_path in image_files:
        try:
            old_name = file_path.name
            new_name = None

            # Пробуем извлечь параметры Корша
            params = extract_korsch_parameters(old_name)

            if params:
                # Генерируем стандартное имя
                new_name = generate_standard_filename(params, file_path.suffix)
            else:
                # Если не удалось извлечь параметры, просто очищаем имя
                new_name = clean_filename(old_name)
                print(f"⚠️  Could not extract Korsch parameters from: {old_name}")
                print(f"    Will clean filename to: {new_name}")

            # Проверяем, нужно ли переименовывать
            if new_name and new_name != old_name:
                new_path = directory / new_name

                # Проверяем, не существует ли уже файл с таким именем
                if new_path.exists():
                    counter = 1
                    base_name = Path(new_name).stem
                    suffix = Path(new_name).suffix

                    while new_path.exists():
                        new_name = f"{base_name}_{counter}{suffix}"
                        new_path = directory / new_name
                        counter += 1

                    print(f"⚠️  File exists, renamed to: {new_name}")

                # Показываем информацию о переименовании
                print(f"📝 {old_name} → {new_name}")

                if not dry_run:
                    # Переименовываем файл
                    file_path.rename(new_path)

                renamed_count += 1
            else:
                skipped_count += 1
                print(f"✅ {old_name} - already correct")

        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {file_path.name}: {e}")

    print("-" * 80)
    print(f"\n📊 Summary:")
    print(f"   Files renamed: {renamed_count}")
    print(f"   Files skipped: {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Total files: {len(image_files)}")

    if dry_run:
        print(f"\n💡 This was a dry run. Use --execute to actually rename files.")

    return renamed_count, skipped_count, error_count


def create_sample_data(directory, num_files=5):
    """
    Создание примеров файлов с проблемными именами.

    Args:
        directory: Директория для создания
        num_files: Количество файлов
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True)

    import cv2
    import numpy as np

    for i in range(num_files):
        # Создаем простое изображение
        size = (256, 256)
        img = np.random.randint(0, 255, size, dtype=np.uint8)

        # Генерируем параметры
        mirror1 = [i*1, i*2, i*3, i*4, i*5]
        mirror2 = [i*-1, i*-2, i*-3, i*-4, i*-5]

        # Различные проблемные имена
        names = [
            f"mirror1 {i} {i+1} {i+2} {i+3} {i+4} mirror2 {-i} {-i-1} {-i-2} {-i-3} {-i-4}.png",
            f"mirror1 ax {i} ay {i+1} lx {i+2} ly {i+3} lz {i+4} mirror2 ax {-i} ay {-i-1} lx {-i-2} ly {-i-3} lz {-i-4}.png",
            f"interferogram #{i} data.png",
            f"Korsch System - Mirror 1 ({i},{i+1},{i+2},{i+3},{i+4}) Mirror 2 ({-i},{-i-1},{-i-2},{-i-3},{-i-4}).png",
            f"Test Data (copy {i}).png"
        ]

        # Выбираем случайное проблемное имя
        if i < len(names):
            filename = names[i]
        else:
            filename = f"test_file_{i}.png"

        filepath = directory / filename
        cv2.imwrite(str(filepath), img)
        print(f"Created: {filename}")


def main():
    """
    Основная функция.
    """
    parser = argparse.ArgumentParser(description="Fix interferogram filenames with spaces and special characters")
    parser.add_argument("directory", help="Directory with interferogram files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    parser.add_argument("--execute", action="store_true", help="Actually rename files")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup")
    parser.add_argument("--create-samples", type=int, metavar="N", help="Create N sample files with problematic names")

    args = parser.parse_args()

    if args.create_samples:
        print(f"🎨 Creating {args.create_samples} sample files with problematic names...")
        create_sample_data(args.directory, args.create_samples)
        return

    # По умолчанию - dry run для безопасности
    dry_run = not args.execute
    backup = not args.no_backup

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be renamed")
        print("   Use --execute to actually rename files")
        print()

    # Обработка директории
    renamed, skipped, errors = process_directory(
        args.directory,
        dry_run=dry_run,
        backup=backup
    )

    if errors > 0:
        print(f"\n⚠️  Completed with {errors} errors")
    else:
        print(f"\n✅ Completed successfully!")


if __name__ == "__main__":
    main()