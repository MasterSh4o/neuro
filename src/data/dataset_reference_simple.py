"""
Simple Dataset with Reference Interferogram Support

Упрощенный датасет с поддержкой эталонной интерферограммы
для достижения субмикронной точности.

Поддерживаемые форматы image_size:
- int: 512 (преобразуется в (512, 512))
- tuple: (512, 512)
- list: [512, 512]
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple, Union, List

from utils.reference_processor import ReferenceProcessor


class ReferenceInterferogramDataset(Dataset):
    """
    Упрощенный датасет с поддержкой эталонной интерферограммы.

    Режимы работы:
    - Normal: обучение на абсолютных интерферограммах
    - Difference: обучение на разнице (текущая - эталонная)
    - Dual: двойной вход сети (текущая + эталонная)
    """

    def __init__(
        self,
        root: str,
        img_glob: str = "**/*.png,**/*.jpg,**/*.tif,**/*.bmp",
        image_size: Union[int, Tuple[int, int], List[int]] = (512, 512),
        *,
        # Korsch параметры
        korsch_mode: bool = True,
        bits_per_number: int = 7,
        linear_range_um: float = 1000.0,
        angular_range_arcsec: float = 60.0,

        # Параметры эталонной интерферограммы
        reference_path: Optional[str] = None,
        mode: str = "normal",  # "normal", "difference", "dual"
        reference_preprocessing: Optional[Dict[str, Any]] = None,

        # Разрешение и аугментации
        resolution_enhancement: bool = True,
        target_resolution: Optional[int] = None,
        enable_augmentations: bool = True,
    ):
        super().__init__()
        self.root = os.path.abspath(root)

        # Преобразование image_size в tuple для совместимости
        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError(f"image_size must be positive integer, got {image_size}")
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)):
            if len(image_size) != 2:
                raise ValueError(f"image_size must have exactly 2 elements, got {len(image_size)}")
            if any(size <= 0 for size in image_size):
                raise ValueError(f"All image_size elements must be positive, got {image_size}")
            self.image_size = tuple(image_size)
        else:
            raise TypeError(f"image_size must be int or tuple/list, got {type(image_size)}")

        self.img_glob = img_glob

        # Валидация основных параметров
        if not isinstance(img_glob, str):
            raise TypeError(f"img_glob must be a string, got {type(img_glob)}")
        if not img_glob.strip():
            raise ValueError("img_glob cannot be empty")

        # Валидация Korsch параметров
        if not isinstance(korsch_mode, bool):
            raise TypeError(f"korsch_mode must be boolean, got {type(korsch_mode)}")
        if not isinstance(bits_per_number, int) or bits_per_number <= 0:
            raise ValueError(f"bits_per_number must be positive integer, got {bits_per_number}")
        if not isinstance(linear_range_um, (int, float)) or linear_range_um <= 0:
            raise ValueError(f"linear_range_um must be positive number, got {linear_range_um}")
        if not isinstance(angular_range_arcsec, (int, float)) or angular_range_arcsec <= 0:
            raise ValueError(f"angular_range_arcsec must be positive number, got {angular_range_arcsec}")

        # Korsch параметры
        self.korsch_mode = korsch_mode
        self.bits_per_number = bits_per_number
        self.linear_range_um = linear_range_um
        self.angular_range_arcsec = angular_range_arcsec

        # Валидация параметров эталона
        if mode not in ["normal", "difference", "dual"]:
            raise ValueError(f"mode must be 'normal', 'difference', or 'dual', got '{mode}'")
        if reference_path is not None and not isinstance(reference_path, str):
            raise TypeError(f"reference_path must be string or None, got {type(reference_path)}")
        if reference_preprocessing is not None and not isinstance(reference_preprocessing, dict):
            raise TypeError(f"reference_preprocessing must be dictionary or None, got {type(reference_preprocessing)}")

        # Параметры эталона
        self.mode = mode
        self.reference_path = reference_path
        self.reference_processor = None

        # Инициализация процессора эталона
        if self.reference_path is not None:
            self.reference_processor = ReferenceProcessor(
                reference_path=self.reference_path,
                target_size=self.image_size,
                preprocessing=reference_preprocessing
            )

        # Валидация параметров разрешения и аугментаций
        if not isinstance(resolution_enhancement, bool):
            raise TypeError(f"resolution_enhancement must be boolean, got {type(resolution_enhancement)}")
        if target_resolution is not None and (not isinstance(target_resolution, int) or target_resolution <= 0):
            raise ValueError(f"target_resolution must be positive integer or None, got {target_resolution}")
        if not isinstance(enable_augmentations, bool):
            raise TypeError(f"enable_augmentations must be boolean, got {type(enable_augmentations)}")

        # Разрешение и аугментации
        self.resolution_enhancement = resolution_enhancement
        self.target_resolution = target_resolution
        self.enable_augmentations = enable_augmentations

        # Нормализация для совместимости с train.py
        self.norm_type = "none"  # Reference dataset handles its own normalization
        self.global_mean = None
        self.global_std = None

        # Валидация и загрузка файлов
        self._validate_and_load_dataset()

    def _validate_and_load_dataset(self) -> None:
        """Валидация и загрузка набора данных с улучшенной диагностикой."""
        # Проверка существования директории
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"Path is not a directory: {self.root}")

        # Валидация и нормализация шаблонов
        patterns = self._validate_and_normalize_patterns()

        # Поиск файлов с диагностикой
        discovered = self._discover_files_with_diagnostic(patterns)

        if not discovered:
            # Предложить альтернативы и решения
            self._handle_no_files_found(patterns)

        print(f"[INFO] Found {len(discovered)} images in dataset")
        self.files = discovered
        self.paths = [os.path.relpath(f, self.root) for f in self.files]

        # Парсинг меток
        self._parse_labels()

    def _validate_and_normalize_patterns(self) -> list:
        """Валидация и нормализация шаблонов поиска."""
        raw_patterns = str(self.img_glob).replace(";", ",").split(",")
        patterns = [p.strip() for p in raw_patterns if p.strip()]

        if not patterns:
            raise ValueError("No search patterns provided in img_glob")

        # Валидация шаблонов
        valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        for pattern in patterns:
            if not any(pattern.lower().endswith(ext) for ext in valid_extensions):
                if not pattern.startswith('**/'):
                    print(f"[WARNING] Pattern '{pattern}' might not match image files")

        print(f"[DEBUG] Using patterns: {patterns}")
        return patterns

    def _discover_files_with_diagnostic(self, patterns: list) -> list:
        """Поиск файлов с подробной диагностикой."""
        print(f"[DEBUG] Searching for images in: {self.root}")

        discovered = []
        pattern_results = []

        for pat in patterns:
            search_path = os.path.join(self.root, pat)
            print(f"[DEBUG] Searching pattern: {search_path}")

            try:
                matches = glob.glob(search_path, recursive=True)
                print(f"[DEBUG] Found {len(matches)} files with pattern '{pat}'")

                # Фильтрация только файлов изображений
                valid_matches = []
                image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
                for match in matches:
                    if os.path.isfile(match):
                        if any(match.lower().endswith(ext) for ext in image_extensions):
                            valid_matches.append(match)

                if len(valid_matches) != len(matches):
                    print(f"[DEBUG] Filtered {len(matches) - len(valid_matches)} non-image files")

                discovered.extend(valid_matches)
                pattern_results.append({
                    'pattern': pat,
                    'matches': len(valid_matches),
                    'examples': [os.path.basename(m) for m in valid_matches[:3]]
                })

            except Exception as e:
                print(f"[ERROR] Error searching pattern '{pat}': {e}")

        # Удаление дубликатов и сортировка
        discovered = sorted(set([os.path.abspath(f) for f in discovered if os.path.exists(f)]))

        # Показать результаты по шаблонам
        print(f"[DEBUG] Pattern summary:")
        for pr in pattern_results:
            status = "✅" if pr['matches'] > 0 else "❌"
            examples = f" ({', '.join(pr['examples'])})" if pr['examples'] else ""
            print(f"   {status} {pr['pattern']}: {pr['matches']} files{examples}")

        return discovered

    def _handle_no_files_found(self, patterns: list) -> None:
        """Обработка случая, когда файлы не найдены."""
        print(f"[DEBUG] No files found. Analyzing directory...")

        # Анализ директории
        try:
            contents = os.listdir(self.root)
            print(f"[DEBUG] Directory contains {len(contents)} items")

            # Поиск изображений независимо от шаблонов
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
            found_images = []
            subdirectories = []

            for item in contents:
                item_path = os.path.join(self.root, item)
                if os.path.isdir(item_path):
                    subdirectories.append(item)
                elif any(item.lower().endswith(ext) for ext in image_extensions):
                    found_images.append(item)

            print(f"[DEBUG] Found {len(found_images)} image files, {len(subdirectories)} subdirectories")

            if found_images:
                print(f"[DEBUG] Image files found: {found_images[:5]}")
                # Предложить лучшие шаблоны
                extensions_found = set()
                for img in found_images:
                    if '.' in img:
                        extensions_found.add(img.split('.')[-1].lower())

                suggested_patterns = []
                for ext in extensions_found:
                    suggested_patterns.append(f"**/*.{ext}")

                print(f"[DEBUG] Suggested patterns: {suggested_patterns}")
                print(f"[DEBUG] Try updating your config: img_glob: {suggested_patterns}")

            elif subdirectories:
                print(f"[DEBUG] Subdirectories: {subdirectories[:5]}")
                print(f"[DEBUG] Images might be in subdirectories. Try patterns: ['**/*.png']")

            else:
                print(f"[DEBUG] Directory appears to be empty or contains no image files")

        except Exception as e:
            print(f"[DEBUG] Could not analyze directory: {e}")

        # Финальное сообщение об ошибке с советами
        error_msg = f"No images found in '{self.root}' with patterns: {patterns}"
        suggestions = [
            "1. Check if the dataset directory exists and contains image files",
            "2. Verify the image patterns match your file names",
            "3. Use recursive patterns like '**/*.png' to search subdirectories",
            "4. Run: python diagnose_dataset.py '{}' for detailed diagnosis".format(self.root)
        ]

        error_msg_with_suggestions = error_msg + "\n" + "\n".join(f"Suggestion: {s}" for s in suggestions)
        raise RuntimeError(error_msg_with_suggestions)

    def _parse_labels(self) -> None:
        """Парсинг меток из имен файлов."""
        self.labels = []
        for path in self.files:
            base = os.path.splitext(os.path.basename(path))[0]

            if self.korsch_mode:
                # Парсинг смещений Корша
                try:
                    # Импортируем функцию парсинга
                    from .dataset import parse_korsch_displacements
                    binary_vector, physical_values = parse_korsch_displacements(
                        base,
                        mirror_count=2,
                        bits_per_parameter=self.bits_per_number,
                        linear_range_um=self.linear_range_um,
                        angular_range_arcsec=self.angular_range_arcsec,
                        value_mode="mod"
                    )
                    self.labels.append(binary_vector)
                except Exception as e:
                    print(f"[WARNING] Failed to parse Korsch displacements from '{base}': {e}")
                    # Создаем нулевой вектор как запасной вариант
                    K = 10 * self.bits_per_number  # 2 mirrors × 5 parameters × bits
                    self.labels.append(np.zeros(K, dtype=np.float32))
            else:
                # Запасной вариант - простые числа из имени файла
                import re
                numbers = re.findall(r'\d+', base)
                if numbers:
                    extracted_numbers = [float(n) for n in numbers[:10]]  # Максимум 10 параметров
                else:
                    extracted_numbers = [0.0] * 10  # Запасной вариант

                # Преобразуем в биты для совместимости с korsch_mode
                binary_vector = []
                for num in extracted_numbers:
                    for bit_pos in range(self.bits_per_number):
                        binary_vector.append((int(num) >> bit_pos) & 1)
                self.labels.append(np.array(binary_vector, dtype=np.float32))

        self.labels = np.array(self.labels, dtype=np.float32)
        self.K = int(self.labels.shape[1])  # Добавляем атрибут K для совместимости с train.py
        print(f"[DEBUG] Parsed {len(self.labels)} labels, K={self.K}")

    def _load_dataset(self) -> None:
        """Загрузка набора данных (устаревший метод для совместимости)."""
        self._validate_and_load_dataset()

    def _parse_korsch_displacement(self, base: str) -> np.ndarray:
        """Парсинг смещений Корша из имени файла."""
        # Поддерживаемые форматы:
        # 1. Int_  0_  0_ 31_ 10_ 22_  6_ 27_ 17_  3_ 23_0.bmp
        # 2. mirror1_ax_ay_lx_ly_lz_mirror2_ax_ay_lx_ly_lz.png

        # Сначала пытаемся формат "Int_"
        if base.startswith('Int_'):
            # Извлекаем все числа после "Int_"
            numbers_str = base[4:]  # убираем "Int_"
            numbers = []

            # Разделяем по подчеркиваниям и фильтруем пустые строки
            parts = [p.strip() for p in numbers_str.split('_') if p.strip()]

            for part in parts:
                try:
                    numbers.append(int(part))
                except ValueError:
                    continue

            # Ожидаем 10 параметров (2 зеркала × 5 параметров)
            expected_numbers = 10
            while len(numbers) < expected_numbers:
                numbers.append(0)

            # Ограничение первыми 10 числами
            numbers = numbers[:expected_numbers]

        else:
            # Стандартный формат с "_mirror"
            parts = base.split('_mirror')
            if len(parts) < 2:
                # Если не подошло ни один формат, используем стандартный парсинг
                return self._parse_standard_label(base)

            numbers = []
            for mirror_part in parts[1:]:
                if not mirror_part:
                    continue

                params = mirror_part.split('_')
                if len(params) != 5:
                    continue

                for param_str in params:
                    try:
                        numbers.append(int(param_str))
                    except ValueError:
                        numbers.append(0)

            # Заполняем недостающие параметры нулями
            expected_numbers = 10  # 2 зеркала × 5 параметров
            while len(numbers) < expected_numbers:
                numbers.append(0)
            numbers = numbers[:expected_numbers]

        # Преобразование в биты
        binary_vector = []
        for num in numbers:
            # Ограничение диапазона
            max_value = (1 << self.bits_per_number) - 1
            param_value = num & max_value

            # Преобразование в бинарный вектор
            for bit_pos in range(self.bits_per_number):
                binary_vector.append((param_value >> bit_pos) & 1)

        return np.array(binary_vector, dtype=np.float32)

    def set_global_stats(self, mean: float, std: float) -> None:
        """Установка глобальных статистик для совместимости."""
        self.global_mean = mean
        self.global_std = std

    def compute_global_stats(self, indices: np.ndarray) -> Tuple[float, float]:
        """Вычисление глобальных статистик для совместимости."""
        # Возвращаем стандартные значения, т.к. эталонный датасет сам обрабатывает нормализацию
        # indices не используются, но нужны для совместимости интерфейса
        _ = indices  # Подавляем предупреждение о неиспользуемом параметре
        return 0.0, 1.0

    def make_subset(self, indices: np.ndarray, augment: bool = False, share_stats: bool = True) -> 'ReferenceInterferogramDataset':
        """Создание поднабора для совместимости с train.py."""
        subset = ReferenceInterferogramDataset(
            root=self.root,
            img_glob=self.img_glob,
            image_size=self.image_size,
            korsch_mode=self.korsch_mode,
            bits_per_number=self.bits_per_number,
            linear_range_um=self.linear_range_um,
            angular_range_arcsec=self.angular_range_arcsec,
            reference_path=self.reference_path,
            mode=self.mode,
            reference_preprocessing=self.reference_processor.preprocessing if self.reference_processor else None,
            resolution_enhancement=self.resolution_enhancement,
            target_resolution=self.target_resolution,
            enable_augmentations=augment,
        )

        # Фильтруем файлы и метки по индексам
        subset.files = [self.files[i] for i in indices]
        subset.paths = [self.paths[i] for i in indices]
        subset.labels = self.labels[indices]

        # Копируем нормализацию
        if share_stats and hasattr(self, 'global_mean'):
            subset.global_mean = self.global_mean
            subset.global_std = self.global_std

        # Копируем атрибут K
        if hasattr(self, 'K'):
            subset.K = self.K

        return subset

    def _parse_standard_label(self, base: str) -> np.ndarray:
        """Стандартный парсинг меток."""
        # Простое извлечение чисел и преобразование в биты
        import re
        numbers = re.findall(r'(\d+)', base)
        expected_numbers = 10  # 2 зеркала × 5 параметров

        # Заполнение недостающих чисел нулями
        while len(numbers) < expected_numbers:
            numbers.append(0)

        binary_vector = []
        for num in numbers[:expected_numbers]:
            for bit_pos in range(self.bits_per_number):
                binary_vector.append((num >> bit_pos) & 1)

        return np.array(binary_vector, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def _load_and_process_image(self, path: str) -> np.ndarray:
        """Загрузка и обработка изображения."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # Изменение размера
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        # Улучшение разрешения если нужно
        if self.resolution_enhancement and self.target_resolution is not None:
            if max(img.shape[:2]) < self.target_resolution:
                # Простое увеличение
                scale_factor = max(2, self.target_resolution // max(img.shape[:2]))
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_CUBIC)

        return img

    def _apply_light_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Легкие аугментации для обучения."""
        if not self.enable_augmentations:
            return img

        # Случайный поворот (малые углы)
        if np.random.random() < 0.1:
            angle = np.random.uniform(-5, 5)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # Случайный шум
        if np.random.random() < 0.1:
            noise = np.random.normal(0, 0.005, img.shape).astype(np.float32)
            img = img + noise

        # Яркостная и контрастная коррекция
        if np.random.random() < 0.05:
            brightness = 1.0 + np.random.uniform(-0.1, 0.1)
            contrast = 1.0 + np.random.uniform(-0.1, 0.1)
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness - 1.0)

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Получение элемента датасета."""
        # Загрузка и обработка изображения
        current_img = self._load_and_process_image(self.files[idx])

        # Применение аугментаций (только для обучения)
        img = self._apply_light_augmentations(current_img)

        # Преобразование в тензор
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
        label_array = self.labels[idx].astype(np.float32, copy=False)
        label_tensor = torch.from_numpy(label_array).float()

        # Обработка с эталонной интерферограммой
        mode_info = {
            'mode': self.mode,
            'has_reference': self.reference_processor is not None,
            'reference_path': self.reference_path,
            'image_path': self.paths[idx],
            'label': label_array,
            'label_dim': int(label_array.shape[0]),
        }

        if self.mode == "difference" and self.reference_processor is not None:
            # Вычисление разности
            difference = self.reference_processor.compute_difference(current_img)

            # Преобразование в тензор с правильной размерностью для consistency
            difference_tensor = torch.from_numpy(difference).float().unsqueeze(0)  # [1, H, W]

            return img_tensor, difference_tensor, mode_info

        elif self.mode == "dual" and self.reference_processor is not None:
            # Создание двойного входа
            dual_input = self.reference_processor.create_dual_input(current_img)

            # Преобразование в тензор с правильной размерностью для channels_last
            dual_tensor = torch.from_numpy(dual_input).float()  # [2, H, W]

            return img_tensor, dual_tensor, mode_info

        else:
            # Стандартный режим
            return img_tensor, label_tensor, mode_info

    def get_reference_info(self) -> Optional[Dict[str, Any]]:
        """Получение информации об эталонной интерферограмме."""
        if self.reference_processor is not None:
            return self.reference_processor.get_reference_info()
        return None

    def get_dataset_info(self) -> Dict[str, Any]:
        """Получение информации о датасете."""
        return {
            'total_images': len(self.files),
            'image_size': self.image_size,
            'korsch_mode': self.korsch_mode,
            'bits_per_number': self.bits_per_number,
            'mode': self.mode,
            'has_reference': self.reference_processor is not None,
            'reference_path': self.reference_path,
            'resolution_enhancement': self.resolution_enhancement,
            'target_resolution': self.target_resolution,
            'enable_augmentations': self.enable_augmentations
        }


# Пример использования
if __name__ == "__main__":
    # Создание датасета с эталонной интерферограммой
    dataset = ReferenceInterferogramDataset(
        root="path/to/interferograms",
        reference_path="reference_ideal.png",
        mode="difference",
        korsch_mode=True,
        bits_per_number=8,
        image_size=(512, 512),
        resolution_enhancement=True,
        target_resolution=1024
    )

    print(f"Dataset created with {len(dataset)} images")
    print(f"Dataset info: {dataset.get_dataset_info()}")

    if dataset.get_reference_info():
        print(f"Reference info: {dataset.get_reference_info()}")

    # Тестирование загрузки
    img, diff, info = dataset[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample difference shape: {diff.shape}")
    print(f"Mode info: {info}")

    print("✅ Dataset with reference support is ready!")