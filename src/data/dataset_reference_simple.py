"""
Simple Dataset with Reference Interferogram Support

Упрощенный датасет с поддержкой эталонной интерферограммы
для достижения субмикронной точности.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple

from .reference_processor import ReferenceProcessor


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
        image_size: Tuple[int, int] = (512, 512),
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
        self.image_size = image_size
        self.img_glob = img_glob

        # Korsch параметры
        self.korsch_mode = korsch_mode
        self.bits_per_number = bits_per_number
        self.linear_range_um = linear_range_um
        self.angular_range_arcsec = angular_range_arcsec

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

        # Разрешение и аугментации
        self.resolution_enhancement = resolution_enhancement
        self.target_resolution = target_resolution
        self.enable_augmentations = enable_augmentations

        # Нормализация для совместимости с train.py
        self.norm_type = "none"  # Reference dataset handles its own normalization
        self.global_mean = None
        self.global_std = None

        # Загрузка файлов
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Загрузка набора данных."""
        # Поиск файлов
        patterns = [
            p.strip() for p in str(self.img_glob).replace(";", ",").split(",") if p.strip()
        ]
        discovered = []
        for pat in patterns:
            discovered.extend(
                os.path.join(self.root, pat) for pat in patterns
            )

        discovered = sorted(set([os.path.abspath(f) for f in discovered if os.path.exists(f)]))
        if not discovered:
            raise RuntimeError(f"No images found in '{self.root}' with patterns: {patterns}")

        self.files = discovered
        self.paths = [os.path.relpath(f, self.root) for f in self.files]
        print(f"Found {len(self.files)} images in dataset")

        # Парсинг меток
        self.labels = []
        for path in self.files:
            base = os.path.splitext(os.path.basename(path))[0]

            if self.korsch_mode:
                # Парсинг смещений Корша
                label = self._parse_korsch_displacement(base)
            else:
                # Стандартный парсинг
                label = self._parse_standard_label(base)

            self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"Parsed {len(self.labels)} labels for {self.bits_per_number} bits per parameter")

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
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

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
        label_tensor = torch.from_numpy(self.labels[idx]).float()

        # Обработка с эталонной интерферограммой
        mode_info = {
            'mode': self.mode,
            'has_reference': self.reference_processor is not None,
            'reference_path': self.reference_path,
            'image_path': self.paths[idx],
        }

        if self.mode == "difference" and self.reference_processor is not None:
            # Вычисление разности
            difference = self.reference_processor.compute_difference(current_img)

            # Преобразование в тензор
            difference_tensor = torch.from_numpy(difference).float().unsqueeze(0)  # [1, H, W]

            return img_tensor, difference_tensor, mode_info

        elif self.mode == "dual" and self.reference_processor is not None:
            # Создание двойного входа
            dual_input = self.reference_processor.create_dual_input(current_img)

            # Преобразование в тензор
            dual_tensor = torch.from_numpy(dual_input).float()  # [2, H, W]

            return dual_tensor, label_tensor, mode_info

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