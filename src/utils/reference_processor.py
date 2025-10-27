"""
Reference Interferogram Processor

Простая и надежная обработка эталонных интерферограмм для обучения с субмикронной точностью.
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union


class ReferenceProcessor:
    """
    Процессор эталонных интерферограмм для обучения с разностными данными.
    """

    def __init__(
        self,
        reference_path: str,
        target_size: Tuple[int, int] = (512, 512),
        preprocessing: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            reference_path: Путь к эталонной интерферограмме
            target_size: Целевой размер изображений
            preprocessing: Параметры препроцессинга эталона
        """
        self.reference_path = reference_path
        self.target_size = target_size
        self.preprocessing = preprocessing or {}

        # Загрузка и обработка эталона
        self.reference_image = self._load_and_preprocess_reference()

    def _load_and_preprocess_reference(self) -> np.ndarray:
        """
        Загрузка и препроцессинг эталонной интерферограммы.
        """
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference interferogram not found: {self.reference_path}")

        # Загрузка изображения
        img = cv2.imread(self.reference_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load reference image: {self.reference_path}")

        # Изменение размера
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        # Применение препроцессинга
        processed = img.astype(np.float32)

        # Нормализация интенсивности
        if self.preprocessing.get("normalize", True):
            processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)

        # Гауссово сглаживание
        if "gaussian_blur" in self.preprocessing:
            kernel_size = self.preprocessing["gaussian_blur"].get("kernel_size", 1)
            if kernel_size > 0 and kernel_size % 2 == 1:
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        # Улучшение контраста
        if "enhance_contrast" in self.preprocessing:
            factor = self.preprocessing["enhance_contrast"].get("factor", 1.2)
            if factor != 1.0:
                mean = processed.mean()
                processed = (processed - mean) * factor + mean

        # Коррекция яркости
        if "brightness_adjust" in self.preprocessing:
            offset = self.preprocessing["brightness_adjust"].get("offset", 0.0)
            if offset != 0.0:
                processed = np.clip(processed + offset, 0, 255)

        return processed

    def compute_difference(self, current_image: np.ndarray) -> np.ndarray:
        """
        Вычисление разности между текущей и эталонной интерферограммами.

        Args:
            current_image: Текущая интерферограмма

        Returns:
            Разностная интерферограмма (текущая - эталонная)
        """
        if current_image.shape != self.reference_image.shape:
            # Изменение размера эталона под текущую
            ref_resized = cv2.resize(
                self.reference_image,
                (current_image.shape[1], current_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            ref_resized = self.reference_image

        # Вычисление разности
        difference = current_image.astype(np.float32) - ref_resized.astype(np.float32)

        # Нормализация разности если нужно
        if self.preprocessing.get("normalize_difference", True):
            # Центрирование вокруг нуля
            diff_mean = np.mean(difference)
            diff_std = np.std(difference)
            if diff_std > 1e-8:
                difference = (difference - diff_mean) / diff_std
            else:
                difference = difference  # Уже нормализована

        # Ограничение разности для предотвращения выбросов
        if self.preprocessing.get("clip_difference", True):
            clip_value = self.preprocessing.get("clip_value", 3.0)
            difference = np.clip(difference, -clip_value, clip_value)

        return difference

    def create_dual_input(self, current_image: np.ndarray) -> np.ndarray:
        """
        Создание двойного входа (текущая + эталонная).

        Args:
            current_image: Текущая интерферограмма

        Returns:
            Двойной тензор [2, H, W]
        """
        if current_image.shape != self.reference_image.shape:
            # Изменение размера эталона под текущую
            ref_resized = cv2.resize(
                self.reference_image,
                (current_image.shape[1], current_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            ref_resized = self.reference_image

        # Создание двойного входа
        dual_input = np.stack([current_image, ref_resized], axis=0)  # [2, H, W]

        return dual_input

    def get_reference_info(self) -> Dict[str, Any]:
        """
        Получение информации об эталонной интерферограмме.
        """
        return {
            'reference_path': self.reference_path,
            'reference_shape': self.reference_image.shape,
            'target_size': self.target_size,
            'preprocessing': self.preprocessing,
            'reference_stats': {
                'mean': float(np.mean(self.reference_image)),
                'std': float(np.std(self.reference_image)),
                'min': float(np.min(self.reference_image)),
                'max': float(np.max(self.reference_image))
            }
        }

    def validate_reference_quality(self) -> Dict[str, Any]:
        """
        Валидация качества эталонной интерферограммы.
        """
        stats = self.get_reference_info()['reference_stats']

        # Простые метрики качества
        quality_score = {
            'contrast': float(stats['max'] - stats['min']) / stats['std'],
            'signal_to_noise': stats['std'] / (stats['mean'] + 1e-8),
            'dynamic_range': stats['std'] / (stats['mean'] + 1e-8),
            'overall_score': 0.0
        }

        # Оценка качества
        if quality_score['contrast'] > 10:  # Хороший контраст
            quality_score['overall_score'] += 2
        if quality_score['signal_to_noise'] < 0.1:  # Низкий шум
            quality_score['overall_score'] += 2

        quality_score['quality_level'] = 'excellent' if quality_score['overall_score'] >= 4 else 'good' if quality_score['overall_score'] >= 2 else 'acceptable'

        return quality_score


def create_reference_processor(
    reference_path: str,
    target_size: Tuple[int, int] = (512, 512),
    preprocessing: Optional[Dict[str, Any]] = None,
    validate_quality: bool = True
) -> ReferenceProcessor:
    """
    Фабричная функция для создания процессора эталонной интерферограммы.

    Args:
        reference_path: Путь к эталонной интерферограмме
        target_size: Целевой размер изображений
        preprocessing: Параметры препроцессинга
        validate_quality: Проводить валидацию качества

    Returns:
        Настроенный процессор эталонной интерферограммы
    """
    processor = ReferenceProcessor(
        reference_path=reference_path,
        target_size=target_size,
        preprocessing=preprocessing
    )

    if validate_quality:
        quality_info = processor.validate_reference_quality()
        print(f"Reference quality analysis:")
        for key, value in quality_info.items():
            print(f"  {key}: {value}")

        if quality_info.get('quality_level') in ['acceptable', 'poor']:
            print("⚠️ Warning: Reference quality may impact training performance")
        else:
            print("✅ Reference quality is suitable for training")

    return processor


if __name__ == "__main__":
    # Тестирование процессора эталонной интерферограммы
    processor = create_reference_processor(
        "reference.png",
        target_size=(512, 512),
        preprocessing={
            "normalize": True,
            "gaussian_blur": {"kernel_size": 1},
            "clip_difference": True
        }
    )

    print("Reference processor created successfully!")
    print(f"Reference info: {processor.get_reference_info()}")

    # Тестирование на случайном изображении
    test_image = np.random.randint(0, 255, (512, 512)).astype(np.uint8)
    test_image = test_image.astype(np.float32) / 255.0

    # Вычисление разности
    difference = processor.compute_difference(test_image)

    # Создание двойного входа
    dual_input = processor.create_dual_input(test_image)

    print(f"Test image shape: {test_image.shape}")
    print(f"Difference computation: OK, range: [{difference.min():.3f}, {difference.max():.3f}]")
    print(f"Dual input creation: OK, shape: {dual_input.shape}")