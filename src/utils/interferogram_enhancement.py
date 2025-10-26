"""
Interferogram Resolution Enhancement Module

Анализ и улучшение разрешения интерферограмм для достижения субмикронной точности.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ResolutionConfig:
    """Конфигурация для анализа и улучшения разрешения."""
    target_resolution: int = 1024       # Целевое разрешение
    min_resolution: int = 256            # Минимальное разрешение
    analysis_threshold: float = 0.8        # Порог анализа качества
    preserve_fringes: bool = True         # Сохранение интерференционных полос
    enhancement_method: str = "sr_cnn"   # sr_cnn, esrgan, lanczos, bicubic


class InterferogramAnalyzer:
    """Анализатор качества и достаточности разрешения интерферограмм."""

    def __init__(self, min_fringe_spacing: float = 2.0):
        """
        Args:
            min_fringe_spacing: Минимальное расстояние между полосами в пикселях
        """
        self.min_fringe_spacing = min_fringe_spacing

    def analyze_resolution_adequacy(
        self,
        interferogram: np.ndarray,
        target_precision_um: float = 1.0
    ) -> Dict[str, Any]:
        """
        Анализ достаточности текущего разрешения для достижения заданной точности.

        Args:
            interferogram: Интерферограмма для анализа
            target_precision_um: Целевая точность в микронах

        Returns:
            Dict с результатами анализа
        """
        height, width = interferogram.shape
        resolution_score = self._calculate_resolution_score(interferogram)

        # Анализ частоты интерференционных полос
        fringe_analysis = self._analyze_fringe_frequency(interferogram)

        # Оценка влияния разрешения на точность
        precision_capability = self._estimate_precision_capability(
            height, width, fringe_analysis['dominant_wavelength_px']
        )

        # Рекомендации по разрешению
        recommendations = self._generate_resolution_recommendations(
            resolution_score, precision_capability, target_precision_um
        )

        return {
            'current_resolution': (height, width),
            'resolution_score': resolution_score,
            'fringe_analysis': fringe_analysis,
            'precision_capability_um': precision_capability,
            'target_precision_um': target_precision_um,
            'adequate_for_target': precision_capability <= target_precision_um,
            'recommendations': recommendations,
            'upscaling_needed': precision_capability > target_precision_um
        }

    def _calculate_resolution_score(self, interferogram: np.ndarray) -> float:
        """Расчет оценки качества разрешения."""
        # Анализ частотных характеристик
        fft = np.fft.fft2(interferogram.astype(np.float32))
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        # Определение максимальной пространственной частоты
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        radius = min(center_y, center_x) // 4  # Анализ высоких частот

        # Создание маски для высоких частот
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 >= radius**2

        high_freq_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)

        # Нормализованная оценка
        resolution_score = high_freq_energy / (total_energy + 1e-8)

        return float(resolution_score)

    def _analyze_fringe_frequency(self, interferogram: np.ndarray) -> Dict[str, Any]:
        """Анализ частоты интерференционных полос."""
        # Вычисление 2D FFT для анализа частот
        fft = np.fft.fft2(interferogram.astype(np.float32))
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        # Поиск доминантной частоты (исключая DC компонент)
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2

        # Создание маски для исключения центральной области
        mask_diameter = min(magnitude.shape) // 10
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        mask = ((x - center_x)**2 + (y - center_y)**2) > mask_diameter**2

        masked_magnitude = magnitude * mask

        # Поиск пикселя с максимальной амплитудой
        max_idx = np.unravel_index(np.argmax(masked_magnitude), masked_magnitude.shape)
        freq_y, freq_x = max_idx[0] - center_y, max_idx[1] - center_x

        # Расчет длины волны в пикселях
        if freq_x == 0 and freq_y == 0:
            dominant_wavelength_px = float('inf')
        else:
            freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)
            dominant_wavelength_px = magnitude.shape[1] / (2 * freq_magnitude + 1e-8)

        return {
            'dominant_frequency_x': freq_x,
            'dominant_frequency_y': freq_y,
            'dominant_wavelength_px': dominant_wavelength_px,
            'peak_magnitude': masked_magnitude[max_idx],
            'high_frequency_content': np.sum(masked_magnitude) / (np.sum(magnitude) + 1e-8)
        }

    def _estimate_precision_capability(
        self,
        height: int,
        width: int,
        fringe_wavelength_px: float
    ) -> float:
        """
        Оценка достижимой точности при текущем разрешении.

        Args:
            height, width: Размеры изображения
            fringe_wavelength_px: Длина волны интерференционных полос в пикселях

        Returns:
            Достижимая точность в микронах
        """
        if fringe_wavelength_px == float('inf') or fringe_wavelength_px <= 0:
            # Нет четких интерференционных полос
            return float('inf')

        # Максимальная детектируемая фаза ~ 1/10 длины волны
        min_detectable_phase_fraction = 0.1

        # Расчет физического разрешения
        # Предполагаем, что 1 пиксель соответствует ~1 мкм (типичное значение для интерферометрии)
        pixel_size_um = 1.0

        # Минимальное обнаруживаемое смещение в пикселях
        min_detectable_pixels = min_detectable_phase_fraction * fringe_wavelength_px

        # Преобразование в микрометры
        precision_capability_um = min_detectable_pixels * pixel_size_um

        # Учет общего размера изображения
        size_factor = min(height, width) / 1024.0  # Нормализация к 1024
        precision_capability_um /= size_factor

        return float(precision_capability_um)

    def _generate_resolution_recommendations(
        self,
        resolution_score: float,
        precision_capability_um: float,
        target_precision_um: float
    ) -> Dict[str, Any]:
        """Генерация рекомендаций по улучшению разрешения."""
        recommendations = {
            'need_upscaling': False,
            'recommended_resolution': None,
            'upscaling_factor': None,
            'quality_preservation_tips': [],
            'alternative_methods': []
        }

        # Анализ необходимости повышения разрешения
        if precision_capability_um > target_precision_um:
            recommendations['need_upscaling'] = True

            # Расчет необходимого фактора увеличения
            required_factor = precision_capability_um / target_precision_um
            recommended_factor = max(2.0, np.ceil(required_factor))

            recommendations['upscaling_factor'] = int(recommended_factor)

            # Рекомендуемое разрешение
            current_min = min(256, resolution_score * 1024)  # Оценка текущего эффективного разрешения
            recommended_resolution = int(current_min * recommended_factor)
            recommended_resolution = min(2048, recommended_resolution)  # Ограничение сверху

            recommendations['recommended_resolution'] = recommended_resolution

        # Советы по сохранению качества
        if resolution_score < 0.5:
            recommendations['quality_preservation_tips'].append(
                "Low frequency content detected - consider noise reduction preprocessing"
            )

        if resolution_score < 0.3:
            recommendations['quality_preservation_tips'].append(
                "Very low resolution - consider super-resolution methods with fringe preservation"
            )

        # Альтернативные методы
        recommendations['alternative_methods'] = [
            "CNN-based super-resolution with frequency domain constraints",
            "Interpolation with fringe-aware algorithms",
            "Multi-scale fusion approach",
            "Phase retrieval enhancement"
        ]

        return recommendations


class InterferogramUpscaler(nn.Module):
    """
    Нейросетевой апскейлер для интерферограмм с сохранением интерференционных полос.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        input_channels: int = 1,
        feature_channels: int = 64,
        num_layers: int = 8,
        preserve_fringes: bool = True
    ):
        """
        Args:
            scale_factor: Коэффициент увеличения (2, 4, 8)
            input_channels: Количество входных каналов
            feature_channels: Количество признаковых каналов
            num_layers: Количество сверточных слоев
            preserve_fringes: Специальное сохранение интерференционных полос
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.preserve_fringes = preserve_fringes

        # Feature extraction backbone
        backbone_layers = []
        in_channels = input_channels

        for i in range(num_layers):
            out_channels = feature_channels * (2 ** min(i // 2, 2))
            backbone_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        self.backbone = nn.Sequential(*backbone_layers)

        # Frequency domain preservation branch
        if preserve_fringes:
            self.freq_branch = nn.Sequential(
                nn.Conv2d(input_channels, feature_channels // 2, 1),
                nn.ReLU(inplace=True)
            )

        # Upsampling layers
        upsample_layers = []
        current_channels = in_channels

        for _ in range(int(np.log2(scale_factor))):
            upsample_layers.extend([
                nn.ConvTranspose2d(current_channels, current_channels // 2, 2, stride=2),
                nn.BatchNorm2d(current_channels // 2),
                nn.ReLU(inplace=True)
            ])
            current_channels //= 2

        self.upsampler = nn.Sequential(*upsample_layers)

        # Final reconstruction layer
        final_input_channels = current_channels + (feature_channels // 2 if preserve_fringes else 0)
        self.final_conv = nn.Conv2d(final_input_channels, input_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass апскейлера.

        Args:
            x: Входная интерферограмма [B, C, H, W]

        Returns:
            Upscaled интерферограмма [B, C, H*scale, W*scale]
        """
        # Feature extraction
        features = self.backbone(x)

        # Frequency preservation
        if self.preserve_fringes:
            freq_features = self.freq_branch(x)
            # Upsample frequency features to match feature map size
            target_size = features.shape[2:]
            freq_features = F.interpolate(freq_features, size=target_size, mode='bilinear', align_corners=False)
            features = torch.cat([features, freq_features], dim=1)

        # Progressive upsampling
        upsampled = self.upsampler(features)

        # Final reconstruction
        if self.preserve_fringes:
            # Upsample frequency features to final size
            final_size = (x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor)
            freq_upsampled = F.interpolate(freq_features, size=final_size, mode='bilinear', align_corners=False)
            upsampled = torch.cat([upsampled, freq_upsampled], dim=1)

        output = self.final_conv(upsampled)

        return output


def create_interferogram_upsampler(
    method: str = "sr_cnn",
    scale_factor: int = 4,
    target_resolution: Optional[int] = None
) -> nn.Module:
    """
    Создает апскейлер для интерферограмм.

    Args:
        method: Метод апскейлинга ('sr_cnn', 'lanczos', 'bicubic', 'esrgan')
        scale_factor: Коэффициент увеличения
        target_resolution: Целевое разрешение (если None, используется scale_factor)

    Returns:
        Модель апскейлинга
    """
    if method == "sr_cnn":
        return InterferogramUpscaler(scale_factor=scale_factor, preserve_fringes=True)
    elif method == "lanczos":
        # Возвращаем функцию для Lanczos интерполяции
        return lambda x: F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    elif method == "bicubic":
        return lambda x: F.interpolate(x, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    else:
        raise ValueError(f"Unknown upsampling method: {method}")


def enhance_interferogram_batch(
    interferograms: torch.Tensor,
    config: ResolutionConfig,
    upscaler: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Пакетное улучшение интерферограмм.

    Args:
        interferograms: Пакет интерферограмм [B, C, H, W]
        config: Конфигурация улучшения
        upscaler: Предварительно загруженный апскейлер

    Returns:
        Улучшенные интерферограммы
    """
    batch_size, channels, height, width = interferograms.shape

    # Определение необходимости апскейлинга
    current_min = min(height, width)
    if current_min >= config.target_resolution:
        return interferograms  # Апскейлинг не требуется

    # Расчет необходимого фактора увеличения
    scale_factor = max(2, int(np.ceil(config.target_resolution / current_min)))
    scale_factor = min(scale_factor, 8)  # Ограничение максимального увеличения

    if upscaler is not None:
        with torch.no_grad():
            enhanced = upscaler(interferograms)
    else:
        # Использование билинейной интерполяции как fallback
        enhanced = F.interpolate(
            interferograms,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )

    # Обрезка до целевого разрешения
    target_height = min(config.target_resolution, enhanced.shape[2])
    target_width = min(config.target_resolution, enhanced.shape[3])

    enhanced = enhanced[:, :, :target_height, :target_width]

    return enhanced


def analyze_resolution_requirements(
    target_precision_um: float = 1.0,
    wavelength_um: float = 0.55,  # Длина волны лазера в мкм
    optical_magnification: float = 10.0
) -> Dict[str, Any]:
    """
    Анализ требований к разрешению для достижения заданной точности.

    Args:
        target_precision_um: Целевая точность в микронах
        wavelength_um: Длина волны лазера
        optical_magnification: Оптическое увеличение

    Returns:
        Рекомендации по разрешению
    """
    # Расчет необходимого разрешения датчика
    # Для интерферометрии: Δφ = 2π * ΔOPD / λ
    # Где ΔOPD - изменение оптического пути
    required_phase_resolution = 2 * np.pi * target_precision_um / wavelength_um

    # Минимальное разрешение для детекции фазовых изменений
    min_required_resolution_um = wavelength_um / (2 * required_phase_resolution)

    # Преобразование в пиксели с учетом оптического увеличения
    min_pixels_per_feature = wavelength_um / (min_required_resolution_um * optical_magnification)

    # Рекомендуемое разрешение изображения
    recommended_resolution = int(min_pixels_per_feature * 10)  # 10x для надежности
    recommended_resolution = min(2048, max(512, recommended_resolution))

    return {
        'target_precision_um': target_precision_um,
        'wavelength_um': wavelength_um,
        'optical_magnification': optical_magnification,
        'required_phase_resolution_rad': required_phase_resolution,
        'min_sensor_resolution_um': min_required_resolution_um,
        'min_pixels_per_feature': min_pixels_per_feature,
        'recommended_image_resolution': recommended_resolution,
        'analysis_256_sufficient': recommended_resolution <= 256,
        'analysis_512_sufficient': recommended_resolution <= 512,
        'analysis_1024_recommended': recommended_resolution > 512
    }


if __name__ == "__main__":
    # Тестирование модуля
    analyzer = InterferogramAnalyzer()

    # Создание тестовой интерферограммы
    y, x = np.meshgrid(np.linspace(0, 256, 256), np.linspace(0, 256, 256))
    interferogram = 128 + 127 * np.sin(2 * np.pi * x / 50) * np.cos(2 * np.pi * y / 60)

    # Анализ разрешения
    analysis = analyzer.analyze_resolution_adequacy(interferogram, target_precision_um=1.0)
    print("Resolution Analysis:")
    for key, value in analysis.items():
        if key != 'recommendations':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for rec_key, rec_value in value.items():
                print(f"    {rec_key}: {rec_value}")

    # Анализ требований к разрешению
    requirements = analyze_resolution_requirements(target_precision_um=1.0)
    print("\nResolution Requirements:")
    for key, value in requirements.items():
        print(f"  {key}: {value}")