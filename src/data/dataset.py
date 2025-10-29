import os
import glob
from copy import deepcopy
from typing import Iterable, Sequence, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.path_utils import validate_path, safe_join, sanitize_path
from utils.memory_manager import ReferenceImageManager, default_reference_manager

# Импорт для апскейлинга
try:
    from utils.interferogram_enhancement import (
        ResolutionConfig,
        InterferogramAnalyzer,
        enhance_interferogram_batch,
        create_interferogram_upsampler
    )
except ImportError:
    # Fallback для случаев, когда utils недоступен
    ResolutionConfig = None
    InterferogramAnalyzer = None
    enhance_interferogram_batch = None
    create_interferogram_upsampler = None


def extract_ints_no_regex(base: str) -> list[int]:
    """
    Извлекает целые числа из строки без использования регулярных выражений.
    Любой нецифровой символ рассматривается как разделитель.
    """
    nums: list[int] = []
    cur: int | None = None
    for ch in base:
        if "0" <= ch <= "9":
            d = ord(ch) - 48
            cur = d if cur is None else cur * 10 + d
        else:
            if cur is not None:
                nums.append(cur)
                cur = None
    if cur is not None:
        nums.append(cur)
    return nums


def parse_bits_from_basename(
    base: str,
    numbers_total: int = 10,
    bits_per_number: int = 7,
    value_mode: str = "mod",
    ignore_last_number: bool = False,
) -> np.ndarray:
    ints = extract_ints_no_regex(base)
    if len(ints) < numbers_total:
        raise ValueError(
            f"'{base}': found {len(ints)} numbers, expected >= {numbers_total}. Parsed={ints}"
        )
    picked = ints[:numbers_total]
    if ignore_last_number and picked:
        picked = picked[:-1]

    maxv = (1 << bits_per_number) - 1
    match value_mode:
        case "mod":
            picked = [v & maxv for v in picked]
        case "clip":
            picked = [max(0, min(maxv, v)) for v in picked]
        case "raise":
            for v in picked:
                if not (0 <= v <= maxv):
                    raise ValueError(f"value {v} out of range [0,{maxv}] in '{base}'")
        case _:
            picked = [v & maxv for v in picked]

    out = np.zeros(len(picked) * bits_per_number, dtype=np.float32)
    for i, value in enumerate(picked):
        for k in range(bits_per_number):
            out[i * bits_per_number + k] = (value >> k) & 1
    return out


def parse_korsch_displacements(
    base: str,
    mirror_count: int = 2,
    bits_per_parameter: int = 7,
    linear_range_um: float = 1000.0,
    angular_range_arcsec: float = 60.0,
    value_mode: str = "mod",
) -> tuple[np.ndarray, dict]:
    """
    Парсинг смещений объектива Корша из имени файла.

    Returns:
        binary_vector: Бинарный вектор для обучения
        physical_values: Словарь с физическими значениями для валидации
    """
    ints = extract_ints_no_regex(base)

    # Ожидаемое количество параметров: зеркала × параметры
    expected_params = mirror_count * 5
    if len(ints) < expected_params:
        raise ValueError(
            f"'{base}': found {len(ints)} numbers, expected >= {expected_params} "
            f"for {mirror_count} mirrors with 5 parameters each. Parsed={ints}"
        )

    picked = ints[:expected_params]

    # Валидация диапазонов
    max_value = (1 << bits_per_parameter) - 1

    match value_mode:
        case "mod":
            picked = [v & max_value for v in picked]
        case "clip":
            picked = [max(0, min(max_value, v)) for v in picked]
        case "raise":
            for v in picked:
                if not (0 <= v <= max_value):
                    raise ValueError(f"value {v} out of range [0,{max_value}] in '{base}'")
        case _:
            picked = [v & max_value for v in picked]

    # Создание бинарного вектора
    binary_vector = np.zeros(len(picked) * bits_per_parameter, dtype=np.float32)
    for i, value in enumerate(picked):
        for k in range(bits_per_parameter):
            binary_vector[i * bits_per_parameter + k] = (value >> k) & 1

    # Расчет физических значений для валидации
    levels = 1 << bits_per_parameter
    linear_step_um = linear_range_um / levels
    angular_step_arcsec = angular_range_arcsec / levels

    physical_values = {}
    for mirror in range(mirror_count):
        mirror_data = {}

        for param_idx in range(5):
            param_idx_global = mirror * 5 + param_idx
            discrete_value = picked[param_idx_global]

            # Преобразование в физические единицы
            if param_idx < 2:  # Угловые параметры X, Y
                physical_value = (discrete_value - levels // 2) * angular_step_arcsec
                param_name = f'angular_{["x", "y"][param_idx]}'
                unit = 'arcsec'
            else:  # Линейные параметры X, Y, Z
                physical_value = (discrete_value - levels // 2) * linear_step_um
                param_name = f'linear_{["x", "y", "z"][param_idx - 2]}'
                unit = 'um'

            mirror_data[param_name] = {
                'value': physical_value,
                'unit': unit,
                'discrete_level': discrete_value
            }

        physical_values[f'mirror_{mirror + 1}'] = mirror_data

    return binary_vector, physical_values


DEFAULT_AUG_CFG = {
    "enabled": True,
    "hflip": {"enabled": True, "prob": 0.5},
    "vflip": {"enabled": True, "prob": 0.5},
    "rotate90": {"enabled": True, "prob": 0.5},
    "gaussian_noise": {"enabled": False, "std": 0.01, "prob": 0.2},
    "random_gamma": {"enabled": False, "range": [0.9, 1.1], "prob": 0.3},
}


def _merge_nested(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            dst[key] = _merge_nested(dst[key], value)
        else:
            dst[key] = value
    return dst


def _normalize_aug_cfg(augmentations: dict | None, legacy_cfg: dict | None) -> dict:
    cfg = deepcopy(DEFAULT_AUG_CFG)
    if legacy_cfg:
        legacy_norm = {
            "enabled": True,
            "hflip": {"enabled": bool(legacy_cfg.get("hflip", True)), "prob": 0.5},
            "vflip": {"enabled": bool(legacy_cfg.get("vflip", True)), "prob": 0.5},
            "rotate90": {"enabled": bool(legacy_cfg.get("rotate90", True)), "prob": 0.5},
            "gaussian_noise": {
                "enabled": bool(legacy_cfg.get("gaussian_noise_std", 0) > 0),
                "std": float(legacy_cfg.get("gaussian_noise_std", 0.01)),
                "prob": 0.2,
            },
            "random_gamma": {
                "enabled": "random_gamma" in legacy_cfg,
                "range": list(legacy_cfg.get("random_gamma", [0.9, 1.1])),
                "prob": 0.3,
            },
        }
        cfg = _merge_nested(cfg, legacy_norm)
    if augmentations:
        cfg = _merge_nested(cfg, augmentations)
    return cfg


class InterferogramDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_glob: str = "**/*.png,**/*.jpg,**/*.tif,**/*.bmp",
        image_size: int = 512,
        *,
        filename_label_regex: str | None = None,  # для обратной совместимости
        numbers_total: int = 10,               # 2 зеркала × 5 параметров
        bits_per_number: int = 7,              # Для субмикронной точности
        value_mode: str = "mod",
        ignore_last_number: bool = False,        # Используем все параметры
        # Korsch specific parameters
        korsch_mode: bool = False,              # Использовать Korch парсинг
        mirror_count: int = 2,
        linear_range_um: float = 1000.0,
        angular_range_arcsec: float = 60.0,
        # Reference interferogram parameters
        reference_interferogram: str | None = None,  # Path to ideal reference
        use_difference_mode: bool = False,            # Compute difference (current - reference)
        reference_preprocessing: dict | None = None,  # Reference-specific processing
        # Resolution enhancement parameters
        resolution_enhancement: dict | None = None,
        target_resolution: int | None = None,
        upscaling_method: str = "adaptive",  # adaptive, sr_cnn, bicubic, lanczos
        preserve_fringes: bool = True,
        # Standard parameters
        augment: bool = True,
        aug_cfg: dict | None = None,
        label_cfg: dict | None = None,
        normalization: dict | None = None,
        augmentations: dict | None = None,
        files: Sequence[str] | None = None,
        labels: np.ndarray | None = None,
    ):
        self.root = os.path.abspath(root)
        self.image_size = int(image_size)
        self.img_glob = img_glob

        if label_cfg:
            numbers_total = label_cfg.get("numbers_total", numbers_total)
            bits_per_number = label_cfg.get("bits_per_number", bits_per_number)
            value_mode = label_cfg.get("value_mode", value_mode)
            ignore_last_number = label_cfg.get("ignore_last_number", ignore_last_number)

        self.numbers_total = int(numbers_total)
        self.bits_per_number = int(bits_per_number)
        self.value_mode = value_mode
        self.ignore_last_number = bool(ignore_last_number)

        # Korsch specific parameters
        self.korsch_mode = bool(korsch_mode)
        self.mirror_count = int(mirror_count)
        self.linear_range_um = float(linear_range_um)
        self.angular_range_arcsec = float(angular_range_arcsec)

        # Reference interferogram parameters
        self.reference_interferogram = reference_interferogram
        self.use_difference_mode = bool(use_difference_mode)
        self.reference_preprocessing = deepcopy(reference_preprocessing) if reference_preprocessing else {}
        self.reference_image = None  # Loaded reference image

        # Resolution enhancement parameters
        self.resolution_enhancement = deepcopy(resolution_enhancement) if resolution_enhancement else {}
        self.target_resolution = int(target_resolution) if target_resolution is not None else None
        self.upscaling_method = str(upscaling_method)
        self.preserve_fringes = bool(preserve_fringes)

        # Resolution analysis
        self._resolution_analyzer = None
        self._upscaler = None
        self._resolution_analysis_cache = {}

        if ResolutionConfig is not None:
            # Инициализация анализатора разрешения
            self._resolution_analyzer = InterferogramAnalyzer()

            # Конфигурация улучшения разрешения
            self._resolution_config = ResolutionConfig(
                target_resolution=self.target_resolution or 1024,
                min_resolution=256,
                analysis_threshold=self.resolution_enhancement.get("analysis_threshold", 0.8),
                preserve_fringes=self.preserve_fringes,
                enhancement_method=upscaling_method
            )

            # Предварительное создание апскейлера если нужно
            if upscaling_method != "adaptive":
                self._upscaler = create_interferogram_upsampler(
                    method=upscaling_method,
                    scale_factor=4,
                    target_resolution=self.target_resolution
                )

        self.normalization_cfg = deepcopy(normalization) if normalization else {}
        self.norm_type = str(self.normalization_cfg.get("type", "none")).lower()
        clip_range = self.normalization_cfg.get("clip_range", None)
        self.clip_range = tuple(clip_range) if clip_range is not None else None
        self.global_mean = (
            float(self.normalization_cfg["mean"])
            if "mean" in self.normalization_cfg
            else None
        )
        self.global_std = (
            float(self.normalization_cfg["std"])
            if "std" in self.normalization_cfg
            else None
        )

        aug_cfg = augmentations if augmentations is not None else aug_cfg
        self.aug_cfg = _normalize_aug_cfg(aug_cfg, aug_cfg if augmentations is None else None)
        self.augment_flag = bool(augment) and bool(self.aug_cfg.get("enabled", True))

        # Load reference interferogram if provided
        if self.reference_interferogram is not None:
            # Sanitize path and validate
            sanitized_ref_path = sanitize_path(self.reference_interferogram)
            if os.path.isabs(sanitized_ref_path):
                ref_path = sanitized_ref_path
            else:
                ref_path = safe_join(self.root, sanitized_ref_path)

            # Validate path
            is_valid, error = validate_path(ref_path, must_exist=True, must_be_file=True)
            if not is_valid:
                raise FileNotFoundError(f"Invalid reference interferogram path: {error}")

            # Use memory-efficient reference image manager
            self.reference_manager = default_reference_manager
            self.reference_image_loader = self.reference_manager.load_reference_image(ref_path)

            # For backward compatibility, keep eager loading option
            enable_lazy_loading = True  # Could be made configurable
            if not enable_lazy_loading:
                self.reference_image = self.reference_image_loader()
                print(f"Loaded reference interferogram (eager): {ref_path}, shape: {self.reference_image.shape}")
            else:
                self.reference_image = None
                print(f"Reference interferogram configured for lazy loading: {ref_path}")
        else:
            self.reference_manager = None
            self.reference_image_loader = None
            self.reference_image = None

        if files is not None and labels is not None:
            # Validate and sanitize file paths
            abs_files = []
            for f in files:
                sanitized_f = sanitize_path(str(f))
                if os.path.isabs(sanitized_f):
                    file_path = sanitized_f
                else:
                    file_path = safe_join(self.root, sanitized_f)

                is_valid, error = validate_path(file_path, must_exist=True, must_be_file=True)
                if not is_valid:
                    raise FileNotFoundError(f"Invalid file path: {error}")

                abs_files.append(os.path.abspath(file_path))

            self.files = abs_files
            self.paths = [os.path.relpath(f, self.root) for f in self.files]
            self.labels = np.asarray(labels, dtype=np.float32)
        else:
            # Sanitize patterns
            raw_patterns = str(img_glob).replace(";", ",").split(",")
            patterns = [sanitize_path(p.strip(), allow_wildcards=True) for p in raw_patterns if p.strip()]

            discovered: list[str] = []
            for pat in patterns:
                # Safe pattern joining
                search_path = safe_join(self.root, pat)
                try:
                    matches = glob.glob(search_path, recursive=True)
                    # Validate each discovered file
                    for match in matches:
                        is_valid, error = validate_path(match, must_exist=True, must_be_file=True)
                        if is_valid:
                            discovered.append(match)
                except Exception as e:
                    # Skip dangerous patterns
                    print(f"Warning: Skipping unsafe pattern '{pat}': {e}")
                    continue

            discovered = sorted({os.path.abspath(f) for f in discovered})
            if not discovered:
                raise RuntimeError(
                    f"No images found in '{self.root}' with patterns: {patterns}"
                )
            self.files = discovered
            self.paths = [os.path.relpath(f, self.root) for f in self.files]

            labels_list: list[np.ndarray] = []
            for path in self.files:
                base = os.path.splitext(os.path.basename(path))[0]

                if self.korsch_mode:
                    # Используем Korsch парсинг с валидацией
                    binary_vector, physical_values = parse_korsch_displacements(
                        base,
                        mirror_count=self.mirror_count,
                        bits_per_parameter=self.bits_per_number,
                        linear_range_um=self.linear_range_um,
                        angular_range_arcsec=self.angular_range_arcsec,
                        value_mode=self.value_mode,
                    )
                    labels_list.append(binary_vector)
                else:
                    # Стандартный парсинг для обратной совместимости
                    labels_list.append(
                        parse_bits_from_basename(
                            base,
                            numbers_total=self.numbers_total,
                            bits_per_number=self.bits_per_number,
                            value_mode=self.value_mode,
                            ignore_last_number=self.ignore_last_number,
                        )
                    )
            self.labels = np.stack(labels_list, axis=0).astype(np.float32)

        self.K = int(self.labels.shape[1])

    def cleanup_memory(self) -> None:
        """
        Clean up memory used by the dataset.
        """
        if hasattr(self, 'reference_manager') and self.reference_manager:
            self.reference_manager.clear_cache()

        # Clear reference image if loaded eagerly
        if hasattr(self, 'reference_image') and self.reference_image is not None:
            self.reference_image = None

        # Force garbage collection
        import gc
        gc.collect()

    def __del__(self):
        """Destructor to ensure memory cleanup."""
        try:
            self.cleanup_memory()
        except:
            pass  # Ignore errors during cleanup

    def __len__(self) -> int:
        return len(self.files)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        img = img.astype(np.float32)
        if img.max() > 1.5 or img.min() < -0.5:
            img = img / 255.0
        return img

    def _get_reference_image(self) -> Optional[np.ndarray]:
        """
        Get reference interferogram image with memory-efficient loading.

        Returns:
            Optional[np.ndarray]: Reference image or None if not configured
        """
        if self.reference_image_loader is None:
            return None

        try:
            # Load reference image using lazy loader
            ref_img = self.reference_image_loader()

            # Ensure correct size and format
            if ref_img.shape[0] != self.image_size or ref_img.shape[1] != self.image_size:
                ref_img = cv2.resize(
                    ref_img,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )

            # Ensure correct data type and range
            if ref_img.max() > 1.5 or ref_img.min() < -0.5:
                ref_img = ref_img / 255.0

            return ref_img.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Error loading reference interferogram: {e}")

    def _compute_difference_interferogram(self, current_img: np.ndarray) -> np.ndarray:
        """
        Вычисление разностной интерферограммы (текущая - эталонная).

        Args:
            current_img: Текущая интерферограмма

        Returns:
            Разностная интерферограмма
        """
        # Get reference image using memory-efficient loading
        reference_image = self._get_reference_image()
        if reference_image is None:
            return current_img  # Нет эталона, возвращаем оригинал

        # Проверка размеров
        if current_img.shape != reference_image.shape:
            # Изменение размера эталона под текущую интерферограмму
            ref_resized = cv2.resize(
                reference_image,
                (current_img.shape[1], current_img.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            ref_resized = reference_image

        # Применение препроцессинга к эталону если нужно
        if self.reference_preprocessing:
            ref_processed = self._apply_reference_preprocessing(ref_resized)
        else:
            ref_processed = ref_resized

        # Вычисление разности
        difference = current_img.astype(np.float32) - ref_processed.astype(np.float32)

        # Нормализация разности
        if self.reference_preprocessing.get("normalize_difference", True):
            # Статистическая нормализация разности
            diff_mean = np.mean(difference)
            diff_std = np.std(difference)
            if diff_std > 1e-6:
                difference = (difference - diff_mean) / diff_std

        return difference

    def _apply_reference_preprocessing(self, reference_img: np.ndarray) -> np.ndarray:
        """
        Применение препроцессинга к эталонной интерферограмме.
        """
        processed = reference_img.copy()

        # Применение фильтров если указаны
        if "gaussian_blur" in self.reference_preprocessing:
            kernel_size = self.reference_preprocessing["gaussian_blur"].get("kernel_size", 1)
            if kernel_size > 0 and kernel_size % 2 == 1:
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        if "contrast_enhancement" in self.reference_preprocessing:
            factor = self.reference_preprocessing["contrast_enhancement"].get("factor", 1.0)
            if factor != 1.0:
                mean = processed.mean()
                processed = (processed - mean) * factor + mean

        if "brightness_adjustment" in self.reference_preprocessing:
            offset = self.reference_preprocessing["brightness_adjustment"].get("offset", 0.0)
            if offset != 0.0:
                processed = processed + offset

        return processed

    def _per_image_zscore(self, img: np.ndarray) -> np.ndarray:
        mean = float(img.mean())
        std = float(img.std())
        return (img - mean) / (std + 1e-6)

    def _should_apply(self, key: str, default_prob: float) -> tuple[bool, float]:
        cfg = self.aug_cfg.get(key, {})
        if isinstance(cfg, bool):
            return cfg, default_prob
        enabled = bool(cfg.get("enabled", True))
        prob = float(cfg.get("prob", default_prob))
        return enabled, prob

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not (self.augment_flag and self.aug_cfg.get("enabled", True)):
            return img

        out = img
        rng = np.random.rand
        
        # Горизонтальное отражение
        enabled, prob = self._should_apply("hflip", 0.5)
        if enabled and rng() < prob:
            out = np.flip(out, axis=1)

        # Вертикальное отражение
        enabled, prob = self._should_apply("vflip", 0.5)
        if enabled and rng() < prob:
            out = np.flip(out, axis=0)

        # Поворот на 90 градусов
        enabled, prob = self._should_apply("rotate90", 0.5)
        if enabled and rng() < prob:
            out = np.rot90(out).copy()

        # Случайный поворот
        rot_cfg = self.aug_cfg.get("random_rotation", {})
        if rot_cfg.get("enabled", False) and rng() < float(rot_cfg.get("prob", 0.3)):
            degrees = float(rot_cfg.get("degrees", 15))
            angle = np.random.uniform(-degrees, degrees)
            h, w = out.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Случайное кадрирование (zoom)
        crop_cfg = self.aug_cfg.get("random_crop", {})
        if crop_cfg.get("enabled", False) and rng() < float(crop_cfg.get("prob", 0.3)):
            scale_range = crop_cfg.get("scale", [0.9, 1.0])
            scale = np.random.uniform(scale_range[0], scale_range[1])
            h, w = out.shape
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                # Кадрируем центр изображения
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                cropped = out[start_h:start_h + new_h, start_w:start_w + new_w]
                # Масштабируем обратно к исходному размеру
                out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Цветовые искажения (яркость и контраст)
        jitter_cfg = self.aug_cfg.get("color_jitter", {})
        if jitter_cfg.get("enabled", False) and rng() < float(jitter_cfg.get("prob", 0.2)):
            brightness = float(jitter_cfg.get("brightness", 0.1))
            contrast = float(jitter_cfg.get("contrast", 0.1))
            
            # Яркость
            if brightness > 0:
                brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
                out = out * brightness_factor
            
            # Контраст
            if contrast > 0:
                contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
                mean = out.mean()
                out = (out - mean) * contrast_factor + mean

        # Гауссов шум
        gn_cfg = self.aug_cfg.get("gaussian_noise", {})
        if gn_cfg.get("enabled", False) and rng() < float(gn_cfg.get("prob", 0.2)):
            std = float(gn_cfg.get("std", 0.01))
            out = out + np.random.normal(0.0, std, size=out.shape).astype(np.float32)

        # Гауссов размытие
        blur_cfg = self.aug_cfg.get("gaussian_blur", {})
        if blur_cfg.get("enabled", False) and rng() < float(blur_cfg.get("prob", 0.2)):
            kernel_size = int(blur_cfg.get("kernel_size", 3))
            if kernel_size > 0 and kernel_size % 2 == 1:  # Только нечетные размеры
                out = cv2.GaussianBlur(out, (kernel_size, kernel_size), 0)

        # Гамма коррекция
        gamma_cfg = self.aug_cfg.get("random_gamma", {})
        if gamma_cfg.get("enabled", False) and rng() < float(gamma_cfg.get("prob", 0.3)):
            lo, hi = gamma_cfg.get("range", [0.9, 1.1])
            gamma = np.random.uniform(lo, hi)
            out = np.sign(out) * (np.abs(out) ** gamma)

        return out.copy()

    def set_global_stats(self, mean: float, std: float) -> None:
        self.global_mean = float(mean)
        self.global_std = float(std if std > 1e-8 else 1.0)
        self.normalization_cfg["mean"] = self.global_mean
        self.normalization_cfg["std"] = self.global_std

    def compute_global_stats(
        self,
        indices: Iterable[int] | None = None,
        *,
        use_fp64: bool = True,
    ) -> tuple[float, float]:
        idxs = list(indices) if indices is not None else range(len(self.files))
        dtype = np.float64 if use_fp64 else np.float32

        pixel_count = 0
        sum_ = 0.0
        sum_sq = 0.0
        for idx in idxs:
            path = self.files[idx]
            img = self._read_image(path).astype(dtype, copy=False)
            sum_ += float(img.sum())
            sum_sq += float(np.square(img, dtype=dtype).sum())
            pixel_count += int(img.size)

        if pixel_count == 0:
            raise RuntimeError("Cannot compute statistics on empty index set")

        mean = sum_ / pixel_count
        variance = max(sum_sq / pixel_count - mean * mean, 0.0)
        std = float(np.sqrt(variance + 1e-12))
        self.set_global_stats(mean, std)
        return float(mean), float(std)

    def make_subset(
        self,
        indices: Sequence[int],
        *,
        augment: bool | None = None,
        share_stats: bool = True,
    ) -> "InterferogramDataset":
        idx_array = np.asarray(indices, dtype=np.int64)
        if idx_array.size == 0:
            raise ValueError("Subset indices must be non-empty")

        subset_files = [self.files[i] for i in idx_array]
        subset_labels = self.labels[idx_array].copy()

        normalization = deepcopy(self.normalization_cfg)
        if share_stats and self.global_mean is not None:
            normalization["mean"] = self.global_mean
            normalization["std"] = self.global_std

        subset = InterferogramDataset(
            root=self.root,
            img_glob=self.img_glob,
            image_size=self.image_size,
            numbers_total=self.numbers_total,
            bits_per_number=self.bits_per_number,
            value_mode=self.value_mode,
            ignore_last_number=self.ignore_last_number,
            augment=self.augment_flag if augment is None else bool(augment),
            augmentations=deepcopy(self.aug_cfg),
            normalization=normalization,
            files=subset_files,
            labels=subset_labels,
        )
        return subset

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        # Use safe_join for path construction
        full_path = safe_join(self.root, rel_path)

        # Validate path before reading
        is_valid, error = validate_path(full_path, must_exist=True, must_be_file=True)
        if not is_valid:
            raise FileNotFoundError(f"Invalid image path at index {idx}: {error}")

        img = self._read_image(full_path)

        if self.norm_type == "per_image_zscore":
            img = self._per_image_zscore(img)

        img = self._augment(img)

        if self.norm_type == "global_zscore" and self.global_mean is not None:
            img = (img - self.global_mean) / (self.global_std + 1e-6)

        if self.clip_range is not None:
            lo, hi = self.clip_range
            img = np.clip(img, lo, hi)

        img = np.expand_dims(np.ascontiguousarray(img, dtype=np.float32), axis=0)
        x = torch.from_numpy(img)
        y = torch.from_numpy(self.labels[idx]).float()
        side = None
        return x, y, side, rel_path
