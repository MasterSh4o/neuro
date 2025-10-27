"""
Base dataset class for NeuroInterf project.
Provides common functionality and reduces code duplication.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from copy import deepcopy

from utils.path_utils import validate_path, safe_join, sanitize_path
from utils.memory_manager import ReferenceImageManager, default_reference_manager


class BaseInterferogramDataset(Dataset, ABC):
    """
    Base class for interferogram datasets with common functionality.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 512,
        reference_interferogram: Optional[str] = None,
        reference_preprocessing: Optional[Dict[str, Any]] = None,
        enable_lazy_loading: bool = True,
        cache_size_mb: float = 500,
    ):
        """
        Initialize base interferogram dataset.

        Args:
            root (str): Root directory of dataset
            image_size (int): Target image size
            reference_interferogram (Optional[str]): Path to reference interferogram
            reference_preprocessing (Optional[Dict[str, Any]]): Reference preprocessing config
            enable_lazy_loading (bool): Enable lazy loading for reference images
            cache_size_mb (float): Cache size for reference images in MB
        """
        self.root = root
        self.image_size = image_size
        self.reference_interferogram = reference_interferogram
        self.reference_preprocessing = reference_preprocessing or {}
        self.enable_lazy_loading = enable_lazy_loading

        # Initialize memory management
        self.reference_manager = ReferenceImageManager(
            max_cache_size_mb=cache_size_mb,
            enable_lazy_loading=enable_lazy_loading
        )

        # Initialize reference image handling
        self._setup_reference_image()

        # Abstract properties to be set by subclasses
        self.files: List[str] = []
        self.paths: List[str] = []
        self.labels: np.ndarray = np.array([])

        # Load dataset
        self._load_dataset()

    def _setup_reference_image(self) -> None:
        """Setup reference interferogram loading."""
        if self.reference_interferogram is None:
            self.reference_image_loader = None
            return

        # Sanitize and validate path
        sanitized_ref_path = sanitize_path(self.reference_interferogram)
        if os.path.isabs(sanitized_ref_path):
            ref_path = sanitized_ref_path
        else:
            ref_path = safe_join(self.root, sanitized_ref_path)

        # Validate path
        is_valid, error = validate_path(ref_path, must_exist=True, must_be_file=True)
        if not is_valid:
            raise FileNotFoundError(f"Invalid reference interferogram path: {error}")

        # Setup memory-efficient loading
        self.reference_image_loader = self.reference_manager.load_reference_image(
            ref_path,
            preprocessor=self._create_reference_preprocessor()
        )

        print(f"Reference interferogram configured: {ref_path}")

    def _create_reference_preprocessor(self):
        """Create preprocessing function for reference image."""
        def preprocess_ref_image(image: np.ndarray) -> np.ndarray:
            # Apply reference preprocessing if configured
            if self.reference_preprocessing.get("gaussian_blur"):
                kernel_size = self.reference_preprocessing["gaussian_blur"].get("kernel_size", 1)
                if kernel_size > 1:
                    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

            return image

        return preprocess_ref_image

    @abstractmethod
    def _load_dataset(self) -> None:
        """
        Load dataset files and labels.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _parse_label_from_filename(self, filename: str) -> np.ndarray:
        """
        Parse label from filename.
        Must be implemented by subclasses.

        Args:
            filename (str): Filename to parse

        Returns:
            np.ndarray: Parsed label
        """
        pass

    def _read_image(self, path: str) -> np.ndarray:
        """
        Read and preprocess image from path.

        Args:
            path (str): Path to image file

        Returns:
            np.ndarray: Preprocessed image
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        # Resize if needed
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )

        # Convert to float and normalize
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
        Compute difference interferogram (current - reference).

        Args:
            current_img (np.ndarray): Current interferogram

        Returns:
            np.ndarray: Difference interferogram
        """
        # Get reference image using memory-efficient loading
        reference_image = self._get_reference_image()
        if reference_image is None:
            return current_img  # No reference, return original

        # Check dimensions
        if current_img.shape != reference_image.shape:
            # Resize reference to match current image
            ref_resized = cv2.resize(
                reference_image,
                (current_img.shape[1], current_img.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            ref_resized = reference_image

        # Apply reference preprocessing if needed
        if self.reference_preprocessing:
            ref_processed = self._apply_reference_preprocessing(ref_resized)
        else:
            ref_processed = ref_resized

        # Compute difference
        difference = current_img.astype(np.float32) - ref_processed.astype(np.float32)

        # Normalize difference if configured
        if self.reference_preprocessing.get("normalize_difference", True):
            diff_mean = np.mean(difference)
            diff_std = np.std(difference)
            if diff_std > 1e-6:
                difference = (difference - diff_mean) / diff_std

        return difference

    def _apply_reference_preprocessing(self, ref_img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to reference image.

        Args:
            ref_img (np.ndarray): Reference image

        Returns:
            np.ndarray: Preprocessed reference image
        """
        processed = ref_img.copy()

        # Apply Gaussian blur if configured
        if self.reference_preprocessing.get("gaussian_blur"):
            kernel_size = self.reference_preprocessing["gaussian_blur"].get("kernel_size", 1)
            if kernel_size > 1 and kernel_size % 2 == 1:  # Must be odd
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        return processed

    def _discover_files(self, patterns: List[str]) -> List[str]:
        """
        Discover files using glob patterns with validation.

        Args:
            patterns (List[str]): Glob patterns to search

        Returns:
            List[str]: List of discovered file paths
        """
        discovered: List[str] = []

        for pattern in patterns:
            # Safe pattern joining
            search_path = safe_join(self.root, pattern)
            try:
                matches = cv2.glob(search_path)  # Use cv2.glob for more compatibility
                # Validate each discovered file
                for match in matches:
                    is_valid, error = validate_path(match, must_exist=True, must_be_file=True)
                    if is_valid:
                        discovered.append(match)
            except Exception as e:
                # Skip dangerous patterns
                print(f"Warning: Skipping unsafe pattern '{pattern}': {e}")
                continue

        return sorted({os.path.abspath(f) for f in discovered})

    def cleanup_memory(self) -> None:
        """Clean up memory used by the dataset."""
        if hasattr(self, 'reference_manager') and self.reference_manager:
            self.reference_manager.clear_cache()

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
        """Return dataset size."""
        return len(self.files)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dict[str, Any]: Memory statistics
        """
        stats = {
            'dataset_size': len(self.files),
            'image_size': f"{self.image_size}x{self.image_size}",
        }

        if hasattr(self, 'reference_manager') and self.reference_manager:
            stats.update(self.reference_manager.get_cache_stats())

        return stats

    def subset(self, indices: List[int], **kwargs) -> 'BaseInterferogramDataset':
        """
        Create a subset of the dataset.

        Args:
            indices (List[int]): Indices to include in subset
            **kwargs: Additional arguments for subset creation

        Returns:
            BaseInterferogramDataset: Dataset subset
        """
        if not indices:
            raise ValueError("Subset indices must be non-empty")

        # Create new dataset instance with same configuration
        subset_class = self.__class__
        subset = subset_class.__new__(subset_class)

        # Copy basic attributes
        subset.root = self.root
        subset.image_size = self.image_size
        subset.reference_interferogram = self.reference_interferogram
        subset.reference_preprocessing = deepcopy(self.reference_preprocessing)
        subset.enable_lazy_loading = self.enable_lazy_loading

        # Setup reference image for subset
        subset._setup_reference_image()

        # Create subset of files and labels
        subset.files = [self.files[i] for i in indices]
        subset.paths = [self.paths[i] for i in indices]
        subset.labels = self.labels[indices].copy() if len(self.labels) > 0 else np.array([])

        return subset