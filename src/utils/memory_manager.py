"""
Memory management utilities for NeuroInterf project.
Provides efficient memory usage for large datasets and reference images.
"""

import os
import gc
import psutil
import threading
from typing import Optional, Dict, Any, Callable, Union
from pathlib import Path
import numpy as np
import cv2


class MemoryMonitor:
    """Monitor memory usage and provide memory statistics."""

    def __init__(self):
        """Initialize memory monitor."""
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dict[str, float]: Memory information in MB
        """
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

    def is_memory_critical(self, threshold_mb: float = 1000) -> bool:
        """
        Check if memory usage is critical.

        Args:
            threshold_mb (float): Memory threshold in MB

        Returns:
            bool: True if memory usage is critical
        """
        memory_info = self.get_memory_info()
        return memory_info['available_mb'] < threshold_mb


class ReferenceImageManager:
    """Efficiently manage reference interferogram images with lazy loading and caching."""

    def __init__(self, max_cache_size_mb: float = 500, enable_lazy_loading: bool = True):
        """
        Initialize reference image manager.

        Args:
            max_cache_size_mb (float): Maximum cache size in MB
            enable_lazy_loading (bool): Enable lazy loading of images
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_lazy_loading = enable_lazy_loading
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_lock = threading.RLock()
        self.memory_monitor = MemoryMonitor()

    def load_reference_image(self, image_path: str, preprocessor: Optional[Callable] = None) -> Callable:
        """
        Get a function to load the reference image (lazy loading).

        Args:
            image_path (str): Path to reference image
            preprocessor (Optional[Callable]): Optional preprocessing function

        Returns:
            Callable: Function that returns the processed image
        """
        if not self.enable_lazy_loading:
            # Eager loading for backward compatibility
            image = self._load_and_process_image(image_path, preprocessor)
            return lambda: image

        # Lazy loading
        def get_reference_image():
            return self._get_cached_image(image_path, preprocessor)

        return get_reference_image

    def _get_cached_image(self, image_path: str, preprocessor: Optional[Callable] = None) -> np.ndarray:
        """
        Get image from cache or load it.

        Args:
            image_path (str): Path to image
            preprocessor (Optional[Callable]): Preprocessing function

        Returns:
            np.ndarray: Processed image
        """
        import time

        with self.cache_lock:
            # Check if image is in cache
            if image_path in self.cache:
                self.cache_access_times[image_path] = time.time()
                return self.cache[image_path].copy()

            # Load and process image
            image = self._load_and_process_image(image_path, preprocessor)

            # Check memory usage before caching
            self._cleanup_if_needed(image.nbytes / 1024 / 1024)

            # Add to cache
            self.cache[image_path] = image.copy()
            self.cache_access_times[image_path] = time.time()

            return image

    def _load_and_process_image(self, image_path: str, preprocessor: Optional[Callable] = None) -> np.ndarray:
        """
        Load and process image from disk.

        Args:
            image_path (str): Path to image
            preprocessor (Optional[Callable]): Preprocessing function

        Returns:
            np.ndarray: Processed image
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Apply preprocessing if provided
            if preprocessor is not None:
                image = preprocessor(image)

            return image

        except Exception as e:
            raise RuntimeError(f"Error loading reference image '{image_path}': {e}")

    def _cleanup_if_needed(self, required_mb: float) -> None:
        """
        Clean up cache if memory is needed.

        Args:
            required_mb (float): Required memory in MB
        """
        current_cache_size = sum(img.nbytes for img in self.cache.values()) / 1024 / 1024

        if current_cache_size + required_mb > self.max_cache_size_mb:
            # Remove least recently used images
            sorted_paths = sorted(
                self.cache_access_times.items(),
                key=lambda x: x[1]
            )

            for path, _ in sorted_paths:
                if current_cache_size <= self.max_cache_size_mb - required_mb:
                    break

                del self.cache[path]
                del self.cache_access_times[path]
                current_cache_size -= self.cache[path].nbytes / 1024 / 1024

            # Force garbage collection
            gc.collect()

    def clear_cache(self) -> None:
        """Clear all cached images."""
        with self.cache_lock:
            self.cache.clear()
            self.cache_access_times.clear()
            gc.collect()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self.cache_lock:
            cache_size_mb = sum(img.nbytes for img in self.cache.values()) / 1024 / 1024
            memory_info = self.memory_monitor.get_memory_info()

            return {
                'cached_images': len(self.cache),
                'cache_size_mb': cache_size_mb,
                'max_cache_size_mb': self.max_cache_size_mb,
                'memory_usage_mb': memory_info['rss_mb'],
                'memory_percent': memory_info['percent'],
                'available_mb': memory_info['available_mb']
            }


class DatasetMemoryOptimizer:
    """Optimize memory usage for large datasets."""

    def __init__(self, target_memory_usage_percent: float = 80):
        """
        Initialize memory optimizer.

        Args:
            target_memory_usage_percent (float): Target memory usage percentage
        """
        self.target_memory_usage_percent = target_memory_usage_percent
        self.memory_monitor = MemoryMonitor()

    def optimize_batch_size(self, base_batch_size: int, image_size: tuple,
                           model_memory_mb: float = 1000) -> int:
        """
        Optimize batch size based on available memory.

        Args:
            base_batch_size (int): Base batch size
            image_size (tuple): Image size (height, width)
            model_memory_mb (float): Estimated model memory usage in MB

        Returns:
            int: Optimized batch size
        """
        memory_info = self.memory_monitor.get_memory_info()
        available_mb = memory_info['available_mb']

        # Reserve memory for model and other processes
        usable_mb = available_mb * 0.8 - model_memory_mb

        # Estimate memory per sample (float32 image + overhead)
        image_memory_mb = (image_size[0] * image_size[1] * 4) / 1024 / 1024
        sample_memory_mb = image_memory_mb * 1.5  # Include processing overhead

        max_batch_size = int(usable_mb / sample_memory_mb)

        # Return the minimum of base batch size and calculated maximum
        optimized_batch_size = min(base_batch_size, max(1, max_batch_size))

        if optimized_batch_size < base_batch_size:
            print(f"Memory optimization: Reduced batch size from {base_batch_size} to {optimized_batch_size}")

        return optimized_batch_size

    def should_enable_lazy_loading(self, dataset_size: int, image_size: tuple) -> bool:
        """
        Determine if lazy loading should be enabled based on dataset size and available memory.

        Args:
            dataset_size (int): Number of samples in dataset
            image_size (tuple): Image size (height, width)

        Returns:
            bool: True if lazy loading should be enabled
        """
        memory_info = self.memory_monitor.get_memory_info()
        available_mb = memory_info['available_mb']

        # Estimate total dataset memory requirement
        image_memory_mb = (image_size[0] * image_size[1] * 4) / 1024 / 1024
        total_dataset_mb = dataset_size * image_memory_mb * 1.5  # Include overhead

        # Enable lazy loading if dataset would use more than 50% of available memory
        return total_dataset_mb > available_mb * 0.5

    def suggest_optimization(self, dataset_size: int, image_size: tuple,
                           current_batch_size: int) -> Dict[str, Any]:
        """
        Suggest memory optimizations.

        Args:
            dataset_size (int): Number of samples in dataset
            image_size (tuple): Image size (height, width)
            current_batch_size (int): Current batch size

        Returns:
            Dict[str, Any]: Optimization suggestions
        """
        memory_info = self.memory_monitor.get_memory_info()

        # Calculate memory requirements
        image_memory_mb = (image_size[0] * image_size[1] * 4) / 1024 / 1024
        dataset_mb = dataset_size * image_memory_mb
        batch_memory_mb = current_batch_size * image_memory_mb

        suggestions = {
            'current_memory_mb': memory_info['rss_mb'],
            'available_mb': memory_info['available_mb'],
            'dataset_memory_mb': dataset_mb,
            'batch_memory_mb': batch_memory_mb,
            'optimizations': []
        }

        # Check if optimizations are needed
        if dataset_mb > memory_info['available_mb'] * 0.5:
            suggestions['optimizations'].append({
                'type': 'lazy_loading',
                'description': 'Enable lazy loading to reduce memory usage',
                'priority': 'high'
            })

        optimized_batch_size = self.optimize_batch_size(current_batch_size, image_size)
        if optimized_batch_size < current_batch_size:
            suggestions['optimizations'].append({
                'type': 'reduce_batch_size',
                'description': f'Reduce batch size from {current_batch_size} to {optimized_batch_size}',
                'priority': 'medium'
            })

        if memory_info['percent'] > 80:
            suggestions['optimizations'].append({
                'type': 'memory_cleanup',
                'description': 'Enable periodic memory cleanup and garbage collection',
                'priority': 'medium'
            })

        return suggestions


# Global instances
default_memory_monitor = MemoryMonitor()
default_reference_manager = ReferenceImageManager()
default_dataset_optimizer = DatasetMemoryOptimizer()

def get_memory_info() -> Dict[str, float]:
    """Get current memory information."""
    return default_memory_monitor.get_memory_info()

def is_memory_critical(threshold_mb: float = 1000) -> bool:
    """Check if memory usage is critical."""
    return default_memory_monitor.is_memory_critical(threshold_mb)

def optimize_memory_usage() -> None:
    """Optimize memory usage by running garbage collection."""
    gc.collect()