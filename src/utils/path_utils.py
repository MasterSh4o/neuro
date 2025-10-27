"""
Platform-aware path utilities for NeuroInterf project.
Provides cross-platform compatible path management.
"""

import os
from pathlib import Path
from typing import Dict, Optional

def get_platform_base_path() -> str:
    """
    Get the base path for the current platform.

    Returns:
        str: Platform-appropriate base path
    """
    if os.name == 'nt':  # Windows
        return str(Path.home() / "workspace" / "NeuroInterf")
    else:  # Unix/Linux/macOS
        return "/home/jupyter/work"

def get_default_paths() -> Dict[str, str]:
    """
    Get default paths for the current platform.

    Returns:
        Dict[str, str]: Dictionary of default paths
    """
    base_path = get_platform_base_path()

    if os.name == 'nt':  # Windows
        return {
            "workspace_root": base_path,
            "datasets_root": str(Path(base_path) / "datasets"),
            "interferogram_dataset": str(Path(base_path) / "datasets" / "InterfDataset"),
            "reference_interferogram": str(Path(base_path) / "datasets" / "reference_ideal_interferogram.png"),
            "runs_dir": str(Path(base_path) / "runs"),
            "cache_dir": str(Path.home() / ".cache" / "neuro"),
            "torch_home": str(Path.home() / ".cache" / "torch"),
            "huggingface_cache": str(Path.home() / ".cache" / "huggingface"),
            "temp_dir": str(Path(base_path) / "temp"),
        }
    else:  # Unix/Linux/macOS
        return {
            "workspace_root": "/home/jupyter/work",
            "datasets_root": "/home/jupyter/work/datasets",
            "interferogram_dataset": "/home/jupyter/work/datasets/InterfDataset",
            "reference_interferogram": "/home/jupyter/work/datasets/reference_ideal_interferogram.png",
            "runs_dir": "/home/jupyter/work/runs",
            "cache_dir": "/home/jupyter/.cache/neuro",
            "torch_home": "/home/jupyter/.cache/torch",
            "huggingface_cache": "/home/jupyter/.cache/huggingface",
            "temp_dir": "/tmp/neuro_interf",
        }

def resolve_path(path: str, create_if_missing: bool = False) -> str:
    """
    Resolve a path, expanding user directory and environment variables.

    Args:
        path (str): Path to resolve
        create_if_missing (bool): Create directory if it doesn't exist

    Returns:
        str: Resolved absolute path
    """
    # Expand user directory (~) and environment variables
    resolved = Path(path).expanduser()

    # If path is relative, make it relative to project root
    if not resolved.is_absolute():
        project_root = Path(__file__).parent.parent
        resolved = project_root / resolved

    resolved = resolved.resolve()

    if create_if_missing and not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)

    return str(resolved)

def validate_path(path: str, must_exist: bool = True, must_be_dir: bool = False,
                  must_be_file: bool = False) -> tuple[bool, str]:
    """
    Validate a path according to specified criteria.

    Args:
        path (str): Path to validate
        must_exist (bool): Path must exist
        must_be_dir (bool): Path must be a directory
        must_be_file (bool): Path must be a file

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        resolved_path = Path(resolve_path(path))

        if must_exist and not resolved_path.exists():
            return False, f"Path does not exist: {resolved_path}"

        if resolved_path.exists():
            if must_be_dir and not resolved_path.is_dir():
                return False, f"Path is not a directory: {resolved_path}"

            if must_be_file and not resolved_path.is_file():
                return False, f"Path is not a file: {resolved_path}"

        # Check for path traversal attempts
        if '..' in str(resolved_path):
            return False, f"Path traversal detected: {path}"

        return True, ""

    except Exception as e:
        return False, f"Error validating path: {e}"

def safe_join(*paths: str) -> str:
    """
    Safely join paths without allowing path traversal.

    Args:
        *paths: Path components to join

    Returns:
        str: Safely joined path
    """
    try:
        result = Path(*paths)
        # Resolve to canonical form to detect traversal
        resolved = result.resolve()

        # Check if the resolved path is within expected bounds
        if '..' in str(result):
            raise ValueError("Path traversal detected")

        return str(resolved)
    except Exception as e:
        raise ValueError(f"Unsafe path join: {e}")

def get_config_path(config_name: str, config_dir: Optional[str] = None) -> str:
    """
    Get path to configuration file.

    Args:
        config_name (str): Name of config file (with or without .yaml extension)
        config_dir (Optional[str]): Custom config directory

    Returns:
        str: Path to config file
    """
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'

    if config_dir is None:
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"
    else:
        config_dir = Path(config_dir)

    return str(config_dir / config_name)

def ensure_directory(path: str, permissions: Optional[int] = None) -> str:
    """
    Ensure directory exists with proper permissions.

    Args:
        path (str): Directory path
        permissions (Optional[int]): Directory permissions (Unix only)

    Returns:
        str: Created/resolved directory path
    """
    resolved_path = Path(resolve_path(path))
    resolved_path.mkdir(parents=True, exist_ok=True)

    if permissions is not None and os.name != 'nt':
        resolved_path.chmod(permissions)

    return str(resolved_path)

def sanitize_path(path: str) -> str:
    """
    Sanitize path string for security.

    Args:
        path (str): Path to sanitize

    Returns:
        str: Sanitized path
    """
    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    sanitized = path

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')

    # Replace multiple path separators with single one
    while '//' in sanitized or '\\\\' in sanitized:
        sanitized = sanitized.replace('//', '/').replace('\\\\', '\\')

    return str(Path(sanitized).as_posix())