"""
Secure configuration loader for NeuroInterf project.
Provides safe loading and validation of YAML configurations.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .path_utils import validate_path, resolve_path, sanitize_path


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigLoader:
    """Secure configuration loader with validation."""

    def __init__(self, allow_eval: bool = False, max_file_size: int = 10 * 1024 * 1024):
        """
        Initialize config loader.

        Args:
            allow_eval (bool): Allow dynamic evaluation of config values (dangerous)
            max_file_size (int): Maximum config file size in bytes
        """
        self.allow_eval = allow_eval
        self.max_file_size = max_file_size

    def load_config(self, config_path: Union[str, Path],
                   schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Safely load configuration from file.

        Args:
            config_path (Union[str, Path]): Path to config file
            schema (Optional[Dict[str, Any]]): Validation schema

        Returns:
            Dict[str, Any]: Loaded configuration

        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = str(config_path)

        # Validate path
        is_valid, error = validate_path(config_path, must_exist=True, must_be_file=True)
        if not is_valid:
            raise ConfigValidationError(f"Invalid config path: {error}")

        # Check file size
        file_size = os.path.getsize(config_path)
        if file_size > self.max_file_size:
            raise ConfigValidationError(f"Config file too large: {file_size} bytes")

        # Load YAML safely
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Security checks
            self._validate_yaml_content(content)

            config = yaml.safe_load(content)
            if config is None:
                config = {}

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"YAML parsing error: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error reading config: {e}")

        # Process configuration
        processed_config = self._process_config(config)

        # Validate against schema if provided
        if schema:
            self._validate_schema(processed_config, schema)

        return processed_config

    def _validate_yaml_content(self, content: str) -> None:
        """
        Validate YAML content for security issues.

        Args:
            content (str): YAML content to validate

        Raises:
            ConfigValidationError: If security issues found
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            r'!!python.*',  # Python-specific tags
            r'!!.*\/.*',    # Custom tags
            r'<\?.*\?>',    # XML processing instructions
            r'&\w+.*\*\w+', # YAML anchors and aliases (could cause DoS)
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                raise ConfigValidationError(f"Dangerous YAML pattern detected: {pattern}")

        # Check for potential command injection
        command_patterns = [
            r'\$\s*\(',     # Command substitution
            r'`[^`]*`',     # Backtick commands
            r'\|\s*\w+',    # Pipes
            r'&&\s*\w+',    # Command chaining
            r'\|\|\s*\w+',  # OR operations
        ]

        for pattern in command_patterns:
            if re.search(pattern, content):
                raise ConfigValidationError(f"Potential command injection detected: {pattern}")

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration values safely.

        Args:
            config (Dict[str, Any]): Raw configuration

        Returns:
            Dict[str, Any]: Processed configuration
        """
        processed = {}

        for key, value in config.items():
            processed[key] = self._process_value(key, value)

        return processed

    def _process_value(self, key: str, value: Any) -> Any:
        """
        Process individual configuration value.

        Args:
            key (str): Configuration key
            value (Any): Configuration value

        Returns:
            Any: Processed value
        """
        if isinstance(value, str):
            return self._process_string_value(key, value)
        elif isinstance(value, dict):
            return {k: self._process_value(f"{key}.{k}", v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._process_value(f"{key}[]", item) for item in value]
        else:
            return value

    def _process_string_value(self, key: str, value: str) -> str:
        """
        Process string configuration values.

        Args:
            key (str): Configuration key
            value (str): String value

        Returns:
            str: Processed value
        """
        # Handle path-like values
        if any(path_keyword in key.lower() for path_keyword in
               ['path', 'dir', 'root', 'folder', 'file']):
            return resolve_path(value)

        # Handle environment variables (safe subset)
        if value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', env_var):
                return os.getenv(env_var, value)
            else:
                raise ConfigValidationError(f"Invalid environment variable name: {env_var}")

        # Allow dynamic evaluation if explicitly enabled (DANGEROUS)
        if self.allow_eval and value.startswith('eval:'):
            try:
                # Very restricted eval environment
                allowed_names = {}
                return eval(value[5:], {"__builtins__": {}}, allowed_names)
            except Exception as e:
                raise ConfigValidationError(f"Error evaluating expression '{value}': {e}")

        return value

    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.

        Args:
            config (Dict[str, Any]): Configuration to validate
            schema (Dict[str, Any]): Validation schema

        Raises:
            ConfigValidationError: If validation fails
        """
        # Basic schema validation
        required_keys = schema.get('required', [])
        for key in required_keys:
            if key not in config:
                raise ConfigValidationError(f"Required configuration key missing: {key}")

        # Type validation
        type_specs = schema.get('types', {})
        for key, expected_type in type_specs.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    raise ConfigValidationError(
                        f"Invalid type for '{key}': expected {expected_type.__name__}, "
                        f"got {type(config[key]).__name__}"
                    )

        # Value validation
        validators = schema.get('validators', {})
        for key, validator in validators.items():
            if key in config:
                if not validator(config[key]):
                    raise ConfigValidationError(f"Invalid value for '{key}': {config[key]}")

    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        Save configuration to file safely.

        Args:
            config (Dict[str, Any]): Configuration to save
            config_path (Union[str, Path]): Output path

        Raises:
            ConfigValidationError: If validation fails
        """
        config_path = str(config_path)

        # Validate output path
        is_valid, error = validate_path(config_path, must_exist=False)
        if not is_valid:
            raise ConfigValidationError(f"Invalid output path: {error}")

        # Sanitize configuration
        sanitized_config = self._sanitize_for_save(config)

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(sanitized_config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

        except Exception as e:
            raise ConfigValidationError(f"Error saving config: {e}")

    def _sanitize_for_save(self, config: Any) -> Any:
        """
        Sanitize configuration for saving.

        Args:
            config (Any): Configuration to sanitize

        Returns:
            Any: Sanitized configuration
        """
        if isinstance(config, dict):
            return {k: self._sanitize_for_save(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._sanitize_for_save(item) for item in config]
        elif isinstance(config, (int, float, bool, str)):
            return config
        else:
            # Convert other types to string representation
            return str(config)


# Global config loader instance
default_loader = ConfigLoader()

def load_config(config_path: Union[str, Path],
               schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration using default loader.

    Args:
        config_path (Union[str, Path]): Path to config file
        schema (Optional[Dict[str, Any]]): Validation schema

    Returns:
        Dict[str, Any]: Loaded configuration
    """
    return default_loader.load_config(config_path, schema)

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration using default loader.

    Args:
        config (Dict[str, Any]): Configuration to save
        config_path (Union[str, Path]): Output path
    """
    default_loader.save_config(config, config_path)