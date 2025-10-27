"""
Configuration validation utilities for NeuroInterf project.
Ensures consistency between different configuration sections.
"""

from typing import Dict, Any, List, Optional, Union


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates configuration consistency and correctness."""

    def __init__(self):
        """Initialize the validator."""
        self.validation_rules = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default validation rules."""
        # Precision consistency rule
        self.add_rule(self._validate_precision_consistency)

        # Path validation rule
        self.add_rule(self._validate_paths)

        # Model-data compatibility rule
        self.add_rule(self._validate_model_data_compatibility)

        # Dimensions consistency rule
        self.add_rule(self._validate_dimensions_consistency)

    def add_rule(self, rule_func):
        """
        Add a validation rule.

        Args:
            rule_func: Function that takes config and returns (is_valid, error_message)
        """
        self.validation_rules.append(rule_func)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of errors.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            List[str]: List of validation errors
        """
        errors = []

        for rule in self.validation_rules:
            try:
                is_valid, error_message = rule(config)
                if not is_valid:
                    errors.append(error_message)
            except Exception as e:
                errors.append(f"Validation rule failed: {e}")

        return errors

    def validate_or_raise(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration and raise exception if errors found.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        errors = self.validate_config(config)
        if errors:
            raise ConfigValidationError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    def _validate_precision_consistency(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate that precision parameters are consistent across sections.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        label_cfg = data_cfg.get('label_parsing', {})

        # Extract precision values
        data_bits = label_cfg.get('bits_per_number')
        model_bits = model_cfg.get('bits_per_parameter')

        if data_bits is not None and model_bits is not None:
            if data_bits != model_bits:
                return False, f"Precision mismatch: data.label_parsing.bits_per_number ({data_bits}) != model.bits_per_parameter ({model_bits})"

        # Validate out_dim matches expected calculation
        if model_bits is not None:
            expected_out_dim = 10 * model_bits  # 2 mirrors × 5 parameters × bits
            actual_out_dim = model_cfg.get('out_dim')

            if actual_out_dim is not None and actual_out_dim != expected_out_dim:
                return False, f"Output dimension mismatch: model.out_dim ({actual_out_dim}) != expected (10 × bits_per_parameter = {expected_out_dim})"

        return True, ""

    def _validate_paths(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate that paths are properly formatted and don't contain platform-specific issues.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        def check_path_recursively(obj, path_context=""):
            """Recursively check paths in nested dictionaries."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(path_keyword in key.lower() for path_keyword in ['path', 'dir', 'root', 'folder']):
                        if isinstance(value, str):
                            # Check for problematic patterns
                            if '\\\\' in value or '//' in value:
                                return False, f"Invalid path separators in {path_context}.{key}: {value}"

                            # Check for absolute Unix paths on Windows
                            import os
                            if os.name == 'nt' and value.startswith('/') and not value.startswith('//'):
                                return False, f"Unix absolute path on Windows in {path_context}.{key}: {value}"

                    # Recursively check nested structures
                    if isinstance(value, (dict, list)):
                        result = check_path_recursively(value, f"{path_context}.{key}" if path_context else key)
                        if not result[0]:
                            return result

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        result = check_path_recursively(item, f"{path_context}[{i}]")
                        if not result[0]:
                            return result

            return True, ""

        return check_path_recursively(config)

    def _validate_model_data_compatibility(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate that model and data configurations are compatible.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})

        # Check image size compatibility
        image_size = data_cfg.get('image_size')
        if isinstance(image_size, int):
            if image_size < 128:
                return False, f"Image size too small: {image_size} (minimum 128)"
            if image_size > 2048:
                return False, f"Image size too large: {image_size} (maximum 2048)"

        # Check augmentation settings for precision
        aug_cfg = data_cfg.get('augmentations', {})
        bits = model_cfg.get('bits_per_parameter', 5)

        if bits >= 7:  # High precision mode
            # Warn about aggressive augmentations
            if aug_cfg.get('rotate90', {}).get('enabled', False):
                return False, "rotate90 augmentation should be disabled for high precision (>=7 bits)"

            if aug_cfg.get('random_rotation', {}).get('enabled', False):
                return False, "random_rotation augmentation should be disabled for high precision (>=7 bits)"

        return True, ""

    def _validate_dimensions_consistency(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate dimensional consistency across configuration.

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        label_cfg = data_cfg.get('label_parsing', {})

        # Check numbers_total consistency
        numbers_total = label_cfg.get('numbers_total')
        if numbers_total is not None:
            expected_numbers = 10  # 2 mirrors × 5 parameters
            if numbers_total != expected_numbers:
                return False, f"Invalid numbers_total: {numbers_total} (expected {expected_numbers} for 2 mirrors × 5 parameters)"

        # Check regression_dim consistency
        regression_dim = model_cfg.get('regression_dim')
        if regression_dim is not None and regression_dim != 10:
            return False, f"Invalid regression_dim: {regression_dim} (expected 10 for 2 mirrors × 5 parameters)"

        return True, ""

    def get_precision_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract precision information from configuration.

        Args:
            config (Dict[str, Any]): Configuration to analyze

        Returns:
            Dict[str, Any]: Precision information
        """
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        label_cfg = data_cfg.get('label_parsing', {})

        bits = label_cfg.get('bits_per_number', model_cfg.get('bits_per_parameter', 5))
        out_dim = model_cfg.get('out_dim', 10 * bits)
        regression_dim = model_cfg.get('regression_dim', 10)

        levels = 1 << bits
        linear_range_um = 1000.0  # Default from parse_korsch_displacements
        angular_range_arcsec = 60.0

        linear_precision_um = linear_range_um / levels
        angular_precision_arcsec = angular_range_arcsec / levels

        return {
            'bits_per_parameter': bits,
            'out_dim': out_dim,
            'regression_dim': regression_dim,
            'discrete_levels': levels,
            'linear_precision_um': linear_precision_um,
            'angular_precision_arcsec': angular_precision_arcsec,
            'sub_micron_precision': linear_precision_um < 1.0,
            'sub_arcsec_precision': angular_precision_arcsec < 1.0
        }


# Global validator instance
default_validator = ConfigValidator()

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration using default validator.

    Args:
        config (Dict[str, Any]): Configuration to validate

    Returns:
        List[str]: List of validation errors
    """
    return default_validator.validate_config(config)

def validate_or_raise(config: Dict[str, Any]) -> None:
    """
    Validate configuration using default validator and raise exception if errors found.

    Args:
        config (Dict[str, Any]): Configuration to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    default_validator.validate_or_raise(config)

def get_precision_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get precision information using default validator.

    Args:
        config (Dict[str, Any]): Configuration to analyze

    Returns:
        Dict[str, Any]: Precision information
    """
    return default_validator.get_precision_info(config)