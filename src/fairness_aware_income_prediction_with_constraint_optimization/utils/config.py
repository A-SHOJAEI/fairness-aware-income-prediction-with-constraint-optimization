"""Configuration management utilities."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """Configuration manager for loading and accessing YAML configuration files.

    This class provides a centralized way to manage configuration settings
    for the fairness-aware income prediction project.

    Attributes:
        config: Dictionary containing the loaded configuration.
        config_path: Path to the configuration file.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file. If None, uses
                the default configuration file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the configuration file is malformed.
        """
        if config_path is None:
            # Get the default config path relative to the project root
            project_root = Path(__file__).parents[4]
            config_path = project_root / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing the configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the configuration file is malformed.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ValueError("Configuration file is empty")

            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def _setup_logging(self) -> None:
        """Set up logging based on configuration."""
        log_config = self.config.get('logging', {})

        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'logs/training.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get(
                'format',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key using dot notation (e.g., 'model.algorithm').
            default: Default value if key is not found.

        Returns:
            The configuration value or default if not found.

        Examples:
            >>> config = Config()
            >>> config.get('data.test_size', 0.2)
            0.2
            >>> config.get('model.algorithm')
            'lightgbm'
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key: Configuration key using dot notation.
            value: Value to set.

        Examples:
            >>> config = Config()
            >>> config.set('data.test_size', 0.3)
            >>> config.get('data.test_size')
            0.3
        """
        keys = key.split('.')
        config_dict = self.config

        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]

        config_dict[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates.

        Examples:
            >>> config = Config()
            >>> config.update({'data': {'test_size': 0.3}})
        """
        self._deep_update(self.config, updates)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary.

        Args:
            base_dict: Base dictionary to update.
            update_dict: Dictionary with updates.
        """
        for key, value in update_dict.items():
            if (key in base_dict and
                isinstance(base_dict[key], dict) and
                isinstance(value, dict)):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save the current configuration to a YAML file.

        Args:
            output_path: Output file path. If None, overwrites the original file.
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)

    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.get('data', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return self.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-related configuration."""
        return self.get('training', {})

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization-related configuration."""
        return self.get('optimization', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation-related configuration."""
        return self.get('evaluation', {})