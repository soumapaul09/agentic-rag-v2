"""Configuration loader for Agentic RAG system."""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Loads and provides access to configuration settings from YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the config loader.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing all configuration settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key_path: Path to the config value using dot notation (e.g., 'models.embedding.name')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_path(self, key_path: str) -> Path:
        """Get a path configuration value as a Path object.

        Args:
            key_path: Path to the config value using dot notation

        Returns:
            Path object
        """
        value = self.get(key_path)
        return Path(value) if value else None

    @property
    def paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self.config.get("paths", {})

    @property
    def models(self) -> Dict[str, Any]:
        """Get all model configurations."""
        return self.config.get("models", {})

    @property
    def prompts(self) -> Dict[str, str]:
        """Get all prompt configurations."""
        return self.config.get("prompts", {})


_config = None # Global config instance


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Get or create the global configuration instance.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config


