"""
Base configuration class for all pipeline components.

Provides common functionality for validation, serialization, and component creation
that all specific configuration classes inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import json
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass(frozen=True)
class BaseConfig(ABC):
    """
    Base configuration class for all pipeline components.
    
    Provides common functionality including:
    - JSON serialization/deserialization
    - Validation framework
    - Component creation interface
    - Configuration comparison and hashing
    """
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
            ConfigurationError: If configuration is inconsistent
        """
        # Default implementation - subclasses can override
        pass
    
    @abstractmethod
    def create_component(self) -> Any:
        """
        Create the component instance from this configuration.
        
        Returns:
            Initialized component instance
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            Configuration instance
        """
        return cls(**config_dict)
    
    def to_json(self) -> str:
        """
        Convert configuration to JSON string.
        
        Returns:
            JSON representation of configuration
        """
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseConfig':
        """
        Create configuration from JSON string.
        
        Args:
            json_str: JSON string with configuration
            
        Returns:
            Configuration instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration instance
        """
        with open(filepath, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)
    
    def copy(self, **kwargs) -> 'BaseConfig':
        """
        Create a copy of configuration with optional parameter updates.
        
        Args:
            **kwargs: Parameters to update in the copy
            
        Returns:
            New configuration instance with updated parameters
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)
    
    def __post_init__(self):
        """Called after dataclass initialization to run validation."""
        self.validate()
    
    def __eq__(self, other) -> bool:
        """Check equality with another configuration."""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()
    
    def __hash__(self) -> int:
        """Generate hash for configuration (useful for caching)."""
        # Convert dict to sorted tuple of items for consistent hashing
        items = sorted(self.to_dict().items())
        return hash(str(items))
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        class_name = self.__class__.__name__
        params = []
        for key, value in self.to_dict().items():
            if isinstance(value, str):
                params.append(f"{key}='{value}'")
            else:
                params.append(f"{key}={value}")
        return f"{class_name}({', '.join(params)})"