"""
Model configuration classes for pipeline model components.

Provides configuration for various model types including Hidden Markov Models,
MCMC methods, and other regime detection approaches with comprehensive
parameter sets for training and inference.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from ..utils.exceptions import ConfigurationError
from .base import BaseConfig


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """
    Base configuration for model components.
    """

    n_states: int = 3
    observed_signal: str = "log_return"
    random_seed: Optional[int] = None

    def validate(self) -> None:
        """Validate model configuration."""
        super().validate()

        if self.n_states < 2:
            raise ConfigurationError(
                f"n_states must be at least 2, got {self.n_states}"
            )

        if self.n_states > 10:
            raise ConfigurationError(
                f"n_states should not exceed 10 for practical reasons, got {self.n_states}"
            )

        if not self.observed_signal:
            raise ConfigurationError("observed_signal cannot be empty")

    def create_component(self) -> Any:
        """Create model component - to be implemented by specific configs."""
        raise NotImplementedError("Subclasses must implement create_component")


@dataclass(frozen=True)
class HMMConfig(ModelConfig):
    """
    Configuration for Hidden Markov Model components.

    Comprehensive configuration covering batch training, online learning,
    stability mechanisms, and all HMM-specific parameters.
    """

    # Core HMM parameters
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization: float = 1e-6
    initialization_method: Literal["kmeans", "random", "custom"] = "kmeans"

    # Stability and robustness
    min_regime_duration: int = 2
    min_variance: float = 1e-8
    check_convergence_every: int = 5
    early_stopping: bool = True
    log_likelihood_threshold: float = -1e10

    # Online learning parameters
    forgetting_factor: float = 0.98
    adaptation_rate: float = 0.05
    min_observations_for_update: int = 10
    parameter_smoothing: bool = True
    smoothing_weight: float = 0.8
    rolling_window_size: int = 1000
    sufficient_stats_decay: float = 0.99

    # Change detection
    enable_change_detection: bool = True
    change_detection_threshold: float = 3.0
    change_detection_window: int = 50
    convergence_tolerance: float = 1e-4
    max_adaptation_iterations: int = 5

    def validate(self) -> None:
        """Validate HMM configuration parameters."""
        super().validate()

        # Validate iteration and convergence parameters
        if self.max_iterations <= 0:
            raise ConfigurationError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )

        if self.tolerance <= 0:
            raise ConfigurationError(
                f"tolerance must be positive, got {self.tolerance}"
            )

        if self.regularization < 0:
            raise ConfigurationError(
                f"regularization must be non-negative, got {self.regularization}"
            )

        # Validate stability parameters
        if self.min_regime_duration < 1:
            raise ConfigurationError(
                f"min_regime_duration must be at least 1, got {self.min_regime_duration}"
            )

        if self.min_variance <= 0:
            raise ConfigurationError(
                f"min_variance must be positive, got {self.min_variance}"
            )

        # Validate online learning parameters
        if not 0.8 <= self.forgetting_factor <= 1.0:
            raise ConfigurationError(
                f"forgetting_factor must be between 0.8 and 1.0, got {self.forgetting_factor}"
            )

        if not 0.001 <= self.adaptation_rate <= 0.5:
            raise ConfigurationError(
                f"adaptation_rate must be between 0.001 and 0.5, got {self.adaptation_rate}"
            )

        if self.min_observations_for_update < 1:
            raise ConfigurationError(
                f"min_observations_for_update must be at least 1, got {self.min_observations_for_update}"
            )

        if not 0.1 <= self.smoothing_weight <= 1.0:
            raise ConfigurationError(
                f"smoothing_weight must be between 0.1 and 1.0, got {self.smoothing_weight}"
            )

        if self.rolling_window_size < 100:
            raise ConfigurationError(
                f"rolling_window_size should be at least 100, got {self.rolling_window_size}"
            )

        if not 0.9 <= self.sufficient_stats_decay <= 1.0:
            raise ConfigurationError(
                f"sufficient_stats_decay must be between 0.9 and 1.0, got {self.sufficient_stats_decay}"
            )

        # Validate change detection parameters
        if self.change_detection_threshold <= 0:
            raise ConfigurationError(
                f"change_detection_threshold must be positive, got {self.change_detection_threshold}"
            )

        if self.change_detection_window < 10:
            raise ConfigurationError(
                f"change_detection_window should be at least 10, got {self.change_detection_window}"
            )

        if self.convergence_tolerance <= 0:
            raise ConfigurationError(
                f"convergence_tolerance must be positive, got {self.convergence_tolerance}"
            )

        if self.max_adaptation_iterations < 1:
            raise ConfigurationError(
                f"max_adaptation_iterations must be at least 1, got {self.max_adaptation_iterations}"
            )

    def create_component(self) -> Any:
        """Create HMM model component."""
        from ..models.hmm import HiddenMarkovModel

        return HiddenMarkovModel(self)

    @classmethod
    def create_conservative(cls) -> "HMMConfig":
        """Create conservative configuration for stable, slow-adapting HMM."""
        return cls(
            n_states=3,
            observed_signal="log_return",
            max_iterations=200,
            tolerance=1e-8,
            initialization_method="kmeans",
            forgetting_factor=0.99,
            adaptation_rate=0.01,
            parameter_smoothing=True,
            smoothing_weight=0.9,
            enable_change_detection=False,
        )

    @classmethod
    def create_aggressive(cls) -> "HMMConfig":
        """Create aggressive configuration for fast-adapting HMM."""
        return cls(
            n_states=4,
            observed_signal="log_return",
            max_iterations=50,
            tolerance=1e-4,
            initialization_method="kmeans",
            forgetting_factor=0.95,
            adaptation_rate=0.1,
            parameter_smoothing=True,
            smoothing_weight=0.6,
            enable_change_detection=True,
            change_detection_threshold=2.0,
        )

    @classmethod
    def create_balanced(cls) -> "HMMConfig":
        """Create balanced configuration for general-purpose HMM."""
        return cls(
            n_states=3,
            observed_signal="log_return",
            max_iterations=100,
            tolerance=1e-6,
            initialization_method="kmeans",
            forgetting_factor=0.98,
            adaptation_rate=0.05,
            parameter_smoothing=True,
            smoothing_weight=0.8,
            enable_change_detection=True,
        )

    def get_cache_key(self) -> str:
        """Generate cache key for this model configuration."""
        return f"hmm_{self.n_states}_{self.observed_signal}_{self.initialization_method}_{hash(self)}"
