"""DIVE-SLR v2 research implementation.

Synthetic fixtures exercise contracts only. They are not benchmark evidence.
"""

from .config import ConfigError, config_hash, load_config, validate_config
from .artifacts import ArtifactError, ArtifactRecord, ArtifactResolver, ResolvedArtifact

__all__ = [
    "ArtifactError",
    "ArtifactRecord",
    "ArtifactResolver",
    "ConfigError",
    "ResolvedArtifact",
    "config_hash",
    "load_config",
    "validate_config",
]
__version__ = "0.1.0"
