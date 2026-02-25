"""
Model Version Manager

Manages versioning, rollback, and metrics tracking for ML models.

Features:
- Save model versions with timestamps and checksums
- Maintain version history (configurable max versions)
- Support rollback to previous versions
- Store and compare metrics across versions
- Track model lineage and training parameters
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError
from utils.paths import get_project_root

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Model saving may be limited.")


@dataclass
class ModelMetrics:
    """Metrics for a model version"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 0.0

    # Additional metrics
    training_samples: int = 0
    validation_samples: int = 0
    training_time_seconds: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        custom = data.pop("custom_metrics", {})
        metrics = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        metrics.custom_metrics = custom
        return metrics


@dataclass
class ModelVersion:
    """Represents a single model version"""
    version_id: str
    model_name: str
    created_at: datetime
    checksum: str
    file_path: str
    metrics: ModelMetrics

    # Training info
    training_params: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["metrics"] = ModelMetrics.from_dict(data["metrics"])
        return cls(**data)


class ModelVersionManager:
    """
    Manages model versions with persistence, rollback, and metrics tracking.

    Example:
        manager = ModelVersionManager("models/ensemble")

        # Save a new version
        version = manager.save_version(
            model=trained_model,
            model_name="ensemble_v1",
            metrics=ModelMetrics(accuracy=0.85, auc_roc=0.92),
            training_params={"learning_rate": 0.01}
        )

        # Get current production version
        current = manager.get_current_version()

        # Rollback if needed
        if current.metrics.accuracy < 0.80:
            manager.rollback()
    """

    DEFAULT_MAX_VERSIONS = 5
    MANIFEST_FILE = "versions.json"

    def __init__(
        self,
        base_path: str,
        max_versions: int = DEFAULT_MAX_VERSIONS,
        auto_cleanup: bool = True
    ):
        """
        Initialize version manager.

        Args:
            base_path: Directory to store model versions
            max_versions: Maximum versions to keep (older ones are cleaned up)
            auto_cleanup: Automatically cleanup old versions
        """
        self.base_path = Path(base_path)
        self.max_versions = max_versions
        self.auto_cleanup = auto_cleanup

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.base_path / "versions"
        self.versions_dir.mkdir(exist_ok=True)

        # Load existing versions
        self._versions: Dict[str, ModelVersion] = {}
        self._version_order: List[str] = []  # Ordered by creation time
        self._current_version: Optional[str] = None

        self._load_manifest()

        logger.info(
            f"ModelVersionManager initialized at {self.base_path} "
            f"({len(self._versions)} versions)"
        )

    def _get_manifest_path(self) -> Path:
        return self.base_path / self.MANIFEST_FILE

    def _load_manifest(self) -> None:
        """Load version manifest from disk"""
        manifest_path = self._get_manifest_path()
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)

                self._version_order = data.get("version_order", [])
                self._current_version = data.get("current_version")

                for version_id, version_data in data.get("versions", {}).items():
                    self._versions[version_id] = ModelVersion.from_dict(version_data)

            except Exception as e:
                logger.error(f"Failed to load version manifest: {e}")
                self._versions = {}
                self._version_order = []
                self._current_version = None

    def _save_manifest(self) -> None:
        """Save version manifest to disk"""
        manifest_path = self._get_manifest_path()
        try:
            data = {
                "version_order": self._version_order,
                "current_version": self._current_version,
                "versions": {
                    vid: version.to_dict()
                    for vid, version in self._versions.items()
                }
            }

            with open(manifest_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save version manifest: {e}")

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = len(self._versions) + 1
        return f"v{counter}_{timestamp}"

    def _cleanup_old_versions(self) -> None:
        """Remove old versions beyond max_versions limit"""
        if not self.auto_cleanup:
            return

        while len(self._version_order) > self.max_versions:
            oldest_id = self._version_order[0]

            # Don't delete current version
            if oldest_id == self._current_version:
                break

            self.delete_version(oldest_id)
            logger.info(f"Cleaned up old version: {oldest_id}")

    def save_version(
        self,
        model: Any,
        model_name: str,
        metrics: Optional[ModelMetrics] = None,
        training_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        set_as_current: bool = True
    ) -> ModelVersion:
        """
        Save a new model version.

        Args:
            model: The model object to save (must be serializable)
            model_name: Name identifier for the model
            metrics: Model performance metrics
            training_params: Training hyperparameters
            feature_names: List of feature names used
            description: Human-readable description
            tags: Tags for categorization
            set_as_current: Set this as the current production version

        Returns:
            ModelVersion object for the saved version
        """
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib required for model saving")

        version_id = self._generate_version_id()
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)

        # Save model file
        model_file = version_dir / f"{model_name}.joblib"
        joblib.dump(model, model_file)

        # Compute checksum
        checksum = self._compute_checksum(model_file)

        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=datetime.now(),
            checksum=checksum,
            file_path=str(model_file),
            metrics=metrics or ModelMetrics(),
            training_params=training_params or {},
            feature_names=feature_names or [],
            description=description,
            parent_version=self._current_version,
            tags=tags or []
        )

        # Update state
        self._versions[version_id] = version
        self._version_order.append(version_id)

        if set_as_current:
            self._current_version = version_id

        self._save_manifest()
        self._cleanup_old_versions()

        logger.info(
            f"Saved model version {version_id} "
            f"(metrics: accuracy={version.metrics.accuracy:.4f})"
        )

        return version

    def load_version(self, version_id: str) -> Tuple[Any, ModelVersion]:
        """
        Load a specific model version.

        Args:
            version_id: Version ID to load

        Returns:
            Tuple of (model, version_info)
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib required for model loading")

        version = self._versions[version_id]
        model_path = Path(version.file_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Verify checksum - reject tampered models
        current_checksum = self._compute_checksum(model_path)
        if current_checksum != version.checksum:
            raise ModelSecurityError(
                f"Model checksum mismatch for {model_path}: "
                f"expected {version.checksum[:16]}..., got {current_checksum[:16]}..."
            )

        model = safe_load_model(str(model_path), allow_unverified=False)
        logger.info(f"Loaded model version {version_id}")

        return model, version

    def load_current(self) -> Tuple[Any, ModelVersion]:
        """Load the current production version"""
        if not self._current_version:
            raise ValueError("No current version set")
        return self.load_version(self._current_version)

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get info about current version without loading model"""
        if self._current_version:
            return self._versions.get(self._current_version)
        return None

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version info by ID"""
        return self._versions.get(version_id)

    def list_versions(self) -> List[ModelVersion]:
        """List all versions in order (newest first)"""
        return [
            self._versions[vid]
            for vid in reversed(self._version_order)
            if vid in self._versions
        ]

    def set_current_version(self, version_id: str) -> None:
        """Set the current production version"""
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        old_version = self._current_version
        self._current_version = version_id
        self._save_manifest()

        logger.info(f"Set current version: {old_version} -> {version_id}")

    def rollback(self, steps: int = 1) -> Optional[ModelVersion]:
        """
        Rollback to a previous version.

        Args:
            steps: Number of versions to rollback (1 = previous version)

        Returns:
            The new current version, or None if rollback not possible
        """
        if not self._current_version or len(self._version_order) < 2:
            logger.warning("Cannot rollback: insufficient version history")
            return None

        try:
            current_idx = self._version_order.index(self._current_version)
            target_idx = current_idx - steps

            if target_idx < 0:
                logger.warning(
                    f"Cannot rollback {steps} steps. "
                    f"Only {current_idx} previous versions available."
                )
                target_idx = 0

            target_version = self._version_order[target_idx]
            old_version = self._current_version
            self._current_version = target_version
            self._save_manifest()

            logger.info(
                f"Rolled back from {old_version} to {target_version} "
                f"({current_idx - target_idx} steps)"
            )

            return self._versions[target_version]

        except ValueError:
            logger.error("Current version not found in version order")
            return None

    def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific version.

        Args:
            version_id: Version to delete

        Returns:
            True if deleted, False if not found or is current version
        """
        if version_id not in self._versions:
            return False

        if version_id == self._current_version:
            logger.warning("Cannot delete current version. Rollback first.")
            return False

        version = self._versions[version_id]

        # Delete files
        version_dir = Path(version.file_path).parent
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Update state
        del self._versions[version_id]
        if version_id in self._version_order:
            self._version_order.remove(version_id)

        self._save_manifest()
        logger.info(f"Deleted version {version_id}")

        return True

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare metrics between two versions.

        Returns:
            Dictionary with comparison results
        """
        v1 = self._versions.get(version_id_1)
        v2 = self._versions.get(version_id_2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        comparison = {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "metrics_diff": {},
            "better_version": None
        }

        # Compare core metrics
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        improvements = 0

        for metric in metrics_to_compare:
            val1 = getattr(v1.metrics, metric, 0)
            val2 = getattr(v2.metrics, metric, 0)
            diff = val2 - val1
            comparison["metrics_diff"][metric] = {
                "v1": val1,
                "v2": val2,
                "diff": diff,
                "improved": diff > 0
            }
            if diff > 0:
                improvements += 1

        # Determine better version (by majority of metrics)
        if improvements > len(metrics_to_compare) / 2:
            comparison["better_version"] = version_id_2
        elif improvements < len(metrics_to_compare) / 2:
            comparison["better_version"] = version_id_1
        else:
            comparison["better_version"] = "tie"

        return comparison

    def get_best_version(self, metric: str = "accuracy") -> Optional[ModelVersion]:
        """
        Get the version with the best value for a given metric.

        Args:
            metric: Metric name to optimize for

        Returns:
            Best version or None if no versions exist
        """
        if not self._versions:
            return None

        best_version = None
        best_value = float('-inf')

        for version in self._versions.values():
            value = getattr(version.metrics, metric, 0)
            if isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best_version = version

        return best_version

    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get metrics history as a DataFrame for analysis.

        Returns:
            DataFrame with version metrics over time
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas required for metrics history")
            return None

        records = []
        for vid in self._version_order:
            version = self._versions.get(vid)
            if version:
                record = {
                    "version_id": vid,
                    "created_at": version.created_at,
                    "is_current": vid == self._current_version,
                    **version.metrics.to_dict()
                }
                records.append(record)

        return pd.DataFrame(records)


# Convenience function to get a version manager
def get_model_version_manager(model_type: str = "ensemble") -> ModelVersionManager:
    """
    Get a model version manager for a specific model type.

    Args:
        model_type: Type of model (e.g., "ensemble", "xgboost", "lstm")

    Returns:
        ModelVersionManager instance
    """
    base_path = get_project_root() / "data" / "models" / model_type
    return ModelVersionManager(str(base_path))
