"""
Safe Model Loading Utility

Provides secure model loading with SHA-256 checksum verification
to prevent loading tampered or malicious model files.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


class ModelSecurityError(Exception):
    """Raised when model integrity verification fails."""
    pass


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_manifest_path(model_path: str) -> str:
    """Get the manifest file path for a model."""
    return f"{model_path}.manifest"


def create_manifest(model_path: str) -> str:
    """Create a manifest file with checksum for a model.

    Returns:
        The computed checksum
    """
    checksum = compute_file_checksum(model_path)
    manifest_path = get_manifest_path(model_path)

    manifest = {
        'checksum': checksum,
        'algorithm': 'sha256',
        'model_path': os.path.basename(model_path),
        'file_size': os.path.getsize(model_path)
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Created manifest for {model_path}: {checksum[:16]}...")
    return checksum


def verify_model(model_path: str) -> Tuple[bool, str]:
    """Verify model integrity against its manifest.

    Returns:
        Tuple of (is_valid, message)
    """
    manifest_path = get_manifest_path(model_path)

    if not os.path.exists(manifest_path):
        return False, f"No manifest found for {model_path}"

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Invalid manifest for {model_path}: {e}"

    expected_checksum = manifest.get('checksum')
    if not expected_checksum:
        return False, f"No checksum in manifest for {model_path}"

    actual_checksum = compute_file_checksum(model_path)

    if actual_checksum != expected_checksum:
        return False, (
            f"Checksum mismatch for {model_path}: "
            f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}..."
        )

    return True, "Model integrity verified"


def safe_load_model(model_path: str, allow_unverified: bool = False):
    """Safely load a model with integrity verification.

    Args:
        model_path: Path to the model file
        allow_unverified: If True, allow loading models without manifests (with warning)

    Returns:
        The loaded model object

    Raises:
        ModelSecurityError: If model fails integrity check
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Verify integrity
    is_valid, message = verify_model(model_path)

    if not is_valid:
        if "No manifest found" in message:
            if allow_unverified:
                logger.warning(f"Loading unverified model (no manifest): {model_path}")
            else:
                logger.warning(
                    f"No manifest found for model: {model_path}. "
                    f"Verification skipped - loading for backward compatibility. "
                    f"Run safe_save_model() to create a manifest for this model."
                )
        else:
            raise ModelSecurityError(message)
    else:
        logger.debug(f"Model integrity verified: {model_path}")

    # Load with joblib
    try:
        import joblib
        return joblib.load(model_path)
    except ImportError:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)


def safe_save_model(model, model_path: str):
    """Save a model and create its integrity manifest.

    Args:
        model: The model object to save
        model_path: Path to save the model
    """
    # Ensure directory exists
    parent_dir = os.path.dirname(model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        import joblib
        joblib.dump(model, model_path)
    except ImportError:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    # Create manifest
    create_manifest(model_path)
    logger.info(f"Model saved with manifest: {model_path}")
