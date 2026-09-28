#!/usr/bin/env python3
"""
Network utilities - connectivity detection, model downloading.
"""

from __future__ import annotations

import socket
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import zipfile
import shutil

from core.config import BASE_DIR
from core.logging import log


def is_online(timeout: float = 2.0) -> bool:
    """Check internet connectivity by pinging Google DNS."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        try:
            urllib.request.urlopen("http://www.google.com", timeout=timeout)
            return True
        except Exception:
            return False


# Vosk model URLs
VOSK_MODELS = {
    "small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "folder": "vosk-model-small-en-us-0.15",
        "size_mb": 40,
    },
    "large": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "folder": "vosk-model-en-us-0.22",
        "size_mb": 1800,
    },
    "large-lgraph": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
        "folder": "vosk-model-en-us-0.22-lgraph",
        "size_mb": 50,
    },
}


def get_vosk_model_dir() -> Path:
    """Get the directory where Vosk models are stored."""
    model_dir = BASE_DIR / "models" / "vosk"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def ensure_vosk_model(model_size: str = "small", model_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Ensure Vosk model exists, downloading if necessary.
    
    Args:
        model_size: "small", "large", or "large-lgraph"
        model_dir: Custom model directory (default: BASE_DIR/models/vosk)
    
    Returns:
        Path to the model folder, or None if failed
    """
    if model_size not in VOSK_MODELS:
        log.error("Unknown Vosk model size: %s", model_size)
        return None
    
    model_info = VOSK_MODELS[model_size]
    target_dir = (model_dir or get_vosk_model_dir()) / model_info["folder"]
    
    if target_dir.exists():
        log.info("Vosk model '%s' already exists at %s", model_size, target_dir)
        return target_dir
    
    log.info("Downloading Vosk model '%s' (%d MB)...", model_size, model_info["size_mb"])
    
    try:
        model_dir = model_dir or get_vosk_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = model_dir / f"{model_info['folder']}.zip"
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                if percent % 10 == 0:
                    log.info("Download progress: %d%%", percent)
        
        urllib.request.urlretrieve(model_info["url"], zip_path, reporthook=report_progress)
        
        # Extract
        log.info("Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up zip
        zip_path.unlink()
        
        log.info("Vosk model '%s' ready at %s", model_size, target_dir)
        return target_dir
        
    except Exception as e:
        log.error("Failed to download/extract Vosk model: %s", e)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        return None


def get_vosk_model_path(model_size: str = "small") -> Optional[str]:
    """Get path to Vosk model, downloading if needed. Returns string path for config."""
    path = ensure_vosk_model(model_size)
    return str(path) if path else None