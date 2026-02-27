from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForTokenClassification | None = None
_loaded_device: str | None = None


def is_model_cached_for_device(device: str) -> bool:
    """Return True if model/tokenizer are already loaded for the given device."""
    return _tokenizer is not None and _model is not None and _loaded_device == device


def load_model(device: str) -> tuple[AutoTokenizer, AutoModelForTokenClassification]:
    """Load and cache the tokenizer and model. Downloads on first call (~1.1 GB)."""
    global _tokenizer, _model, _loaded_device

    if is_model_cached_for_device(device):
        return _tokenizer, _model

    if _loaded_device is not None and _loaded_device != device:
        _tokenizer = None
        _model = None

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    _model.eval()
    _model = _model.to(device)
    _loaded_device = device

    return _tokenizer, _model


def resolve_device(requested_device: str | None) -> str:
    """Resolve device to 'cuda' or 'cpu'. Auto-detects if None."""
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Device 'cuda' was requested but CUDA is not available. "
            "Use device='cpu' or None to auto-detect."
        )

    return requested_device
