"""Utilities for loading models with attention backend fallbacks."""

from __future__ import annotations

from typing import Any, Dict


def load_pretrained_with_attention_fallback(
    model_cls,
    model_path: str,
    *,
    preferred: str = "flash_attention_2",
    fallback: str = "eager",
    **kwargs: Dict[str, Any],
):
    """Load a model and retry with a safer attention backend if needed."""

    try:
        return model_cls.from_pretrained(model_path, attn_implementation=preferred, **kwargs)
    except Exception as exc:
        print(
            f"Warning: failed to load {model_cls.__name__} with {preferred}: {exc}. "
            f"Falling back to {fallback}.",
            flush=True,
        )
        return model_cls.from_pretrained(model_path, attn_implementation=fallback, **kwargs)
