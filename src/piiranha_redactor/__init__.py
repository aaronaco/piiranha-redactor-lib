"""
piiranha-redactor: Local PII detection and redaction powered by Piiranha v1 (DeBERTa-v3).

    from piiranha_redactor import PIIRedactor

    redactor = PIIRedactor()
    redactor.redact("Hi I'm John Smith, email is john.smith@example.com")
    # "Hi I'm [GIVENNAME] [SURNAME], email is [EMAIL]"
"""

from __future__ import annotations

import logging

from tabulate import tabulate

from piiranha_redactor.detector import DetectedEntity, detect, redact
from piiranha_redactor.model import load_model, resolve_device

__all__ = ["PIIRedactor", "DetectedEntity"]
__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _log_entities(entities: list[DetectedEntity]) -> None:
    if not entities or not logger.isEnabledFor(logging.DEBUG):
        return

    table = tabulate(
        [[e.word, e.label, f"{e.score:.4f}", e.start, e.end] for e in entities],
        headers=["Word", "Label", "Score", "Start", "End"],
        tablefmt="simple",
    )
    logger.debug("Detected %d PII entities:\n%s", len(entities), table)


class PIIRedactor:
    """Main entry point for PII detection and redaction.

    Load once at application startup, reuse for every call.

    Args:
        device: "cuda", "cpu", or None (auto-detect).
        threshold: Minimum confidence score (0.0-1.0). Default 0.5.
    """

    def __init__(
        self,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._device = resolve_device(device)
        self._default_threshold = threshold
        self._tokenizer, self._model = load_model(self._device)

    def detect(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[DetectedEntity]:
        """Detect PII entities in text, returning positions and confidence scores."""
        if not text or not text.strip():
            return []

        active_threshold = threshold if threshold is not None else self._default_threshold

        entities = detect(
            text=text,
            tokenizer=self._tokenizer,
            model=self._model,
            device=self._device,
            threshold=active_threshold,
        )

        _log_entities(entities)
        return entities

    def redact(
        self,
        text: str,
        threshold: float | None = None,
    ) -> str:
        """Replace detected PII with ``[LABEL]`` tags."""
        if not text or not text.strip():
            return text

        active_threshold = threshold if threshold is not None else self._default_threshold

        entities = detect(
            text=text,
            tokenizer=self._tokenizer,
            model=self._model,
            device=self._device,
            threshold=active_threshold,
        )

        _log_entities(entities)
        return redact(text=text, entities=entities)

    def redact_with_details(
        self,
        text: str,
        threshold: float | None = None,
    ) -> dict:
        """Redact PII and return both the cleaned text and entity list."""
        if not text or not text.strip():
            return {"redacted_text": text, "entities": []}

        active_threshold = threshold if threshold is not None else self._default_threshold

        entities = detect(
            text=text,
            tokenizer=self._tokenizer,
            model=self._model,
            device=self._device,
            threshold=active_threshold,
        )

        _log_entities(entities)
        redacted_text = redact(text=text, entities=entities)

        return {
            "redacted_text": redacted_text,
            "entities": entities,
        }

    @property
    def device(self) -> str:
        return self._device

    @property
    def threshold(self) -> float:
        return self._default_threshold
