"""Shared Presidio recognizer setup for the text and tabular detectors.

Presidio's default registry loads 17 recognizers, which covers US identifiers plus
a handful of global formats (card, IBAN, email, phone). The installed package also
ships country ID recognizers for Australia, Canada, India, Nigeria, the
Philippines, Singapore, the UK and South Africa, but leaves them switched off.
This module turns the English-capable ones on and applies per-entity score
thresholds, because recognizers with no checksum to validate against score well
below the global confidence threshold even when they are correct.
"""
import presidio_analyzer.predefined_recognizers as predefined_recognizers
from presidio_analyzer import RecognizerRegistry

from config import (
    CONFIDENCE_THRESHOLD,
    ENTITY_SCORE_THRESHOLDS,
    INTERNATIONAL_RECOGNIZERS,
)


def build_analyzer_registry(languages=("en",)):
    """Return Presidio's default registry plus the country ID recognizers."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=list(languages))

    for class_name in INTERNATIONAL_RECOGNIZERS:
        recognizer_cls = getattr(predefined_recognizers, class_name, None)
        if recognizer_cls is None:
            # A different Presidio build may not ship every recognizer. Skipping one
            # loses that country's coverage; failing here would lose all detection.
            print(f"⚠️  Recognizer unavailable in this Presidio build: {class_name}")
            continue
        registry.add_recognizer(recognizer_cls())

    return registry


def passes_entity_threshold(entity_type, score, default_threshold=CONFIDENCE_THRESHOLD):
    """Apply the per-entity threshold, falling back to the global one."""
    return score >= ENTITY_SCORE_THRESHOLDS.get(entity_type, default_threshold)
