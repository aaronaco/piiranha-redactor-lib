from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class DetectedEntity:
    """A single PII entity found in text."""

    def __init__(self, word: str, label: str, score: float, start: int, end: int):
        self.word = word
        self.label = label
        self.score = score
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"DetectedEntity(label={self.label!r}, word={self.word!r}, score={self.score})"


def chunk_text(text: str, tokenizer: AutoTokenizer, max_tokens: int = 256) -> list[str]:
    """Split text into word-boundary-safe chunks that fit the model's 256-token window."""
    words = text.split()
    chunks: list[str] = []
    current_chunk_words: list[str] = []
    current_token_count = 0

    for word in words:
        tokens_for_this_word = tokenizer.tokenize(word)
        # +2 for [CLS] and [SEP] special tokens
        would_exceed_limit = (current_token_count + len(tokens_for_this_word) + 2) > max_tokens

        if would_exceed_limit:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]
            current_token_count = len(tokens_for_this_word)
        else:
            current_chunk_words.append(word)
            current_token_count += len(tokens_for_this_word)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def detect(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: str,
    threshold: float,
) -> list[DetectedEntity]:
    """Detect all PII entities in text with their character positions."""
    chunks = chunk_text(text, tokenizer)
    all_detected_entities: list[DetectedEntity] = []
    current_char_offset = 0

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_label_ids = probabilities.argmax(dim=-1)
        confidence_scores = probabilities.max(dim=-1).values

        merged_entities: list[dict] = []

        for label_id, confidence_score, (char_start, char_end) in zip(
            predicted_label_ids, confidence_scores, offset_mapping
        ):
            label = model.config.id2label[label_id.item()]

            if label == "O":
                continue

            if char_start == char_end:
                continue

            score = round(confidence_score.item(), 4)

            if score < threshold:
                continue

            char_start = char_start.item()
            char_end = char_end.item()

            is_continuation = (
                merged_entities
                and merged_entities[-1]["label"] == label
                and char_start == merged_entities[-1]["end"]
            )

            if is_continuation:
                merged_entities[-1]["word"] = chunk[merged_entities[-1]["start"]:char_end]
                merged_entities[-1]["end"] = char_end
                merged_entities[-1]["score"] = round(
                    (merged_entities[-1]["score"] + score) / 2, 4
                )
            else:
                merged_entities.append({
                    "word": chunk[char_start:char_end],
                    "label": label,
                    "score": score,
                    "start": char_start,
                    "end": char_end,
                })

        for entity in merged_entities:
            all_detected_entities.append(DetectedEntity(
                word=entity["word"],
                label=entity["label"].replace("I-", "").replace("B-", ""),
                score=entity["score"],
                start=entity["start"] + current_char_offset,
                end=entity["end"] + current_char_offset,
            ))

        current_char_offset += len(chunk) + 1

    return all_detected_entities


def redact(text: str, entities: list[DetectedEntity]) -> str:
    """Replace detected PII entities with ``[LABEL]`` tags.

    Replaces from end-to-start so earlier character positions stay valid.
    """
    entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)

    redacted_text = text
    for entity in entities_sorted:
        redacted_text = redacted_text[:entity.start] + f"[{entity.label}]" + redacted_text[entity.end:]

    return redacted_text
