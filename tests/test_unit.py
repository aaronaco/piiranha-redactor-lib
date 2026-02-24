from unittest.mock import patch

import pytest

from piiranha_redactor.detector import DetectedEntity, redact
from piiranha_redactor.model import resolve_device


class TestDetectedEntity:

    def test_stores_all_fields(self):
        entity = DetectedEntity(word="Sarah", label="GIVENNAME", score=0.98, start=11, end=16)
        assert entity.word == "Sarah"
        assert entity.label == "GIVENNAME"
        assert entity.score == 0.98
        assert entity.start == 11
        assert entity.end == 16

    def test_repr(self):
        entity = DetectedEntity(word="Sarah", label="GIVENNAME", score=0.98, start=0, end=5)
        result = repr(entity)
        assert "GIVENNAME" in result
        assert "Sarah" in result
        assert "0.98" in result


class TestRedact:

    def test_single_entity(self):
        text = "Contact sarah.connor@skynet.com for details"
        entities = [DetectedEntity(word="sarah.connor@skynet.com", label="EMAIL", score=0.99, start=8, end=31)]
        assert redact(text, entities) == "Contact [EMAIL] for details"

    def test_multiple_entities(self):
        text = "My name is Sarah Connor"
        entities = [
            DetectedEntity(word="Sarah", label="GIVENNAME", score=0.99, start=11, end=16),
            DetectedEntity(word="Connor", label="SURNAME", score=0.98, start=17, end=23),
        ]
        assert redact(text, entities) == "My name is [GIVENNAME] [SURNAME]"

    def test_entities_replaced_without_corrupting_positions(self):
        text = "Email sarah@mail.com and phone 555-123-4567"
        entities = [
            DetectedEntity(word="sarah@mail.com", label="EMAIL", score=0.99, start=6, end=20),
            DetectedEntity(word="555-123-4567", label="TELEPHONENUM", score=0.95, start=31, end=43),
        ]
        result = redact(text, entities)
        assert "sarah@mail.com" not in result
        assert "555-123-4567" not in result
        assert "[EMAIL]" in result
        assert "[TELEPHONENUM]" in result

    def test_unordered_entities_still_work(self):
        text = "My name is Sarah Connor"
        entities = [
            DetectedEntity(word="Connor", label="SURNAME", score=0.98, start=17, end=23),
            DetectedEntity(word="Sarah", label="GIVENNAME", score=0.99, start=11, end=16),
        ]
        assert redact(text, entities) == "My name is [GIVENNAME] [SURNAME]"

    def test_no_entities_returns_original(self):
        text = "The weather is nice today"
        assert redact(text, []) == text

    def test_entity_at_start_of_text(self):
        text = "Sarah lives here"
        entities = [DetectedEntity(word="Sarah", label="GIVENNAME", score=0.99, start=0, end=5)]
        assert redact(text, entities) == "[GIVENNAME] lives here"

    def test_entity_at_end_of_text(self):
        text = "Contact sarah@mail.com"
        entities = [DetectedEntity(word="sarah@mail.com", label="EMAIL", score=0.99, start=8, end=22)]
        assert redact(text, entities) == "Contact [EMAIL]"

    def test_adjacent_entities(self):
        text = "SarahConnor"
        entities = [
            DetectedEntity(word="Sarah", label="GIVENNAME", score=0.99, start=0, end=5),
            DetectedEntity(word="Connor", label="SURNAME", score=0.98, start=5, end=11),
        ]
        assert redact(text, entities) == "[GIVENNAME][SURNAME]"


class TestResolveDevice:

    @patch("piiranha_redactor.model.torch")
    def test_auto_detect_returns_cuda_when_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert resolve_device(None) == "cuda"

    @patch("piiranha_redactor.model.torch")
    def test_auto_detect_returns_cpu_when_no_cuda(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        assert resolve_device(None) == "cpu"

    def test_explicit_cpu_always_works(self):
        assert resolve_device("cpu") == "cpu"

    @patch("piiranha_redactor.model.torch")
    def test_explicit_cuda_raises_when_unavailable(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        with pytest.raises(ValueError, match="CUDA is not available"):
            resolve_device("cuda")

    @patch("piiranha_redactor.model.torch")
    def test_explicit_cuda_works_when_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert resolve_device("cuda") == "cuda"
