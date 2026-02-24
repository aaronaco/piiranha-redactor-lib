import pytest
from piiranha_redactor import PIIRedactor, DetectedEntity


@pytest.fixture(scope="module")
def redactor():
    return PIIRedactor()


class TestInstantiation:

    def test_default_instantiation(self):
        r = PIIRedactor()
        assert r.device in ("cuda", "cpu")
        assert r.threshold == 0.5

    def test_custom_threshold(self):
        r = PIIRedactor(threshold=0.8)
        assert r.threshold == 0.8

    def test_explicit_cpu(self):
        r = PIIRedactor(device="cpu")
        assert r.device == "cpu"

    def test_invalid_cuda_raises(self):
        import torch
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA is not available"):
                PIIRedactor(device="cuda")


class TestDetect:

    def test_detects_email(self, redactor):
        entities = redactor.detect("Please reach out to margaret.chen@protonmail.com for the report")
        labels = [e.label for e in entities]
        assert "EMAIL" in labels

    def test_detects_name(self, redactor):
        entities = redactor.detect("The application was submitted by Priya Raghavan last Monday")
        labels = [e.label for e in entities]
        assert "GIVENNAME" in labels or "SURNAME" in labels

    def test_detects_phone_number(self, redactor):
        entities = redactor.detect("You can call me at +1 (415) 839-2746 after noon")
        labels = [e.label for e in entities]
        assert "TELEPHONENUM" in labels

    def test_detects_street_address(self, redactor):
        entities = redactor.detect("I live at 742 Evergreen Terrace in Springfield")
        labels = [e.label for e in entities]
        has_address = any(l in labels for l in ("STREET", "CITY", "BUILDINGNUM"))
        assert has_address

    def test_entity_positions_align_with_text(self, redactor):
        text = "Send the invoice to margaret.chen@protonmail.com by Friday"
        entities = redactor.detect(text)
        for entity in entities:
            assert text[entity.start:entity.end] == entity.word

    def test_entity_fields_have_correct_types(self, redactor):
        entities = redactor.detect("My name is Priya Raghavan and my email is priya.r@outlook.com")
        assert len(entities) > 0
        for entity in entities:
            assert isinstance(entity, DetectedEntity)
            assert isinstance(entity.word, str)
            assert isinstance(entity.label, str)
            assert isinstance(entity.score, float)
            assert isinstance(entity.start, int)
            assert isinstance(entity.end, int)
            assert 0.0 <= entity.score <= 1.0

    def test_empty_string(self, redactor):
        assert redactor.detect("") == []

    def test_whitespace_only(self, redactor):
        assert redactor.detect("   ") == []

    def test_no_pii(self, redactor):
        assert redactor.detect("The quarterly earnings report is due next week") == []

    def test_lower_threshold_finds_more_or_equal(self, redactor):
        text = "Priya Raghavan started working at the firm in March"
        low = redactor.detect(text, threshold=0.1)
        high = redactor.detect(text, threshold=0.99)
        assert len(low) >= len(high)

    def test_multiple_entity_types_in_one_text(self, redactor):
        text = "My name is Carlos Mendoza, email carlos.mendoza@gmail.com, phone +34 612 345 678"
        entities = redactor.detect(text)
        labels = {e.label for e in entities}
        assert len(labels) >= 2


class TestRedact:

    def test_email_is_replaced(self, redactor):
        result = redactor.redact("Forward this to margaret.chen@protonmail.com please")
        assert "margaret.chen@protonmail.com" not in result
        assert "[EMAIL]" in result

    def test_name_is_replaced(self, redactor):
        result = redactor.redact("The hiring manager Priya Raghavan approved the offer")
        assert "[GIVENNAME]" in result or "[SURNAME]" in result

    def test_non_pii_text_unchanged(self, redactor):
        text = "The quarterly earnings report is due next week"
        assert redactor.redact(text) == text

    def test_surrounding_text_preserved(self, redactor):
        result = redactor.redact("Please contact margaret.chen@protonmail.com before Friday")
        assert result.startswith("Please contact")
        assert result.endswith("before Friday")

    def test_empty_string_passthrough(self, redactor):
        assert redactor.redact("") == ""

    def test_high_threshold_redacts_less(self, redactor):
        text = "Priya Raghavan sent the documents to carlos.mendoza@gmail.com"
        strict = redactor.redact(text, threshold=0.9999)
        default = redactor.redact(text)
        assert len(default) <= len(strict)


class TestRedactWithDetails:

    def test_returns_expected_keys(self, redactor):
        result = redactor.redact_with_details("Contact margaret.chen@protonmail.com for info")
        assert "redacted_text" in result
        assert "entities" in result

    def test_redacted_text_matches_redact(self, redactor):
        text = "My email is margaret.chen@protonmail.com"
        details = redactor.redact_with_details(text)
        simple = redactor.redact(text)
        assert details["redacted_text"] == simple

    def test_entities_match_detect(self, redactor):
        text = "My email is margaret.chen@protonmail.com"
        details = redactor.redact_with_details(text)
        detected = redactor.detect(text)
        assert len(details["entities"]) == len(detected)

    def test_empty_string(self, redactor):
        result = redactor.redact_with_details("")
        assert result["redacted_text"] == ""
        assert result["entities"] == []
