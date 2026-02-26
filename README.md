# piiranha-redactor

A local PII detection and redaction library powered by [Piiranha v1](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) (DeBERTa-v3).

**The model runs entirely on your machine. No data ever leaves your environment.**

Detects 17 types of PII across 6 languages with 98.27% recall.

## Installation

```bash
pip install git+https://github.com/aaronaco/piiranha-redactor-lib.git
```

> **Note:** PyTorch is required but not installed automatically because the right version depends on your hardware (CPU vs CUDA). Install it first:
>
> ```bash
> # CPU only
> pip install torch --index-url https://download.pytorch.org/whl/cpu
>
> # CUDA 12.4 (for NVIDIA GPUs)
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

The model (~1.1GB) downloads automatically on first use and is cached locally at `~/.cache/huggingface/hub`.

## Usage

```python
from piiranha_redactor import PIIRedactor

redactor = PIIRedactor()
```

### Redact PII

```python
clean_text = redactor.redact("Hi I'm John Smith, email is john.smith@example.com")
# "Hi I'm [GIVENNAME] [SURNAME], email is [EMAIL]"
```

### Detect PII (with positions)

```python
entities = redactor.detect("Hi I'm John Smith, email is john.smith@example.com")
for entity in entities:
    print(entity.label, entity.word, entity.score)
# GIVENNAME  John                    0.9983
# SURNAME    Smith                   0.9980
# EMAIL      john.smith@example.com  1.0
```

### Redact and get details

```python
result = redactor.redact_with_details("Hi I'm John, email is john.smith@example.com")
print(result["redacted_text"])  # "Hi I'm [GIVENNAME], email is [EMAIL]"
print(result["entities"])       # [DetectedEntity(...), ...]
```

### Use with any LLM provider

```python
from piiranha_redactor import PIIRedactor
import openai

redactor = PIIRedactor()
client = openai.OpenAI(api_key="...")

user_input = "My name is John Smith and my email is john.smith@example.com. Summarize this."
clean_input = redactor.redact(user_input)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": clean_input}]
)
```

## Configuration

```python
redactor = PIIRedactor()                   # auto-detect device (GPU if available)
redactor = PIIRedactor(device="cpu")       # force CPU
redactor = PIIRedactor(device="cuda")      # force GPU
redactor = PIIRedactor(threshold=0.8)      # higher confidence required

redactor.redact("some text", threshold=0.9)   # override per call
redactor.detect("some text", threshold=0.3)
```

## Supported PII Types

| Label | Description |
|---|---|
| GIVENNAME | First name |
| SURNAME | Last name |
| EMAIL | Email address |
| TELEPHONENUM | Phone number |
| USERNAME | Username / handle |
| PASSWORD | Password |
| STREET | Street address |
| CITY | City |
| ZIPCODE | Zip / postal code |
| BUILDINGNUM | Building number |
| SOCIALNUM | Social security number |
| CREDITCARDNUMBER | Credit card number |
| DATEOFBIRTH | Date of birth |
| DRIVERLICENSENUM | Driver's license number |
| IDCARDNUM | ID card number |
| TAXNUM | Tax identification number |
| ACCOUNTNUM | Account number |

Supported languages: English, Spanish, French, German, Italian, Dutch.

## Important: Test Data vs Real Data

The Piiranha model is context-aware — it can distinguish between real PII and obvious test/placeholder data. For example, `john.doe@example.com` or `123 Main Street` may score lower or not be flagged at all, because the model recognizes them as synthetic. This means:

- **Don't use fake data to evaluate detection accuracy.** Use realistic (but consented) data for testing.
- **In production, this is a feature.** The model won't over-redact boilerplate or template text, reducing false positives.
- **Lower your threshold** (`threshold=0.1`) if you want to catch everything regardless of context.

## License

The code in this repository is MIT licensed.

The underlying model ([Piiranha v1](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)) is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). This means the model weights **may not be used for commercial purposes**. By using this library you agree to the model's license terms.
