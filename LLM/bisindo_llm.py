import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "bisindo-gemma1b"
DEFAULT_TIMEOUT = 60
DEFAULT_KEEP_ALIVE = "10m"


def clean_output(text: str) -> str:
    result = text.strip()

    bad_prefixes = [
        "Output:",
        "Kalimat:",
        "Kalimat Indonesia:",
        "Jawaban:",
        "Hasil:",
    ]

    for prefix in bad_prefixes:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()

    result = result.splitlines()[0].strip()
    return result


def build_prompt(tokens, allow_word_correction: bool = False) -> str:
    if isinstance(tokens, list):
        token_text = " ".join(tokens)
    else:
        token_text = str(tokens).strip()

    correction_rule = (
        "Mode: boleh memperbaiki token yang jelas salah/tidak nyambung tanpa menambah informasi baru.\n"
        if allow_word_correction
        else "Mode: pertahankan token inti; hanya rapikan struktur menjadi Bahasa Indonesia tulis.\n"
    )
    return f"""{correction_rule}Token BISINDO: {token_text}
Output:"""


def gloss_to_sentence(
    tokens,
    *,
    model: str | None = None,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    allow_word_correction: bool = False,
):
    prompt = build_prompt(tokens, allow_word_correction=allow_word_correction)

    payload = {
        "model": model or MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.0,
            "num_ctx": 512,
            "num_predict": 40,
            "stop": [
                "\nToken BISINDO:",
                "\nContoh:",
                "\nOutput:",
            ],
        },
    }

    response = requests.post(url or OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()

    return clean_output(response.json()["response"])
