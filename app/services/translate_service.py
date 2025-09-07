# app/services/translate_service.py
import asyncio
import os
from gradio_client import Client
import requests

# Hugging Face Space config
TRANSLATE_SPACE = "UNESCO/nllb"
TRANSLATE_API = "/translate"

client = Client(TRANSLATE_SPACE)

# OpenRouter config for language detection
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


async def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text using UNESCO/nllb Hugging Face Space.
    - text: input string
    - src_lang: language name from space dropdown (e.g. "Hindi")
    - tgt_lang: language name (e.g. "English")
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: client.predict(
                text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                api_name=TRANSLATE_API,
            ),
        )
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        print("Translation failed:", e)
        return text  # fallback: return original text


async def detect_language(text: str) -> str:
    """
    Detects the language of the given text using OpenRouter LLM.
    Returns the language name exactly as expected by UNESCO/nllb (first letter capitalized).
    """
    if not text.strip():
        return "English"

    prompt = f"""
    Detect the language of this text: "{text}".
    Respond with ONLY the language name (e.g., 'Hindi', 'Tamil', 'English').
    """

    try:
        payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        language = data["choices"][0]["message"]["content"].strip()
        # Ensure proper capitalization (UNESCO space expects first letter capitalized)
        return language.capitalize()
    except Exception as e:
        print("Language detection failed:", e)
        return "English"
