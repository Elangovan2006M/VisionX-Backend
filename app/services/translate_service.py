import asyncio
from gradio_client import Client

# Hugging Face Space config (UNESCO/nllb translator)
TRANSLATE_SPACE = "UNESCO/nllb"
TRANSLATE_API = "/translate"

client = Client(TRANSLATE_SPACE)


async def translate_text(text: str, tgt_lang: str) -> str:
    """
    Translate text between English and Malayalam using UNESCO/nllb Hugging Face Space.
    - text: input string
    - tgt_lang: target language ("English" or "Malayalam")
    - Source language is auto-detected based on target (only 2 languages supported)
    """
    if not text.strip():
        return text

    # Decide source automatically
    src_lang = "Malayalam" if tgt_lang == "English" else "English"

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
        return text  # fallback


def simple_detect_language(text: str) -> str:
    """
    Very lightweight detection: 
    - If Malayalam Unicode range is found → 'Malayalam'
    - Else default to 'English'
    """
    if any("\u0D00" <= ch <= "\u0D7F" for ch in text):
        return "Malayalam"
    return "English"
