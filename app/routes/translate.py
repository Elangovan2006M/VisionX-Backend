from fastapi import APIRouter, Form
from app.services.translate_service import translate_text, simple_detect_language

router = APIRouter()

@router.post("/translate")
async def translate_api(
    text: str = Form(...),
    tgt_lang: str = Form(...)
):
    """
    Translate text between English and Malayalam using UNESCO/nllb.
    Automatically detects source language.
    """
    # Detect source automatically
    src_lang = simple_detect_language(text)

    # If same language, just return text
    if src_lang == tgt_lang:
        return {
            "input_text": text,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "translated_text": text
        }

    # ✅ Now only pass text + tgt_lang
    translated = await translate_text(text, tgt_lang)

    return {
        "input_text": text,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "translated_text": translated
    }
