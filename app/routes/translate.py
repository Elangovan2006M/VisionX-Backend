# app/routes/translate.py
from fastapi import APIRouter, Form
from app.services.translate_service import translate_text

router = APIRouter()

@router.post("/translate")
async def translate_api(
    text: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...)
):
    """
    Translate any text using UNESCO/nllb Hugging Face Space.
    """
    translated = await translate_text(text, src_lang, tgt_lang)
    return {
        "input_text": text,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "translated_text": translated
    }
