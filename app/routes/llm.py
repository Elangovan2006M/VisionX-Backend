from fastapi import APIRouter, UploadFile, Form
from app.services.llm_service import handle_farmer_query

router = APIRouter()

@router.post("/ask")
async def ask_farming_assistant(
    query: str = Form(...),
    image: UploadFile = None
):
    """
    Entry point for farmer queries.
    - Weather queries -> auto-detect city
    - Image queries -> LLM vision + classifier fallback
    - General queries -> LLM
    """
    result = await handle_farmer_query(query, image=image)
    return result
