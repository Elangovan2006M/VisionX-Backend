from fastapi import APIRouter
from app.services.weather_service import get_weather

router = APIRouter()

@router.get("/")
async def fetch_weather(location: str = "Kerala"):
    return {"result": await get_weather(location)}
