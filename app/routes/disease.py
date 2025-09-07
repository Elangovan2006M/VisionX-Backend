from fastapi import APIRouter, UploadFile, File
from app.services.disease_service import analyze_plant_image

router = APIRouter()

@router.post("/disease")
async def disease(file: UploadFile = File(...)):
    try:
        result = await analyze_plant_image(file)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
