from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

from app.routes import llm, weather, disease, translate

app = FastAPI(title="Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(llm.router, prefix="/llm", tags=["Llm"])
app.include_router(weather.router, prefix="/weather", tags=["Weather"])
app.include_router(disease.router, prefix="/disease", tags=["Disease"])
app.include_router(translate.router, prefix="/translate", tags=["Translate"])


@app.get("/")
async def root():
    return {"message": "VisionX Backend is running!"}
