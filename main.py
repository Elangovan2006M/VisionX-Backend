from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Import your routes
from app.routes import llm, weather, disease, translate

# Initialize FastAPI app
app = FastAPI(title="VisionX Backend", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(llm.router, prefix="/llm", tags=["LLM"])
app.include_router(weather.router, prefix="/weather", tags=["Weather"])
app.include_router(disease.router, prefix="/disease", tags=["Disease"])
app.include_router(translate.router, prefix="/translate", tags=["Translate"])

@app.get("/")
async def root():
    return {"message": "VisionX Backend is running!"}

# Run Uvicorn if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or default 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
