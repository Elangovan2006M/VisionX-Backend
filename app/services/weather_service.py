import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

async def get_weather(location: str):
    if not OPENWEATHER_API_KEY:
        return {"error": "API key missing."}

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        res = requests.get(url)
        data = res.json()
    except Exception as e:
        return {"error": f"Could not parse response: {str(e)}"}

    if res.status_code != 200:
        return {"error": data.get("message", "Unknown error"), "status_code": res.status_code}

    # Return all details from the API response
    return data
