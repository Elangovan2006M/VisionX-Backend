import re
import os
import difflib
import requests
from fastapi import UploadFile
from app.services.disease_service import analyze_plant_image
from app.routes.weather import get_weather
from app.services.scraper_service import scrape_kerala_agri
from app.services.translate_service import translate_text, simple_detect_language
import asyncio


KERALA_CITIES = [
    "Thiruvananthapuram", "Kozhikode", "Kochi", "Kollam", "Thrissur", "Kannur", "Alappuzha", "Kottayam", "Palakkad",
    "Manjeri", "Thalassery", "Thrippunithura", "Ponnani", "Vatakara", "Kanhangad", "Payyanur", "Koyilandy",
    "Parappanangadi", "Kalamassery", "Kodungallur", "Neyyattinkara", "Tanur", "Kayamkulam", "Malappuram",
    "Guruvayur", "Thrikkakkara", "Irinjalakuda", "Wadakkancherry", "Nedumangad", "Kondotty", "Tirurangadi",
    "Tirur", "Panoor", "Kasaragod", "Feroke", "Kunnamkulam", "Ottappalam", "Thiruvalla", "Thodupuzha",
    "Perinthalmanna", "Karunagappalli", "Chalakudy", "Payyoli", "Koduvally", "Mananthavady", "Changanassery",
    "Mattanur", "Punalur", "Nilambur", "Cherthala", "Sultan Bathery", "Maradu", "Kottakkal", "Taliparamba",
    "Shornur", "Pandalam", "Kattappana", "Cherpulassery", "Mukkam", "Iritty", "Valanchery", "Varkala",
    "Nileshwaram", "Chavakkad", "Kothamangalam", "Pathanamthitta", "Attingal", "Paravur", "Ramanattukara",
    "Mannarkkad", "Erattupetta", "Sreekandapuram", "Angamaly", "Chittur-Thathamangalam", "Kalpetta",
    "North Paravur", "Haripad", "Muvattupuzha", "Kottarakara", "Kuthuparamba", "Adoor", "Pattambi", "Anthoor",
    "Perumbavoor", "Piravom", "Ettumanoor", "Mavelikkara", "Eloor", "Chengannur", "Vaikom", "Aluva", "Pala",
    "Koothattukulam"
]

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


# Query LLM
async def query_llm(messages: list, model: str = None, retries: int = 2) -> str:
    payload = {"model": model or OPENROUTER_MODEL, "messages": messages}
    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=HEADERS, json=payload, timeout=20
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == retries - 1:
                return "Sorry, I couldn’t process the request right now. Please try again."
    return "Unexpected error."


# Extract city name
async def extract_city_from_query(query: str) -> str:
    words = re.findall(r'\w+', query)
    matches = difflib.get_close_matches(' '.join(words), KERALA_CITIES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    for word in words:
        matches = difflib.get_close_matches(word, KERALA_CITIES, n=1, cutoff=0.7)
        if matches:
            return matches[0]
    return "Kerala"


async def handle_farmer_query(query: str = None, image: UploadFile = None):
    response_data = None

    # 📝 Detect user language
    if query:
        user_lang = simple_detect_language(query)
        # Translate query if Malayalam → English (LLM works best in English internally)
        query_en = await translate_text(query, "English") if user_lang == "Malayalam" else query
    else:
        user_lang = "English"
        query_en = None

    # 🌦 Keywords for weather queries
    weather_keywords = [
        "weather", "climate", "temperature", "rain", "humidity", "forecast", "wind",
        "monsoon", "storm", "sunny", "cloudy", "rainfall", "precipitation", "heat",
        "cold", "fog", "dew", "season", "drizzle", "thunder", "lightning", "cyclone",
        "flood", "drought"
    ]

    # 🌦 Weather query branch
    if query_en and any(word in query_en.lower() for word in weather_keywords):
        city = await extract_city_from_query(query_en)
        weather_data = await get_weather(city)

        weather_prompt = [
            {"role": "system", "content": f"You are an agricultural expert. Reply in {user_lang} using farmer-friendly language."},
            {"role": "user", "content": f"Farmer asked: {query_en}. Weather data for {city}: {weather_data}. Provide simple advice."}
        ]
        llm_answer = await query_llm(weather_prompt)

        response_data = {"type": "weather", "city": city, "answer": llm_answer}

    # 🌱 Disease detection branch
    elif image:
        analysis = await analyze_plant_image(image)
        await image.close()

        predicted_class = analysis.get("predicted_class", "Unknown crop")
        confidence = analysis.get("confidence", 0.0)
        description = analysis.get("description", "An image of a plant.")

        context = f"Plant description: {description}. Predicted class: {predicted_class} ({confidence}%)."

        disease_prompt = [
            {"role": "system",
             "content": f"You are an agricultural expert. Reply in {user_lang}. "
                        "Explain clearly in farmer-friendly language with remedies and prevention steps."},
            {"role": "user", "content": f"{context} Farmer asked: '{query_en or query}'"}
        ]
        llm_answer = await query_llm(disease_prompt)

        response_data = {
            "type": "disease",
            "crop": predicted_class,
            "description": description,
            "answer": llm_answer
        }

    # 🏛 Government schemes branch
    elif query_en and any(word in query_en.lower() for word in ["scheme", "subsidy", "government", "policy", "loan", "insurance"]):
        gov_content = scrape_kerala_agri()

        gov_prompt = [
            {"role": "system",
             "content": f"You are an agricultural assistant specialized in Kerala government schemes. Reply in {user_lang}."},
            {"role": "user", "content": f"Farmer asked: {query_en}. Scraped content: {gov_content}. Summarize clearly."}
        ]
        llm_answer = await query_llm(gov_prompt)

        response_data = {"type": "government", "source": "kerala_agri_site", "answer": llm_answer}

    # 🌾 General farming query branch
    elif query_en:
        general_prompt = [
            {"role": "system",
             "content": f"You are a helpful farming assistant. Reply in {user_lang}. "
                        "ONLY answer questions related to farming, crops, soil, pests, fertilizers, weather, irrigation."},
            {"role": "user", "content": query_en}
        ]
        llm_answer = await query_llm(general_prompt)

        response_data = {"type": "general", "answer": llm_answer}

    else:
        response_data = {"error": "No valid input provided"}

    return response_data
