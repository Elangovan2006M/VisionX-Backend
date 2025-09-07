# app/services/llm_service.py
import os
import requests
from fastapi import UploadFile
from app.services.disease_service import analyze_plant_image
from app.routes.weather import get_weather
from app.services.scraper_service import scrape_kerala_agri
from app.services.translate_service import translate_text, detect_language   # 👈 NEW

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
    prompt = f"""
    Extract the city name from this user query: "{query}".
    If no city is mentioned, return only 'Kerala'.
    Return ONLY the city name, nothing else.
    """
    city = await query_llm([{"role": "user", "content": prompt}])
    return city.strip()


# Main handler
async def handle_farmer_query(query: str = None, image: UploadFile = None):
    response_data = None

    # 🌐 Detect user language
    user_lang = "English"
    if query:
        user_lang = await detect_language(query)   # e.g., "Tamil", "Hindi", "English"
        query = await translate_text(query, user_lang, "English")  # 👈 always send English to LLM

    weather_keywords = ["weather", "climate", "temperature", "rain", "humidity", "forecast", "wind"]
    if query and any(word in query.lower() for word in weather_keywords):
        city = await extract_city_from_query(query)
        weather_data = await get_weather(city)
        weather_prompt = [
            {"role": "system",
             "content": "You are an agricultural expert. Explain in simple farmer-friendly language."},
            {"role": "user",
             "content": f"Farmer asked: {query}. Weather data for {city}: {weather_data}. Provide simple advice."}
        ]
        llm_answer = await query_llm(weather_prompt)
        llm_answer = await translate_text(llm_answer, "English", user_lang)  # 👈 back to user lang
        response_data = {"type": "weather", "city": city, "answer": llm_answer}

    elif image:
        try:
            analysis = await analyze_plant_image(image)
            await image.close()

            predicted_class = analysis.get("predicted_class", "Unknown crop")
            confidence = analysis.get("confidence", 0.0)
            description = analysis.get("description", "An image of a plant.")

            context = f"Plant description: {description}. Predicted class: {predicted_class} ({confidence}%)."

            disease_prompt = [
                {"role": "system",
                 "content": "You are an agricultural expert. Explain clearly in farmer-friendly language, "
                            "with remedies and prevention steps if needed."},
                {"role": "user",
                 "content": f"{context} Farmer asked: '{query}'"}
            ]
            llm_answer = await query_llm(disease_prompt)
            llm_answer = await translate_text(llm_answer, "English", user_lang)  # 👈 back translate
            response_data = {
                "type": "disease",
                "crop": predicted_class,
                "description": description,
                "answer": llm_answer
            }

        except Exception as e:
            response_data = {"type": "disease", "error": str(e)}

    elif query and any(word in query.lower() for word in ["scheme", "subsidy", "government", "policy", "loan", "insurance"]):
        gov_content = scrape_kerala_agri()
        gov_prompt = [
            {"role": "system",
             "content": "You are an agricultural assistant specialized in Kerala government schemes."},
            {"role": "user",
             "content": f"Farmer asked: {query}. Scraped content: {gov_content}. Summarize clearly."}
        ]
        llm_answer = await query_llm(gov_prompt)
        llm_answer = await translate_text(llm_answer, "English", user_lang)  # 👈 back translate
        response_data = {"type": "government", "source": "kerala_agri_site", "answer": llm_answer}

    elif query:
        general_prompt = [
            {"role": "system",
             "content": "You are a helpful farming assistant. ONLY answer questions related to farming, crops, soil, pests, fertilizers, weather, irrigation."},
            {"role": "user", "content": query}
        ]
        llm_answer = await query_llm(general_prompt)
        llm_answer = await translate_text(llm_answer, "English", user_lang)  # 👈 back translate
        response_data = {"type": "general", "answer": llm_answer}

    else:
        response_data = {"error": "No valid input provided"}

    return response_data
