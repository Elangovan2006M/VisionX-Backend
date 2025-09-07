import requests
from bs4 import BeautifulSoup

KERALA_AGRI_URL = "https://minister-agriculture.kerala.gov.in/"

def scrape_kerala_agri():
    """
    Scrape Kerala Agriculture Ministry website and return cleaned text content.
    Only the first 2000 chars are returned for LLM context to avoid overload.
    """
    try:
        response = requests.get(KERALA_AGRI_URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text from <p> tags
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        # Limit to avoid too large payload
        return text[:2000] if text else "No content found."
    except Exception as e:
        return f"Error scraping Kerala Agriculture site: {str(e)}"
