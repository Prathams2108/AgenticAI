import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def dish_image_search(dish: str) -> list:
    """
    Search and return image URLs for a given dish name using Unsplash.
    """

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": dish,
        "client_id": os.getenv("UNSPLASH_API_KEY"),
        "per_page": 3
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    return [
        img["urls"]["regular"]
        for img in data.get("results", [])
    ]

# Expose tool for agent
image_search_tool = dish_image_search
