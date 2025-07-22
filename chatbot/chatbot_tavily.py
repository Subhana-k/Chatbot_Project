import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("❌ Missing TAVILY_API_KEY in .env file.")

# Tavily Search Tool Setup
tavily_tool = TavilySearch(api_key=tavily_api_key)
