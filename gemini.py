import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env file.")

# Gemini LLM Setup
llm = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.5-flash",
    model_kwargs={"streaming": True}
)

system_message = "You act like a helpful AI assistant."
