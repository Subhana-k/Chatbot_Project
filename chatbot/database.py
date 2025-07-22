import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Setup
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("❌ Missing MONGO_URI in .env file.")

mongo_client = MongoClient(mongo_uri)
db = mongo_client["chat_app"]
chats_collection = db["chats"]

# Helpers
def save_message(chat_id, role, content):
    """Save chat message to MongoDB."""
    chat = chats_collection.find_one({"_id": chat_id})
    if chat and chat.get("messages") and chat["messages"][-1] == {"role": role, "content": content}:
        return  # Avoid duplicate
    chats_collection.update_one(
        {"_id": chat_id},
        {"$push": {"messages": {"role": role, "content": content}}},
        upsert=True
    )

def get_chat_history(chat_id):
    """Fetch chat history by chat_id."""
    chat = chats_collection.find_one({"_id": chat_id})
    return chat["messages"] if chat and "messages" in chat else []

def list_all_chats():
    """List all saved chats with summaries."""
    chats = []
    for chat in chats_collection.find():
        if "messages" in chat and len(chat["messages"]) > 0:
            first_user_msg = next(
                (msg["content"] for msg in chat["messages"] if msg["role"] == "user"),
                "(Untitled Chat)"
            )
            summary = first_user_msg.strip()[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
            chats.append((summary, chat["_id"]))
    return chats
