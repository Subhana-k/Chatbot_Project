from langchain import hub
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from chatbot.gemini import llm, system_message
from chatbot.chatbot_tavily import tavily_tool
from chatbot.database import save_message

# Load Prompt Template
prompt_template = hub.pull("rlm/rag-prompt")

def process_user_message(message, chat_id, history):
    """Process user message with RAG and return assistant response."""
    # Save user message
    save_message(chat_id, "user", message)

    # Tavily web search
    search_results = tavily_tool.invoke(message)
    context = "\n\n".join(
        f"{i+1}. {result['title']}\n{result['content']}\nURL: {result['url']}"
        for i, result in enumerate(search_results["results"])
    )

    # Build prompt
    final_prompt = prompt_template.invoke({
        "question": message,
        "context": context
    }).to_string()

    # Prepare chat history for Gemini
    history_langchain_format = [SystemMessage(content=system_message)]
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"])) # format gemini understands 
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))

    # Stream Gemini response
    return llm.stream(history_langchain_format + [HumanMessage(content=final_prompt)]) # passes chat history and Rag prompt to gemini
