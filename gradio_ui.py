import uuid
import time
import gradio as gr
import os
import PyPDF2
from chatbot.database import list_all_chats, get_chat_history, save_message
from chatbot.chatbot_rag import llm, prompt_template, tavily_tool, system_message
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load resume text from chatbot folder
resume_path = os.path.join(os.path.dirname(__file__), "Resume.pdf")
resume_text = ""
try:
    with open(resume_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text + "\n"
except FileNotFoundError:
    print("❌ Resume.pdf not found in chatbot folder.")
    resume_text = "(Resume not found)"

def stream_response(message, chat_id, history):
    # If chat_id is None, create a new one
    if not chat_id:
        chat_id = str(uuid.uuid4())
        print(f"🔄 New Chat Created: {chat_id}")

    print(f"\n📝 User Prompt: {message} (Chat ID: {chat_id})")

    # Save user message to MongoDB
    save_message(chat_id, "user", message)

    # Show user message instantly in the UI
    partial_history = history + [{"role": "user", "content": message}]
    assistant_partial_response = ""
    updated_history = partial_history + [{"role": "assistant", "content": ""}]
    yield updated_history, "", gr.update(choices=list_all_chats(), value=chat_id)

    # Fetch full chat history from MongoDB
    full_chat_history = get_chat_history(chat_id)

    # Determine if this is a resume-related question
    resume_keywords = ["resume", "cv", "work experience", "education", "skills", "projects", "job", "career"]
    is_resume_question = any(keyword in message.lower() for keyword in resume_keywords)

    # Build combined context
    combined_context = ""
    if is_resume_question:
        combined_context += f"This is my resume:\n{resume_text}\n\n"
    combined_context += "Conversation so far:\n"
    for msg in full_chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        combined_context += f"{role}: {msg['content']}\n"
    combined_context += f"User: {message}\nAssistant:"

    # Only perform Tavily search if not resume-related
    tavily_context = ""
    if not is_resume_question:
        print("🌐 Performing Tavily web search...")
        search_results = tavily_tool.invoke(message)
        tavily_context = "\n\n".join(
            f"{i+1}. {result['title']}\n{result['content']}\nURL: {result['url']}"
            for i, result in enumerate(search_results["results"])
        )

    # Create prompt
    final_prompt = f"{combined_context}\n\nAdditional Context:\n{tavily_context}"

    # Get Gemini response
    response = llm.invoke([HumanMessage(content=final_prompt)])
    full_response = response.content

    # Simulate typing effect
    words = full_response.split()
    for i in range(1, len(words) + 1):
        assistant_partial_response = " ".join(words[:i])
        updated_history[-1]["content"] = assistant_partial_response
        yield updated_history, "", gr.update(...)
        time.sleep(0.1)  # Adjust speed for words


    # Save assistant response to MongoDB
    save_message(chat_id, "assistant", full_response)
    print(f"🤖 Gemini Response: {full_response}\n")

def launch_app():
    """Launch Gradio UI for Gemini Chatbot"""
    with gr.Blocks() as demo_interface:
        chat_id_state = gr.State(str(uuid.uuid4()))

        with gr.Row():
            with gr.Column(scale=1):
                new_chat_btn = gr.Button("➕ New Chat")
                saved_chats = gr.Dropdown(
                    label="📂 Saved Chats",
                    choices=list_all_chats(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True
                )
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Gemini + Tavily RAG", type="messages")
                msg_box = gr.Textbox(
                    placeholder="Send a message to Gemini LLM...",
                    container=False,
                    autoscroll=True,
                    scale=7
                )

        # Logic for new chat
        def start_new_chat(current_chat_id, current_history):
            new_id = str(uuid.uuid4())
            print(f"🔄 New Chat Created: {new_id}")
            return (new_id, [], gr.update(choices=list_all_chats(), value=None, interactive=True))

        new_chat_btn.click(
            fn=start_new_chat,
            inputs=[chat_id_state, chatbot_ui],
            outputs=[chat_id_state, chatbot_ui, saved_chats]
        )

        # Load selected chat
        saved_chats.change(
            fn=lambda cid: (get_chat_history(cid), cid),
            inputs=saved_chats,
            outputs=[chatbot_ui, chat_id_state]
        )

        # Send message
        msg_box.submit(
            fn=stream_response,
            inputs=[msg_box, chat_id_state, chatbot_ui],
            outputs=[chatbot_ui, msg_box, saved_chats]
        )

    demo_interface.launch(debug=True, share=True)
