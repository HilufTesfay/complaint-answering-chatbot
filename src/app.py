from typing import List, Dict
import logging
import gradio as gr
from src.rag import retrieve_documents,generate_answer
from src.utils.logger import setup_logging
#set up logging
setup_logging("main")
logger= logging.getLogger(__name__)

def chat_ui(user_question:str,history:list)->str:
    """chat interface for user to ask questions and get answers."""
    try:
        final_answer = generate_answer(user_question)
        return final_answer
    except Exception as e:
        logger.error(f"Error in chat_ui: {e}")
        return "An error occurred while processing your request."

app=gr.ChatInterface(
    fn=chat_ui,
    title="RAG Chatbot",
    description="Ask questions and get answers based on the provided context.",
    examples=["What is the most common product?", "How many complaints are there?"],
    theme="soft")


if __name__ == "__main__":
    logger.info("Starting complaint Chatbot")
    app.launch()