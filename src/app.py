import logging
import gradio as gr
from src.rag import retrieve_documents,generate_answer
 
#set up logging
logger= logging.getLogger(__name__)

def chat_ui(user_question:str)->str:
    """chat interface"""
    try:
        document = retrieve_documents(user_question)
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
    theme="default",
    allow_flagging="never",)


if __name__ == "__main__":
    logger.INFO("Starting RAG Chatbot")
    app.launch()