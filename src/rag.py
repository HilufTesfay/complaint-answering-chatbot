import os
import logging
import google.generativeai as genai
from typing import List, Dict
from langchain.schema import Document
from dotenv import load_dotenv
from src.utils.helper import get_conversation_history, get_prompt,write_conversation_history
from src.vector_db.chrom import ChromaVectorStore
from src.utils.path import get_project_root


logger=logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model= genai.Model("gemini-1.5-flash-001")
root_dir = get_project_root()
vector_store=ChromaVectorStore()


system_prompt = get_prompt()
conversation_history = get_conversation_history()

def  retrieve_documents(user_question: str, top_k: int = 5) -> List[Document]:
    """Retrieve documents from ChromaDB based on user question."""
    results = vector_store.query_collection(query=user_question, n_results=top_k)
    return results

def generate_answer(user_question: str, top_k: int = 5) -> str:
    """Generate an answer based on the retrieved documents."""
    write_conversation_history(conversation_history + f"\nUser: {user_question}\n")
    documents = retrieve_documents(user_question, top_k)
    if not documents:
        return "No relevant documents found from the embedding vector."

    context = "\n".join([doc.page_content for doc in documents])
    full_prompt= f"{system_prompt}\n\nUser question: {user_question}\n\nContext:{context}"
    if conversation_history:
        full_prompt += f"\n\nConversation History:\n{conversation_history}\n"
    
    try:
        logger.info(f"Generating answer for question: {user_question}")
        answer = model.generate(prompt=full_prompt)
        final_answer = answer.text.strip() if answer else "Unable to generate an answer."
        write_conversation_history(conversation_history + f"Assistant: {final_answer}\n")
        return final_answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."