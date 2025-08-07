import logging
from src.utils.path import get_project_root

logger = logging.getLogger(__name__)
root_dir = get_project_root()

def get_conversation_history() -> str:
    """Retrieve the conversation history from the file."""
    history_file = root_dir / "conversation_history.txt"
    if not history_file.exists():
        logger.warning("Conversation history file does not exist.")
        return ""
    with open(history_file, "r") as file:
        return file.read()
 
def write_conversation_history(history: str):
    """Write the conversation history to a file."""
    history_file = root_dir / "conversation_history.txt"
    try:
        with open(history_file, "w") as file:
            file.write(history)
        logger.info("Conversation history updated successfully.")
    except Exception as e:
    
        logger.error(f"Failed to write conversation history: {e}")
        return

def get_prompt() -> str:
    try:
     with open(root_dir/"system_prompt.txt", "r") as file:
        system_prompt = file.read()
     return system_prompt
    except FileNotFoundError:
        logger.error("System prompt file not found.")
        return "You are a helpful assistant. Please answer the user's questions based on the provided context."
    except Exception as e:
        logger.error(f"An error occurred while reading the system prompt: {e}")
        return "You are a helpful assistant. Please answer the user's questions based on the provided context."
    