import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List
import logging
from pathlib import Path

# Initialize logging
def setup_logging(log_dir: str = "../logs") -> logging.Logger:
    """Configure logging for EDA tasks"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eda.log"
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load and validate the complaint dataset"""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Check required columns exist
        required_cols = ['Product', 'Consumer complaint narrative']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def perform_initial_eda(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis on complaint data"""
    # Product distribution
    plt.figure(figsize=(10, 6))
    product_dist = df['Product'].value_counts()
    product_dist.plot(kind='bar')
    plt.title('Complaint Distribution by Product')
    plt.savefig('../logs/product_distribution.png')
    logger.info(f"Product distribution:\n{product_dist}")
    
    # Narrative length analysis
    df['narrative_length'] = df['Consumer complaint narrative'].str.split().str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['narrative_length'], bins=50)
    plt.title('Distribution of Complaint Narrative Lengths')
    plt.savefig('../logs/narrative_lengths.png')
    
    short_threshold = 10
    long_threshold = 500
    logger.info(f"Narratives < {short_threshold} words: {(df['narrative_length'] < short_threshold).sum()}")
    logger.info(f"Narratives > {long_threshold} words: {(df['narrative_length'] > long_threshold).sum()}")
    
    # Missing narratives
    missing_narratives = df['Consumer complaint narrative'].isna().sum()
    logger.info(f"Complaints without narratives: {missing_narratives}")

def filter_and_clean_data(df: pd.DataFrame, target_products: List[str]) -> pd.DataFrame:
    """Filter and clean complaint data for RAG pipeline"""
    # Filter products
    filtered = df[df['Product'].isin(target_products)].copy()
    logger.info(f"Data after product filtering: {filtered.shape}")
    
    # Remove empty narratives
    filtered = filtered[filtered['Consumer complaint narrative'].notna()]
    logger.info(f"Data after removing empty narratives: {filtered.shape}")
    
    # Text cleaning
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
        text = re.sub(r'\b(i am writing to file a complaint|dear sir/madam)\b', '', text)
        return text.strip()
    
    filtered['cleaned_narrative'] = filtered['Consumer complaint narrative'].apply(clean_text)
    logger.info("Text cleaning completed")
    
    return filtered

def prepare_rag_data(filepath: str) -> pd.DataFrame:
    """End-to-end data preparation for RAG pipeline"""
    target_products = [
        'Credit card', 
        'Personal loan', 
        'Buy Now, Pay Later', 
        'Savings account', 
        'Money transfers'
    ]
    
    df = load_and_validate_data(filepath)
    perform_initial_eda(df)
    return filter_and_clean_data(df, target_products)

