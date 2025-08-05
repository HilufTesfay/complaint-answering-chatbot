import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import gc
import numpy as np


# Initialize logging
def setup_logging(log_dir: str = "../logs") -> logging.Logger:
    """Configure structured logging for EDA tasks"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eda.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Pre-compile regex patterns for performance
TEXT_CLEANING_PATTERNS = {
    "boilerplate": [
        r'i\s+(?:am|have)\s+writing\s+to\s+(?:file|submit)\s+a\s+complaint',
        r'dear\s+(?:sir|madam|team)',
        r'this\s+is\s+(?:regarding|about)',
        r'(?:please|kindly)\s+(?:help|assist)'
    ],
    "special_chars": re.compile(r'[^\w\s]'),
    "extra_spaces": re.compile(r'\s+')
}

def clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).lower().strip()
    
    # Apply each boilerplate pattern separately
    for pattern in TEXT_CLEANING_PATTERNS['boilerplate']:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = TEXT_CLEANING_PATTERNS['special_chars'].sub(' ', text)
    text = TEXT_CLEANING_PATTERNS['extra_spaces'].sub(' ', text)
    return text.strip()

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Robust data loader with validation and memory optimization"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        dtypes = {
            "Product": "category",
            "Consumer complaint narrative": "string",
            "Complaint ID": "string",
            "State": "category",
        }

        
        df = pd.read_csv(
            filepath,
            dtype=dtypes,
            usecols=list(dtypes.keys()),  
        )

        # Validation
        if len(df) == 0:
            raise ValueError("Empty dataframe loaded")

        required_cols = ["Product", "Consumer complaint narrative"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise


def perform_initial_eda(df: pd.DataFrame) -> None:
    """Comprehensive EDA with enhanced visualizations"""
    try:
        # 1. Product Distribution
        plt.figure(figsize=(12, 8))
        product_dist = df["Product"].value_counts(normalize=True) * 100
        ax = product_dist.plot(kind="barh")
        plt.title("Complaint Distribution by Product", pad=20)
        plt.xlabel("Percentage (%)")

        # Annotate bars with percentages
        for i, v in enumerate(product_dist):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        plt.tight_layout()
        plt.savefig("../logs/product_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Narrative Length Analysis
        df["narrative_length"] = (
            df["Consumer complaint narrative"].str.split().str.len()
        )

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x="Product",
            y="narrative_length",
            data=df,
            showfliers=False,  # Exclude outliers for better visualization
        )
        plt.title("Complaint Length Distribution by Product", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Word Count")
        plt.tight_layout()
        plt.savefig("../logs/narrative_length_distribution.png", dpi=300)
        plt.close()

        # 3. Missing Data Analysis
        missing_stats = df.isnull().mean() * 100
        logger.info(f"Missing data percentages:\n{missing_stats.to_string()}")

    except Exception as e:
        logger.error(f"EDA failed: {str(e)}", exc_info=True)
        raise


def filter_and_clean_data(
    df: pd.DataFrame,
    target_products: List[str],
    chunksize: Optional[int] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Memory-optimized data processing pipeline

    Args:
        df: Input DataFrame
        target_products: List of products to include
        chunksize: Process in chunks if provided (recommended >1M rows)
        output_file: If provided, saves chunks directly to disk

    Returns:
        Cleaned DataFrame
    """
    try:
        # Validate target products
        valid_products = set(df["Product"].unique())
        invalid_products = set(target_products) - valid_products
        if invalid_products:
            logger.warning(f"Invalid products specified: {invalid_products}")

        # Prepare for chunked processing
        if chunksize and len(df) > chunksize:
            logger.info(f"Processing in chunks of {chunksize:,} rows")

            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(exist_ok=True)
                # Write header
                pd.DataFrame(columns=df.columns).to_csv(output_path, index=False)

            processed_chunks = []

            for i in tqdm(range(0, len(df), chunksize), desc="Processing chunks"):
                chunk = df.iloc[i : i + chunksize].copy()

                # Filter and clean
                chunk = chunk[chunk["Product"].isin(target_products)]
                chunk = chunk[chunk["Consumer complaint narrative"].notna()]
                chunk["cleaned_narrative"] = chunk[
                    "Consumer complaint narrative"
                ].apply(clean_text)

                if output_file:
                    chunk.to_csv(output_file, mode="a", header=False, index=False)
                else:
                    processed_chunks.append(chunk)

                del chunk
                gc.collect()

            result = (
                pd.concat(processed_chunks, ignore_index=True)
                if not output_file
                else pd.read_csv(output_file)
            )
        else:
            # Single-pass processing
            logger.info("Processing in single pass")
            result = df[df["Product"].isin(target_products)].copy()
            result = result[result["Consumer complaint narrative"].notna()]
            result["cleaned_narrative"] = result["Consumer complaint narrative"].apply(
                clean_text
            )

        logger.info(f"Final cleaned data shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}", exc_info=True)
        raise


def prepare_rag_data(
    filepath: str, chunksize: Optional[int] = None, output_file: Optional[str] = None
) -> pd.DataFrame:
    """End-to-end data preparation pipeline"""
    target_products = [
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later",
        "Savings account",
        "Money transfers",
    ]

    try:
        logger.info("Starting RAG data preparation")

        # 1. Load and validate
        df = load_and_validate_data(filepath)

        # 2. Perform EDA
        perform_initial_eda(df)

        # 3. Filter and clean
        cleaned_data = filter_and_clean_data(
            df=df,
            target_products=target_products,
            chunksize=chunksize,
            output_file=output_file,
        )

        # 4. Save memory by removing intermediate objects
        del df
        gc.collect()

        logger.info("RAG data preparation completed successfully")
        return cleaned_data

    except Exception as e:
        logger.error(f"RAG preparation pipeline failed: {str(e)}", exc_info=True)
        raise
