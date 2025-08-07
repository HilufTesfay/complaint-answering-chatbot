import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import gc
from typing import Dict
from src.utils.path import get_project_root
# Initialize logger
logger = logging.getLogger(__name__)

# Set up the root directory for the project
root_dir = get_project_root()

# Pre-compile regex patterns for performance
TEXT_CLEANING_PATTERNS = {
    "boilerplate": [
        r"i\s+(?:am|have)\s+writing\s+to\s+(?:file|submit)\s+a\s+complaint",
        r"dear\s+(?:sir|madam|team)",
        r"this\s+is\s+(?:regarding|about)",
        r"(?:please|kindly)\s+(?:help|assist)",
    ],
    "special_chars": re.compile(r"[^\w\s]"),
    "extra_spaces": re.compile(r"\s+"),
}


def clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = str(text).lower().strip()

    # Apply each boilerplate pattern separately
    for pattern in TEXT_CLEANING_PATTERNS["boilerplate"]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = TEXT_CLEANING_PATTERNS["s" \
    "pecial_chars"].sub(" ", text)
    text = TEXT_CLEANING_PATTERNS["extra_spaces"].sub(" ", text)
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


def perform_initial_eda(df: pd.DataFrame, chunk_size: int = 1000000) -> None:
    """Comprehensive EDA with chunking for large datasets"""
    try:
        # Initialize aggregators for chunked processing
        product_counts: Dict[str, int] = {}
        missing_counts: Dict[str, int] = {}
        total_rows = len(df)
        narrative_lengths = []

        # Validate required columns
        required_columns = ["Product", "Consumer complaint narrative"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in DataFrame: {missing_columns}")
            raise ValueError(f"Required columns missing: {missing_columns}")

        # Process DataFrame in chunks
        for start_idx in range(0, total_rows, chunk_size):
            chunk = df[start_idx : start_idx + chunk_size]
            logger.info(
                f"Processing chunk {start_idx // chunk_size + 1} ({len(chunk)} rows)"
            )

            # 1. Aggregate product counts
            chunk_product_counts = chunk["Product"].value_counts()
            for product, count in chunk_product_counts.items():
                product_counts[product] = product_counts.get(product, 0) + count

            # 2. Compute narrative lengths for chunk
            chunk["narrative_length"] = (
                chunk["Consumer complaint narrative"].str.split().str.len()
            )
            narrative_lengths.append(chunk[["Product", "narrative_length"]].dropna())

            # 3. Aggregate missing data counts
            chunk_missing = chunk.isnull().sum()
            for col, count in chunk_missing.items():
                missing_counts[col] = missing_counts.get(col, 0) + count

        # Convert aggregated results to DataFrames
        product_dist = pd.Series(product_counts, name="Product").sort_values(
            ascending=False
        )
        product_dist = product_dist / product_dist.sum() * 100
        narrative_lengths_df = pd.concat(narrative_lengths, axis=0)
        missing_stats = pd.Series(missing_counts, name="Missing") / total_rows * 100

        # 1. Product Distribution Plot
        plt.figure(figsize=(12, 8))
        ax = product_dist.plot(kind="barh")
        plt.title("Complaint Distribution by Product", pad=20)
        plt.xlabel("Percentage (%)")
        for i, v in enumerate(product_dist):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        plt.tight_layout()
        output_path = root_dir / "output" / "product_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved product distribution plot to {output_path}")

        # 2. Narrative Length Boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x="Product",
            y="narrative_length",
            data=narrative_lengths_df,
            showfliers=False,
        )
        plt.title("Complaint Length Distribution by Product", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Word Count")
        plt.tight_layout()
        output_path = root_dir / "output" / "narrative_length_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved narrative length distribution plot to {output_path}")

        # 3. Missing Data Analysis
        logger.info(f"Missing data percentages:\n{missing_stats.to_string()}")

        # Optional: Visualize missing data
        plt.figure(figsize=(10, 6))
        missing_stats.plot(kind="bar")
        plt.title("Percentage of Missing Values by Column")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_path = root_dir / "output" / "missing_data.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved missing data plot to {output_path}")

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
