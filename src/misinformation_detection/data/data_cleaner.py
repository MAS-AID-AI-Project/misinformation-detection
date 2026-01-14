import pandas as pd
import re
from pathlib import Path

def fakenewsnet_data_cleaning_pipeline(df, save_path=None):
    """
    Clean FakeNewsNet text data.

    Steps:
      1. Remove missing or empty articles
      2. Clean article text (URLs, HTML tags, non-ASCII chars, etc.)
      3. Optionally save cleaned data

    Args:
        df (pd.DataFrame): Raw FakeNewsNet dataframe with an 'article' column.
        save_path (str or Path, optional): Path to save cleaned CSV. If None, no file is saved.

    Returns:
        pd.DataFrame: Cleaned dataframe with a new 'article_cleaned' column.
    """

    def remove(df):
        # --- Step 1: Remove missing/empty ---
        df = df[
            df["article"].notnull() &
            (df["article"].astype(str).str.strip() != "")
        ].copy()

        # --- Step 1: Remove outliers
        df["article_length"] = df["article"].astype(str).apply(len)

        df = df[
            (df["article_length"] >= 250) &
            (df["article_length"] <= 30000)
        ].copy()

        # --- Step 3: Remove duplicate entries
        df = df.drop_duplicates(subset=["title", "article"], keep="first")

        return df

    def impute(df):

        # Fill URL and date with default placeholder
        df["url"] = df["url"].fillna("UNKNOWN_URL")

        df["date"] = df["date"].fillna("UNKNOWN_DATE")

        from urllib.parse import urlparse

        def extract_domain(url):
            # Treat placeholder as unknown
            if url == "UNKNOWN_URL" or not isinstance(url, str) or url.strip() == "":
                return "unknown"
            try:
                cleaned = url.strip()
                if not cleaned.startswith(("http://", "https://")):
                    cleaned = "https://" + cleaned  # default scheme
                netloc = urlparse(cleaned).netloc
                return netloc.replace("www.", "") if netloc else "invalid"
            except:
                return "invalid"

        # Apply source domain extraction
        df["source_domain"] = df["url"].apply(extract_domain)

        return df

    
    def transform(df):
        # --- Step 1: Clean text ---
        
        def clean_text(text):
            text = str(text).lower().strip()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"[^\x20-\x7E]", " ", text) 
            text = re.sub(r"\s+", " ", text).strip() 
            return text

        df["article_cleaned"] = df["article"].apply(clean_text)

        # Step 2: Parse dates
        from dateutil import parser
        import pandas as pd

        # Define a robust parser
        def robust_parse_date(x):
            try:
                return parser.parse(str(x))
            except:
                return pd.NaT

        # Apply parsing
        df["date_parsed"] = df["date"].apply(robust_parse_date)

        # Format to DD/MM/YYYY
        df["date_cleaned"] = df["date_parsed"].apply(
            lambda x: x.strftime("%d/%m/%Y") if pd.notnull(x) else "UNKNOWN_DATE"
        )

        return df

    if len(df["labels"].unique()) > 2:
        df['label'] = df['label'].replace({'0': 'real', '1': 'fake'})

    # Apply the 3 steps of the pipeline
    df = remove(df)
    df = impute(df)
    df = transform(df)

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Cleaned data saved to {save_path}")

    print(f"Cleaned {len(df):,} rows total")
    
    return df
