import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from .data_loader import load_fakenewsnet_from_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Helper functions
def basic_lingusitic_features(df):
    # Define helper functions
    def word_count(text):
        tokens = word_tokenize(text)
        return len(tokens)

    def sentence_count(text):
        sentences = sent_tokenize(text)
        return len(sentences)

    def avg_word_length(text):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0
        return np.mean([len(w) for w in tokens])

    def lexical_richness(text):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    # Apply to the cleaned article text
    df["word_count"] = df["article_cleaned"].apply(word_count)
    df["sentence_count"] = df["article_cleaned"].apply(sentence_count)
    df["avg_word_length"] = df["article_cleaned"].apply(avg_word_length)
    df["lexical_richness"] = df["article_cleaned"].apply(lexical_richness)

    return df

def advanced_linguistic_features(df):
    # Make sure resources are ready
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    # --- POS Tag Ratios ---
    def pos_ratios(text):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return pd.Series({
                "noun_ratio": 0,
                "verb_ratio": 0,
                "adj_ratio": 0
            })
        tags = nltk.pos_tag(tokens)
        num_nouns = sum(1 for word, tag in tags if tag.startswith('NN'))
        num_verbs = sum(1 for word, tag in tags if tag.startswith('VB'))
        num_adjs  = sum(1 for word, tag in tags if tag.startswith('JJ'))
        total = len(tags)
        return pd.Series({
            "noun_ratio": num_nouns / total,
            "verb_ratio": num_verbs / total,
            "adj_ratio": num_adjs / total
        })

    # --- Punctuation & Exclamation ---
    def punctuation_count(text):
        return sum(1 for c in str(text) if c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    def exclamation_count(text):
        return str(text).count("!")

    def exclamation_ratio(text):
        tokens = word_tokenize(text)
        return str(text).count("!") / len(tokens) if len(tokens) > 0 else 0

    # --- Stopword Ratio ---
    def stopword_ratio(text):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0
        num_stopwords = sum(1 for w in tokens if w in stop_words)
        return num_stopwords / len(tokens)

    # --- Readability ---
    def flesch_score(text):
        return flesch_reading_ease(text)

    # --- Clickbait Phrase Flag ---
    clickbait_phrases = [
        "you won’t believe", "what happened next", "shocking", "this is why",
        "top 10", "goes viral", "are freaking out", "epic fail", "can’t stop laughing"
    ]

    def has_clickbait(text):
        text = str(text).lower()
        for phrase in clickbait_phrases:
            if phrase in text:
                return 1
        return 0

    # === Apply to cleaned article text ===
    df[["noun_ratio", "verb_ratio", "adj_ratio"]] = df["article_cleaned"].apply(pos_ratios)
    df["punctuation_count"] = df["article_cleaned"].apply(punctuation_count)
    df["exclamation_count"] = df["article_cleaned"].apply(exclamation_count)
    df["exclamation_ratio"] = df["article_cleaned"].apply(exclamation_ratio)
    df["stopword_ratio"] = df["article_cleaned"].apply(stopword_ratio)
    df["flesch_reading_ease"] = df["article_cleaned"].apply(flesch_score)
    df["has_clickbait"] = df["article_cleaned"].apply(has_clickbait)

    return df

def source_domain_one_hot_encoding(df):
    # Count top 15 domains
    top_domains = df["source_domain"].value_counts().nlargest(15).index.tolist()

    # Assign 'Other' to all less common domains
    df["source_domain_grouped"] = df["source_domain"].apply(
        lambda x: x if x in top_domains else "Other"
    )

    # One-hot encode the grouped domain column
    source_dummies = pd.get_dummies(df["source_domain_grouped"], prefix="source")

    # Add to main DataFrame
    df = pd.concat([df, source_dummies], axis=1)

    return df

def source_domain_missing_valid_url_flag(df):
    df["missing_valid_url"] = df["source_domain"].apply(
        lambda x: 0 if x in ["unknown", "invalid"] else 1
    )

    return df

def engineer_features(df, save_path=None):
    """
    Perform comprehensive feature engineering on the input DataFrame containing news articles.

    This function applies multiple linguistic and source-related transformations to enrich the dataset 
    with meaningful features for downstream analysis or machine learning modeling.

    Features engineered include:

    1. Basic linguistic features extracted from the article text:
       - Word count
       - Sentence count
       - Average word length
       - Lexical richness (unique word ratio)

    2. Advanced linguistic features including:
       - Part-of-speech (POS) tag ratios: noun, verb, adjective proportions
       - Punctuation and exclamation counts and ratios
       - Stopword usage ratio
       - Readability score (Flesch Reading Ease)
       - Clickbait phrase presence flag

    3. Source domain encoding:
       - One-hot encoding of the top 15 most frequent source domains with others grouped as 'Other'

    4. URL validity flag:
       - Binary indicator flag marking whether a URL is valid or missing/invalid based on extracted domain

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with at least the following columns:
         - "article_cleaned": cleaned plain-text article content for linguistic features
         - "source_domain": domain extracted from article URL for source encoding

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame augmented with newly engineered feature columns ready for model use.
    """
    

    # Apply feature engineering steps:
    df = basic_lingusitic_features(df)
    df = advanced_linguistic_features(df)
    df = source_domain_one_hot_encoding(df)
    df = source_domain_missing_valid_url_flag(df)

    # Save if requested
    if save_path is not None:
        save_path = os.path.abspath(save_path)
        if hasattr(save_path, 'parent'):
            save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Feature engineered data saved to {save_path}")

    return df


def prepare_text_structured_features_full(
    csv_path="",
    df=None,
    text_column="article_lemmatized",
    target_column="label",
    columns_to_exclude=None,
    test_size=0.2,
    random_state=42,
    max_tfidf_features=100,
    vectorizer=None,
    inference_mode=False,
    verbose=False
):
    """
    Loads data, applies TF-IDF, and merges structured features.
    If inference_mode=True, assumes df is unlabeled and skips splitting.

    Returns:
      If training mode: 
          X_train_combined, X_test_combined, X_structured_train_full, 
          X_structured_test_full, y_train, y_test, vectorizer
      If inference mode:
          X_inference_combined, X_structured_inference_full, vectorizer
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load the dataframe if not provided
    if df is None:
        df = load_fakenewsnet_from_dataframe(csv_path)

    if verbose:
        print(f"Loaded DataFrame shape: {df.shape}")

    # Columns to exclude by default
    if columns_to_exclude is None:
        columns_to_exclude = [
            "id", "title", "label", "dataset_source", "source_domain", "source_domain.1",
            "article_cleaned", "date_cleaned", "source_domain_grouped",
            "article_lemmatized", "rule_pred"
        ]

    # ---------------- INFERENCE MODE ----------------
    if inference_mode:
        if vectorizer is None:
            raise ValueError("Inference mode requires a fitted vectorizer.")

        X_text_inference = df[text_column]
        X_structured_inference_full = df.copy()

        X_inference_tfidf = vectorizer.transform(X_text_inference)
        tfidf_inference_df = pd.DataFrame(
            X_inference_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=X_structured_inference_full.index
        )

        X_structured_inference = X_structured_inference_full.drop(columns=columns_to_exclude, errors="ignore")
        X_inference_combined = pd.concat([tfidf_inference_df, X_structured_inference], axis=1)

        if verbose:
            print(f"Inference combined shape: {X_inference_combined.shape}")

        return X_inference_combined, X_structured_inference_full, vectorizer

    # ---------------- TRAINING MODE ----------------
    X_text = df[text_column]
    y = df[target_column]

    X_text_train, X_text_test, X_structured_train_full, X_structured_test_full, y_train, y_test = train_test_split(
        X_text,
        df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words="english")

    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    tfidf_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=X_structured_train_full.index)
    tfidf_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=X_structured_test_full.index)

    X_structured_train = X_structured_train_full.drop(columns=columns_to_exclude, errors="ignore")
    X_structured_test = X_structured_test_full.drop(columns=columns_to_exclude, errors="ignore")

    X_train_combined = pd.concat([tfidf_train_df, X_structured_train], axis=1)
    X_test_combined = pd.concat([tfidf_test_df, X_structured_test], axis=1)

    if verbose:
        print(f"Train combined shape: {X_train_combined.shape}")
        print(f"Test combined shape: {X_test_combined.shape}")

    return (
        X_train_combined,
        X_test_combined,
        X_structured_train_full,
        X_structured_test_full,
        y_train,
        y_test,
        vectorizer
    )
