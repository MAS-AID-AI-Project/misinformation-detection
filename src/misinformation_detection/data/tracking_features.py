"""Feature helpers for the week 5 tracking and deployment notebook.

These utilities keep the notebook focused on model tracking while preserving a
stable, auditable feature schema for retraining and serving examples.
"""

from __future__ import annotations

import hashlib
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease


RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 100
LABEL_MAP = {"real": 0, "fake": 1}
INVERSE_LABEL_MAP = {0: "real", 1: "fake"}

BASE_STRUCTURED_COLUMNS = [
    "word_count",
    "sentence_count",
    "avg_word_length",
    "lexical_richness",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "punctuation_count",
    "exclamation_count",
    "exclamation_ratio",
    "stopword_ratio",
    "flesch_reading_ease",
    "has_clickbait",
    "missing_valid_url",
]

# Columns that previous notebooks may create but that should never enter the
# model schema by accident.
DRIFT_COLUMNS_TO_IGNORE = {
    "group_weight",
    "source_domain.1",
    "stopword_count",
    "rule_pred",
    "predicted_label",
    "predicted_proba_fake",
    "true_label",
    "is_correct",
}

CLICKBAIT_PHRASES = [
    "you won't believe",
    "what happened next",
    "shocking",
    "this is why",
    "top 10",
    "goes viral",
    "are freaking out",
    "epic fail",
    "can't stop laughing",
]

lemmatizer = WordNetLemmatizer()


def _stop_words() -> set[str]:
    return set(stopwords.words("english"))


def clean_text(text: object) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_one(text: object) -> str:
    tokens = word_tokenize(str(text))
    return " ".join(lemmatizer.lemmatize(token) for token in tokens)


def clean_and_lemmatize_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "article_cleaned" not in df.columns:
        source_col = "article" if "article" in df.columns else "article_cleaned"
        df["article_cleaned"] = df[source_col].apply(clean_text)
    else:
        df["article_cleaned"] = df["article_cleaned"].fillna("").apply(clean_text)

    if "article_lemmatized" not in df.columns:
        df["article_lemmatized"] = df["article_cleaned"].apply(lemmatize_one)
    else:
        df["article_lemmatized"] = df["article_lemmatized"].fillna("").astype(str)

    return df


def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_and_lemmatize_articles(df).copy()
    stop_words = _stop_words()

    def tokens(text: object) -> list[str]:
        return word_tokenize(str(text))

    def word_count(text: object) -> int:
        return len(tokens(text))

    def sentence_count(text: object) -> int:
        return len(sent_tokenize(str(text)))

    def avg_word_length(text: object) -> float:
        toks = tokens(text)
        return float(np.mean([len(t) for t in toks])) if toks else 0.0

    def lexical_richness(text: object) -> float:
        toks = tokens(text)
        return len(set(toks)) / len(toks) if toks else 0.0

    def pos_ratios(text: object) -> pd.Series:
        toks = tokens(text)
        if not toks:
            return pd.Series({"noun_ratio": 0.0, "verb_ratio": 0.0, "adj_ratio": 0.0})
        tags = nltk.pos_tag(toks)
        total = len(tags)
        return pd.Series(
            {
                "noun_ratio": sum(1 for _, tag in tags if tag.startswith("NN")) / total,
                "verb_ratio": sum(1 for _, tag in tags if tag.startswith("VB")) / total,
                "adj_ratio": sum(1 for _, tag in tags if tag.startswith("JJ")) / total,
            }
        )

    def punctuation_count(text: object) -> int:
        return sum(1 for char in str(text) if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    def exclamation_count(text: object) -> int:
        return str(text).count("!")

    def exclamation_ratio(text: object) -> float:
        toks = tokens(text)
        return str(text).count("!") / len(toks) if toks else 0.0

    def stopword_ratio(text: object) -> float:
        toks = tokens(text)
        return sum(1 for token in toks if token in stop_words) / len(toks) if toks else 0.0

    def has_clickbait(text: object) -> int:
        lower = str(text).lower()
        return int(any(phrase in lower for phrase in CLICKBAIT_PHRASES))

    for col in BASE_STRUCTURED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    rows_needing_features = df[BASE_STRUCTURED_COLUMNS].isna().any(axis=1)
    if rows_needing_features.any():
        idx = rows_needing_features
        text = df.loc[idx, "article_cleaned"]
        df.loc[idx, "word_count"] = text.apply(word_count)
        df.loc[idx, "sentence_count"] = text.apply(sentence_count)
        df.loc[idx, "avg_word_length"] = text.apply(avg_word_length)
        df.loc[idx, "lexical_richness"] = text.apply(lexical_richness)
        df.loc[idx, ["noun_ratio", "verb_ratio", "adj_ratio"]] = text.apply(pos_ratios).values
        df.loc[idx, "punctuation_count"] = text.apply(punctuation_count)
        df.loc[idx, "exclamation_count"] = text.apply(exclamation_count)
        df.loc[idx, "exclamation_ratio"] = text.apply(exclamation_ratio)
        df.loc[idx, "stopword_ratio"] = text.apply(stopword_ratio)
        df.loc[idx, "flesch_reading_ease"] = text.apply(flesch_reading_ease)
        df.loc[idx, "has_clickbait"] = text.apply(has_clickbait)

    if "missing_valid_url" not in df.columns or df["missing_valid_url"].isna().any():
        if "source_domain" not in df.columns:
            df["source_domain"] = "unknown"
        if "missing_valid_url" in df.columns:
            missing_url_idx = df["missing_valid_url"].isna()
        else:
            missing_url_idx = pd.Series(True, index=df.index)
        df.loc[missing_url_idx, "missing_valid_url"] = df.loc[missing_url_idx, "source_domain"].apply(
            lambda domain: 0 if str(domain) in ["unknown", "invalid", "nan"] else 1
        )

    return df


def build_source_columns(
    df: pd.DataFrame, source_domains: list[str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()
    if "source_domain" not in df.columns:
        df["source_domain"] = "unknown"
    df["source_domain"] = df["source_domain"].fillna("unknown").astype(str)

    if source_domains is None:
        source_domains = df["source_domain"].value_counts().nlargest(15).index.tolist()

    df["source_domain_grouped"] = df["source_domain"].where(
        df["source_domain"].isin(source_domains),
        "Other",
    )

    source_columns = ["source_Other"] + [f"source_{domain}" for domain in source_domains]
    df["source_Other"] = df["source_domain_grouped"].eq("Other")
    for domain in source_domains:
        df[f"source_{domain}"] = df["source_domain_grouped"].eq(domain)

    return df, source_domains, source_columns


def canonicalize_feature_frame(
    df: pd.DataFrame, source_domains: list[str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = add_linguistic_features(df)
    df, source_domains, source_columns = build_source_columns(df, source_domains=source_domains)

    for col in BASE_STRUCTURED_COLUMNS + source_columns:
        if col not in df.columns:
            df[col] = 0

    structured_columns = BASE_STRUCTURED_COLUMNS + source_columns
    feature_frame = df[["article_lemmatized", "label"] + structured_columns].copy()
    return feature_frame, source_domains, structured_columns


def matrix_from_feature_frame(
    feature_frame: pd.DataFrame,
    vectorizer,
    structured_columns: list[str],
    feature_names: list[str] | None = None,
    fit_vectorizer: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    text = feature_frame["article_lemmatized"].fillna("").astype(str)
    if fit_vectorizer:
        tfidf = vectorizer.fit_transform(text)
    else:
        tfidf = vectorizer.transform(text)

    tfidf_df = pd.DataFrame(
        tfidf.toarray(),
        columns=[f"tfidf__{name}" for name in vectorizer.get_feature_names_out()],
        index=feature_frame.index,
    )

    structured_df = feature_frame[structured_columns].copy()
    for col in structured_df.columns:
        structured_df[col] = structured_df[col].astype(float)

    x_matrix = pd.concat([tfidf_df, structured_df], axis=1)
    x_matrix = x_matrix.replace([np.inf, -np.inf], 0).fillna(0)

    if feature_names is not None:
        x_matrix = x_matrix.reindex(columns=feature_names, fill_value=0)

    y = feature_frame["label"].map(LABEL_MAP).astype(int)
    return x_matrix.astype(float), y


def load_bloomberg_shift(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"real_news", "fake_news"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Bloomberg shift file is missing columns: {missing}")

    long_df = (
        raw.rename(columns={"real_news": "real", "fake_news": "fake"})
        .melt(value_vars=["real", "fake"], var_name="label", value_name="article")
        .dropna(subset=["article"])
        .reset_index(drop=True)
    )
    long_df["id"] = [f"bloomberg_shift_{i:04d}" for i in range(len(long_df))]
    long_df["title"] = "Bloomberg/GPT-4o article"
    long_df["dataset_source"] = "bloomberg_gpt4o"
    long_df["source_domain"] = "bloomberg.com"
    long_df["date_cleaned"] = "UNKNOWN_DATE"
    return long_df


def build_external_feature_matrix(
    df: pd.DataFrame,
    vectorizer,
    train_feature_names: list[str],
    source_domains: list[str],
    structured_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame, _, _ = canonicalize_feature_frame(df, source_domains=source_domains)
    x_matrix, y = matrix_from_feature_frame(
        feature_frame,
        vectorizer=vectorizer,
        structured_columns=structured_columns,
        feature_names=train_feature_names,
        fit_vectorizer=False,
    )
    return x_matrix, y


def dataset_fingerprint(df: pd.DataFrame, dataset_name: str, version: str) -> dict:
    stable = df.copy()
    stable = stable.reindex(sorted(stable.columns), axis=1)
    csv_bytes = stable.to_csv(index=False).encode("utf-8")
    label_counts = stable["label"].value_counts(dropna=False).to_dict() if "label" in stable.columns else {}
    return {
        "dataset_name": dataset_name,
        "version": version,
        "rows": int(len(stable)),
        "columns": list(stable.columns),
        "ignored_drift_columns": sorted(DRIFT_COLUMNS_TO_IGNORE & set(stable.columns)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
