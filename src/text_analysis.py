import re
from collections import Counter

import pandas as pd


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "he", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with", "i", "my", "we", "you",
    "your", "very", "after", "again", "than", "too"
}


def clean_review_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści dane tekstowe zawierające opinie klientów.

    Oczekiwane kolumny:
    - ReviewID,
    - ProductID,
    - ProductName,
    - Rating,
    - ReviewDate,
    - ReviewText.
    """
    cleaned = data.copy()

    required_columns = [
        "ReviewID",
        "ProductID",
        "ProductName",
        "Rating",
        "ReviewDate",
        "ReviewText",
    ]

    missing_columns = [col for col in required_columns if col not in cleaned.columns]
    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn opinii: {missing_columns}")

    cleaned = cleaned.dropna(subset=["ReviewText", "Rating", "ProductName"])

    cleaned["Rating"] = pd.to_numeric(cleaned["Rating"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Rating"])

    cleaned = cleaned[(cleaned["Rating"] >= 1) & (cleaned["Rating"] <= 5)]

    cleaned["ReviewDate"] = pd.to_datetime(cleaned["ReviewDate"], errors="coerce")

    cleaned["ReviewText"] = cleaned["ReviewText"].astype(str)
    cleaned["CleanReviewText"] = cleaned["ReviewText"].apply(normalize_text)
    cleaned["ReviewLength"] = cleaned["ReviewText"].str.len()
    cleaned["WordCount"] = cleaned["CleanReviewText"].apply(lambda text: len(text.split()))

    cleaned["RatingSentiment"] = cleaned["Rating"].apply(classify_sentiment_by_rating)

    return cleaned


def normalize_text(text: str) -> str:
    """
    Normalizuje tekst opinii:
    - zmienia litery na małe,
    - usuwa znaki specjalne,
    - usuwa nadmiarowe spacje.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify_sentiment_by_rating(rating: float) -> str:
    """
    Klasyfikuje sentyment na podstawie oceny gwiazdkowej.
    """
    if rating >= 4:
        return "Pozytywny"
    if rating == 3:
        return "Neutralny"
    return "Negatywny"


def review_kpis(data: pd.DataFrame) -> dict:
    """
    Oblicza podstawowe wskaźniki dla opinii klientów.
    """
    total_reviews = len(data)
    average_rating = data["Rating"].mean() if total_reviews > 0 else 0
    average_review_length = data["ReviewLength"].mean() if total_reviews > 0 else 0
    unique_products = data["ProductName"].nunique() if total_reviews > 0 else 0

    return {
        "total_reviews": total_reviews,
        "average_rating": average_rating,
        "average_review_length": average_review_length,
        "unique_products": unique_products,
    }


def sentiment_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca rozkład sentymentu opinii.
    """
    result = (
        data.groupby("RatingSentiment", as_index=False)
        .agg(Reviews=("ReviewID", "count"))
        .sort_values("Reviews", ascending=False)
    )

    return result


def rating_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca rozkład ocen gwiazdkowych.
    """
    result = (
        data.groupby("Rating", as_index=False)
        .agg(Reviews=("ReviewID", "count"))
        .sort_values("Rating")
    )

    return result


def top_review_words(data: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """
    Zwraca najczęściej występujące słowa w oczyszczonych opiniach.
    """
    all_words = []

    for text in data["CleanReviewText"]:
        words = [
            word
            for word in text.split()
            if len(word) > 2 and word not in STOPWORDS
        ]
        all_words.extend(words)

    word_counts = Counter(all_words)

    result = pd.DataFrame(
        word_counts.most_common(limit),
        columns=["Word", "Count"],
    )

    return result


def product_review_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje opinie według produktu.
    """
    result = (
        data.groupby("ProductName", as_index=False)
        .agg(
            Reviews=("ReviewID", "count"),
            AverageRating=("Rating", "mean"),
            AverageReviewLength=("ReviewLength", "mean"),
            PositiveReviews=("RatingSentiment", lambda x: (x == "Pozytywny").sum()),
            NeutralReviews=("RatingSentiment", lambda x: (x == "Neutralny").sum()),
            NegativeReviews=("RatingSentiment", lambda x: (x == "Negatywny").sum()),
        )
    )

    result["NegativeShare"] = result["NegativeReviews"] / result["Reviews"]
    result["PositiveShare"] = result["PositiveReviews"] / result["Reviews"]

    result = result.sort_values(["AverageRating", "Reviews"], ascending=[False, False])

    return result