from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

from src.text_resources import get_stopwords


TEXT_STOPWORDS = set(
    get_stopwords(
        language="multilingual",
        include_domain=True,
    )
)


def clean_review_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
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

    missing_columns = [
        column
        for column in required_columns
        if column not in cleaned.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn opinii: "
            f"{missing_columns}"
        )

    cleaned = cleaned.dropna(
        subset=[
            "ReviewText",
            "Rating",
            "ProductName",
        ]
    )

    cleaned["ReviewText"] = (
        cleaned["ReviewText"]
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned["ReviewText"].ne("")
    ].copy()

    cleaned["Rating"] = pd.to_numeric(
        cleaned["Rating"],
        errors="coerce",
    )

    cleaned = cleaned.dropna(
        subset=["Rating"]
    )

    cleaned = cleaned[
        cleaned["Rating"].between(
            1,
            5,
        )
    ].copy()

    cleaned["ReviewDate"] = pd.to_datetime(
        cleaned["ReviewDate"],
        errors="coerce",
    )

    cleaned["CleanReviewText"] = (
        cleaned["ReviewText"]
        .apply(normalize_text)
    )

    cleaned = cleaned[
        cleaned["CleanReviewText"].ne("")
    ].copy()

    cleaned["ReviewLength"] = (
        cleaned["ReviewText"]
        .str.len()
    )

    cleaned["WordCount"] = (
        cleaned["CleanReviewText"]
        .apply(
            lambda text: len(
                text.split()
            )
        )
    )

    cleaned["RatingSentiment"] = (
        cleaned["Rating"]
        .apply(
            classify_sentiment_by_rating
        )
    )

    return cleaned.reset_index(
        drop=True
    )


def normalize_text(
    text: str,
) -> str:
    """
    Normalizuje tekst opinii:

    - zmienia litery na małe,
    - zachowuje litery diakrytyczne,
    - usuwa cyfry i znaki specjalne,
    - usuwa nadmiarowe spacje.

    Funkcja jest zgodna między innymi z tekstami
    angielskimi i portugalskimi.
    """
    normalized = unicodedata.normalize(
        "NFKC",
        str(text).lower(),
    )

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    normalized = re.sub(
        r"[_\d]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def classify_sentiment_by_rating(
    rating: float,
) -> str:
    """
    Klasyfikuje sentyment na podstawie oceny
    gwiazdkowej w zakresie 1–5.
    """
    if rating >= 4:
        return "Pozytywny"

    if rating == 3:
        return "Neutralny"

    return "Negatywny"


def review_kpis(
    data: pd.DataFrame,
) -> dict:
    """
    Oblicza podstawowe wskaźniki dla opinii klientów.
    """
    total_reviews = len(data)

    average_rating = (
        data["Rating"].mean()
        if total_reviews > 0
        else 0
    )

    average_review_length = (
        data["ReviewLength"].mean()
        if total_reviews > 0
        else 0
    )

    unique_products = (
        data["ProductName"].nunique()
        if total_reviews > 0
        else 0
    )

    return {
        "total_reviews": total_reviews,
        "average_rating": average_rating,
        "average_review_length": (
            average_review_length
        ),
        "unique_products": unique_products,
    }


def sentiment_distribution(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zwraca rozkład sentymentu opinii.
    """
    return (
        data
        .groupby(
            "RatingSentiment",
            as_index=False,
        )
        .agg(
            Reviews=(
                "ReviewID",
                "count",
            )
        )
        .sort_values(
            "Reviews",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def rating_distribution(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zwraca rozkład ocen gwiazdkowych.
    """
    return (
        data
        .groupby(
            "Rating",
            as_index=False,
        )
        .agg(
            Reviews=(
                "ReviewID",
                "count",
            )
        )
        .sort_values(
            "Rating"
        )
        .reset_index(drop=True)
    )


def top_review_words(
    data: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    """
    Zwraca najczęściej występujące słowa w opiniach
    po usunięciu stopwords językowych i dziedzinowych.
    """
    all_words: list[str] = []

    for text in data[
        "CleanReviewText"
    ].dropna():
        words = [
            word
            for word in str(text).split()
            if (
                len(word) > 2
                and word
                not in TEXT_STOPWORDS
            )
        ]

        all_words.extend(
            words
        )

    word_counts = Counter(
        all_words
    )

    return pd.DataFrame(
        word_counts.most_common(
            limit
        ),
        columns=[
            "Word",
            "Count",
        ],
    )


def product_review_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agreguje opinie według produktu.
    """
    result = (
        data
        .groupby(
            "ProductName",
            as_index=False,
        )
        .agg(
            Reviews=(
                "ReviewID",
                "count",
            ),
            AverageRating=(
                "Rating",
                "mean",
            ),
            AverageReviewLength=(
                "ReviewLength",
                "mean",
            ),
            PositiveReviews=(
                "RatingSentiment",
                lambda values: (
                    values
                    == "Pozytywny"
                ).sum(),
            ),
            NeutralReviews=(
                "RatingSentiment",
                lambda values: (
                    values
                    == "Neutralny"
                ).sum(),
            ),
            NegativeReviews=(
                "RatingSentiment",
                lambda values: (
                    values
                    == "Negatywny"
                ).sum(),
            ),
        )
    )

    result["NegativeShare"] = (
        result["NegativeReviews"]
        / result["Reviews"]
    )

    result["PositiveShare"] = (
        result["PositiveReviews"]
        / result["Reviews"]
    )

    return (
        result
        .sort_values(
            by=[
                "AverageRating",
                "Reviews",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )