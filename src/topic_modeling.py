from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def _prepare_topic_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Waliduje i przygotowuje dane tekstowe do modelowania tematów.
    """
    required_columns = [
        "CleanReviewText",
        "RatingSentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych do modelowania tematów: "
            f"{missing_columns}"
        )

    topic_data = data.dropna(
        subset=required_columns
    ).copy()

    topic_data["CleanReviewText"] = (
        topic_data["CleanReviewText"]
        .astype(str)
        .str.strip()
    )

    topic_data = topic_data[
        topic_data["CleanReviewText"].str.len() > 0
    ]

    if len(topic_data) < 30:
        raise ValueError(
            "Do modelowania tematów wymagane jest "
            "co najmniej 30 opinii."
        )

    if topic_data["CleanReviewText"].nunique() < 10:
        raise ValueError(
            "Zbiór zawiera zbyt mało unikalnych opinii "
            "do wiarygodnego wykrywania tematów."
        )

    return topic_data.reset_index(drop=True)


def _build_topic_label(
    topic_number: int,
    top_terms: list[str],
) -> str:
    """
    Buduje automatyczną etykietę tematu na podstawie
    trzech najwyżej ocenionych terminów.
    """
    label_terms = top_terms[:3]

    return (
        f"Temat {topic_number + 1}: "
        + " / ".join(label_terms)
    )


def _calculate_topic_diversity(
    topic_terms: pd.DataFrame,
) -> float:
    """
    Oblicza udział unikalnych terminów wśród wszystkich
    najważniejszych terminów tematów.

    Wartość bliska 1 oznacza, że tematy korzystają
    z odmiennych słów. Niska wartość wskazuje,
    że tematy są do siebie podobne.
    """
    if topic_terms.empty:
        return 0.0

    total_terms = len(topic_terms)
    unique_terms = topic_terms["Term"].nunique()

    return (
        unique_terms / total_terms
        if total_terms > 0
        else 0.0
    )


def build_topic_analysis(
    data: pd.DataFrame,
    number_of_topics: int = 5,
    top_terms_per_topic: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Wykonuje modelowanie tematów za pomocą TF-IDF i NMF.

    """
    if number_of_topics < 2:
        raise ValueError(
            "Liczba tematów musi wynosić co najmniej 2."
        )

    if top_terms_per_topic < 3:
        raise ValueError(
            "Każdy temat powinien zawierać "
            "co najmniej 3 najważniejsze terminy."
        )

    topic_data = _prepare_topic_data(data)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )

    document_term_matrix = vectorizer.fit_transform(
        topic_data["CleanReviewText"]
    )

    number_of_features = document_term_matrix.shape[1]

    if number_of_features < number_of_topics:
        raise ValueError(
            "Liczba dostępnych cech tekstowych jest mniejsza "
            "od wybranej liczby tematów."
        )

    topic_model = NMF(
        n_components=number_of_topics,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=600,
        random_state=random_state,
    )

    document_topic_matrix = topic_model.fit_transform(
        document_term_matrix
    )

    feature_names = vectorizer.get_feature_names_out()

    topic_term_rows = []
    topic_labels = {}
    topic_overview_rows = []

    for topic_index, topic_components in enumerate(
        topic_model.components_
    ):
        strongest_indices = np.argsort(
            topic_components
        )[-top_terms_per_topic:][::-1]

        strongest_terms = [
            feature_names[feature_index]
            for feature_index in strongest_indices
        ]

        topic_label = _build_topic_label(
            topic_number=topic_index,
            top_terms=strongest_terms,
        )

        topic_labels[topic_index] = topic_label

        topic_overview_rows.append(
            {
                "TopicIndex": topic_index,
                "TopicLabel": topic_label,
                "TopTerms": ", ".join(strongest_terms),
            }
        )

        for rank, feature_index in enumerate(
            strongest_indices,
            start=1,
        ):
            topic_term_rows.append(
                {
                    "TopicIndex": topic_index,
                    "TopicLabel": topic_label,
                    "Rank": rank,
                    "Term": feature_names[feature_index],
                    "Weight": topic_components[
                        feature_index
                    ],
                }
            )

    topic_terms = pd.DataFrame(topic_term_rows)
    topic_overview = pd.DataFrame(topic_overview_rows)

    dominant_topic_indices = np.argmax(
        document_topic_matrix,
        axis=1,
    )

    dominant_topic_weights = np.max(
        document_topic_matrix,
        axis=1,
    )

    topic_weight_sums = document_topic_matrix.sum(
        axis=1
    )

    dominant_topic_shares = np.divide(
        dominant_topic_weights,
        topic_weight_sums,
        out=np.zeros_like(dominant_topic_weights),
        where=topic_weight_sums != 0,
    )

    assignment_columns = [
        column
        for column in [
            "ReviewID",
            "ProductID",
            "ProductName",
            "Rating",
            "ReviewDate",
            "RatingSentiment",
            "ReviewText",
            "CleanReviewText",
        ]
        if column in topic_data.columns
    ]

    review_assignments = topic_data[
        assignment_columns
    ].copy()

    review_assignments["TopicIndex"] = (
        dominant_topic_indices
    )

    review_assignments["TopicLabel"] = (
        review_assignments["TopicIndex"]
        .map(topic_labels)
    )

    review_assignments["DominantTopicWeight"] = (
        dominant_topic_weights
    )

    review_assignments["DominantTopicShare"] = (
        dominant_topic_shares
    )

    topic_distribution = (
        review_assignments
        .groupby(
            ["TopicIndex", "TopicLabel"],
            as_index=False,
        )
        .agg(
            Reviews=("TopicLabel", "size"),
            AverageDominance=(
                "DominantTopicShare",
                "mean",
            ),
        )
        .sort_values(
            by="Reviews",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    topic_distribution["ReviewShare"] = (
        topic_distribution["Reviews"]
        / len(review_assignments)
    )

    topic_sentiment_summary = (
        review_assignments
        .groupby(
            ["TopicIndex", "TopicLabel", "RatingSentiment"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Reviews"})
    )

    topic_sentiment_totals = (
        topic_sentiment_summary
        .groupby(
            ["TopicIndex", "TopicLabel"]
        )["Reviews"]
        .transform("sum")
    )

    topic_sentiment_summary["SentimentShare"] = (
        topic_sentiment_summary["Reviews"]
        / topic_sentiment_totals
    )

    if "ProductName" in review_assignments.columns:
        product_topic_summary = (
            review_assignments
            .groupby(
                ["ProductName", "TopicIndex", "TopicLabel"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "Reviews"})
            .sort_values(
                by="Reviews",
                ascending=False,
            )
            .reset_index(drop=True)
        )
    else:
        product_topic_summary = pd.DataFrame()

    topic_diversity = _calculate_topic_diversity(
        topic_terms
    )

    return {
        "model": topic_model,
        "vectorizer": vectorizer,
        "topic_terms": topic_terms,
        "topic_overview": topic_overview,
        "review_assignments": review_assignments,
        "topic_distribution": topic_distribution,
        "topic_sentiment_summary": topic_sentiment_summary,
        "product_topic_summary": product_topic_summary,
        "number_of_topics": number_of_topics,
        "number_of_reviews": len(topic_data),
        "number_of_features": number_of_features,
        "reconstruction_error": float(
            topic_model.reconstruction_err_
        ),
        "topic_diversity": float(topic_diversity),
        "average_topic_dominance": float(
            dominant_topic_shares.mean()
        ),
    }