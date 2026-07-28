from __future__ import annotations

import pandas as pd

from src.model_comparison import (
    compare_sentiment_models,
)
from src.model_interpretability import (
    build_interpretability_analysis,
    explain_single_review,
)
from src.topic_modeling import (
    build_topic_analysis,
)


def test_sentiment_model_comparison(
    review_data: pd.DataFrame,
) -> None:
    results = compare_sentiment_models(
        data=review_data,
        requested_folds=3,
        random_state=42,
        text_language="pt-br",
    )

    comparison_table = results[
        "comparison_table"
    ]

    expected_models = {
        "Logistic Regression",
        "Linear SVM",
        "Multinomial Naive Bayes",
        "Complement Naive Bayes",
    }

    assert len(comparison_table) == 4

    assert set(
        comparison_table["Model"]
    ) == expected_models

    assert results["best_model_name"] in (
        expected_models
    )

    assert comparison_table[
        "AccuracyMean"
    ].between(
        0,
        1,
    ).all()

    assert comparison_table[
        "F1MacroMean"
    ].between(
        0,
        1,
    ).all()

    assert len(
        results["predictions"]
    ) == len(review_data)

    assert results["text_language"] == (
        "portuguese"
    )

    vectorizer = results[
        "best_model"
    ].named_steps[
        "tfidf"
    ]

    configured_stopwords = set(
        vectorizer.stop_words
    )

    assert "que" in configured_stopwords
    assert "não" not in configured_stopwords
    assert "produto" in configured_stopwords


def test_interpretability_analysis(
    review_data: pd.DataFrame,
) -> None:
    results = build_interpretability_analysis(
        data=review_data,
        top_n=10,
        text_language="pt-br",
        random_state=42,
    )

    assert results["number_of_reviews"] == len(
        review_data
    )

    assert results["number_of_features"] > 0

    assert not results[
        "supporting_terms"
    ].empty

    assert not results[
        "opposing_terms"
    ].empty

    explanation = explain_single_review(
        model=results["model"],
        review_text=(
            "excellent quality and durable product"
        ),
        top_n=10,
    )

    assert explanation[
        "predicted_sentiment"
    ] in {
        "Pozytywny",
        "Neutralny",
        "Negatywny",
    }

    assert 0 <= explanation["confidence"] <= 1

    assert not explanation[
        "probabilities"
    ].empty

    assert results["text_language"] == (
        "portuguese"
    )

    vectorizer = results[
        "model"
    ].named_steps[
        "tfidf"
    ]

    configured_stopwords = set(
        vectorizer.stop_words
    )

    assert "que" in configured_stopwords
    assert "não" not in configured_stopwords
    assert "produto" in configured_stopwords


def test_topic_modeling(
    review_data: pd.DataFrame,
) -> None:
    results = build_topic_analysis(
        data=review_data,
        number_of_topics=3,
        top_terms_per_topic=6,
        random_state=42,
        text_language="pt-br",
    )

    assert results["number_of_topics"] == 3

    assert results["number_of_reviews"] == len(
        review_data
    )

    assert len(
        results["topic_overview"]
    ) == 3

    assert len(
        results["review_assignments"]
    ) == len(review_data)

    assert results[
        "review_assignments"
    ]["TopicLabel"].notna().all()

    assert 0 <= results[
        "topic_diversity"
    ] <= 1

    assert 0 <= results[
        "average_topic_dominance"
    ] <= 1

    assert results["text_language"] == (
        "portuguese"
    )

    vectorizer = results[
        "vectorizer"
    ]

    configured_stopwords = set(
        vectorizer.stop_words
    )

    assert "que" in configured_stopwords
    assert "não" not in configured_stopwords
    assert "produto" in configured_stopwords