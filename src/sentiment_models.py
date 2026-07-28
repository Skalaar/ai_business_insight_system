from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_resources import (
    get_stopwords,
    normalize_language_code,
)


def _prepare_sentiment_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Waliduje i przygotowuje dane do trenowania
    bazowego modelu sentymentu.
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
            "Brakuje wymaganych kolumn do trenowania modelu: "
            f"{missing_columns}"
        )

    model_data = data.dropna(
        subset=required_columns
    ).copy()

    model_data["CleanReviewText"] = (
        model_data["CleanReviewText"]
        .astype(str)
        .str.strip()
    )

    model_data = model_data[
        model_data["CleanReviewText"].str.len() > 0
    ].copy()

    if len(model_data) < 10:
        raise ValueError(
            "Zbyt mało opinii do trenowania modelu. "
            "Wymagane jest co najmniej 10 rekordów."
        )

    class_counts = (
        model_data["RatingSentiment"]
        .value_counts()
    )

    available_classes = (
        class_counts[
            class_counts >= 2
        ]
        .index
        .tolist()
    )

    model_data = model_data[
        model_data["RatingSentiment"].isin(
            available_classes
        )
    ].copy()

    if (
        model_data["RatingSentiment"]
        .nunique()
        < 2
    ):
        raise ValueError(
            "Do trenowania modelu wymagane są "
            "co najmniej dwie klasy sentymentu."
        )

    return model_data.reset_index(
        drop=True
    )


def train_tfidf_logistic_regression_model(
    data: pd.DataFrame,
    text_language: str | None = "multilingual",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Trenuje bazowy model klasyfikacji sentymentu.

    Model wykorzystuje:
    - reprezentację tekstu TF-IDF,
    - Logistic Regression,
    - stopwords dopasowane do języka danych,
    - zbalansowane wagi klas.

    Etykietą uczącą jest RatingSentiment,
    czyli sentyment wyznaczony na podstawie
    oceny gwiazdkowej.
    """
    model_data = _prepare_sentiment_data(
        data
    )

    normalized_language = (
        normalize_language_code(
            text_language
        )
    )

    stopwords = get_stopwords(
        language=normalized_language,
        include_domain=True,
        preserve_negations=True,
    )

    features = model_data[
        "CleanReviewText"
    ]

    labels = model_data[
        "RatingSentiment"
    ]

    stratify_values = (
        labels
        if labels.value_counts().min() >= 2
        else None
    )

    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify_values,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words=stopwords,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(
        features_train,
        labels_train,
    )

    predicted_labels = pipeline.predict(
        features_test
    )

    accuracy = accuracy_score(
        labels_test,
        predicted_labels,
    )

    report_dictionary = (
        classification_report(
            labels_test,
            predicted_labels,
            output_dict=True,
            zero_division=0,
        )
    )

    preferred_label_order = [
        "Negatywny",
        "Neutralny",
        "Pozytywny",
    ]

    available_labels = (
        model_data["RatingSentiment"]
        .unique()
        .tolist()
    )

    labels_order = [
        label
        for label in preferred_label_order
        if label in available_labels
    ]

    labels_order.extend(
        sorted(
            label
            for label in available_labels
            if label not in labels_order
        )
    )

    confusion = confusion_matrix(
        y_true=labels_test,
        y_pred=predicted_labels,
        labels=labels_order,
    )

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=[
            f"Rzeczywiste: {label}"
            for label in labels_order
        ],
        columns=[
            f"Predykcja: {label}"
            for label in labels_order
        ],
    )

    report_dataframe = (
        pd.DataFrame(
            report_dictionary
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "Class",
            }
        )
    )

    predictions_dataframe = pd.DataFrame(
        {
            "ReviewText": (
                features_test.values
            ),
            "ActualSentiment": (
                labels_test.values
            ),
            "PredictedSentiment": (
                predicted_labels
            ),
        }
    )

    predictions_dataframe[
        "CorrectPrediction"
    ] = (
        predictions_dataframe[
            "ActualSentiment"
        ]
        == predictions_dataframe[
            "PredictedSentiment"
        ]
    )

    vectorizer = pipeline.named_steps[
        "tfidf"
    ]

    return {
        "model": pipeline,
        "accuracy": float(accuracy),
        "classification_report": (
            report_dataframe
        ),
        "confusion_matrix": (
            confusion_dataframe
        ),
        "predictions": (
            predictions_dataframe
        ),
        "train_size": len(
            features_train
        ),
        "test_size": len(
            features_test
        ),
        "classes": labels_order,
        "text_language": (
            normalized_language
        ),
        "number_of_features": len(
            vectorizer.get_feature_names_out()
        ),
    }


def predict_sentiment_for_reviews(
    data: pd.DataFrame,
    model: Pipeline,
) -> pd.DataFrame:
    """
    Dodaje do danych opinii sentyment
    przewidziany przez model ML.
    """
    if "CleanReviewText" not in data.columns:
        raise ValueError(
            "Brakuje kolumny CleanReviewText "
            "w danych przeznaczonych do predykcji."
        )

    predicted_data = data.copy()

    predicted_data["MLSentiment"] = (
        model.predict(
            predicted_data[
                "CleanReviewText"
            ]
        )
    )

    if "RatingSentiment" in predicted_data.columns:
        predicted_data[
            "SentimentAgreement"
        ] = (
            predicted_data[
                "RatingSentiment"
            ]
            == predicted_data[
                "MLSentiment"
            ]
        )

    return predicted_data


def ml_sentiment_distribution(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zwraca rozkład sentymentu przewidzianego
    przez model ML.
    """
    if "MLSentiment" not in data.columns:
        raise ValueError(
            "Brakuje kolumny MLSentiment."
        )

    return (
        data
        .groupby(
            "MLSentiment",
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


def sentiment_comparison_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Porównuje sentyment wynikający z oceny
    gwiazdkowej z sentymentem modelu ML.
    """
    required_columns = [
        "RatingSentiment",
        "MLSentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych do "
            "porównania sentymentu: "
            f"{missing_columns}"
        )

    return (
        data
        .groupby(
            [
                "RatingSentiment",
                "MLSentiment",
            ],
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