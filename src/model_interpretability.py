from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.text_resources import (
    get_stopwords,
    normalize_language_code,
)
from src.text_sampling import (
    create_stratified_text_sample,
)


def _prepare_interpretability_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje dane tekstowe do trenowania modelu interpretowalnego.
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
            "Brakuje kolumn wymaganych do interpretacji modelu: "
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
    ]

    if len(model_data) < 30:
        raise ValueError(
            "Do interpretacji modelu wymagane jest "
            "co najmniej 30 opinii."
        )

    class_counts = (
        model_data["RatingSentiment"]
        .value_counts()
    )

    if len(class_counts) < 2:
        raise ValueError(
            "Model wymaga co najmniej dwóch klas sentymentu."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Każda klasa sentymentu musi zawierać "
            "co najmniej dwa przykłady."
        )

    return model_data.reset_index(drop=True)


def train_interpretable_sentiment_model(
    data: pd.DataFrame,
    random_state: int = 42,
    text_language: str | None = "multilingual",
) -> Pipeline:
    """
    Trenuje interpretowalny model
    TF-IDF + Logistic Regression.
    """
    model_data = _prepare_interpretability_data(
        data
    )

    normalized_language = normalize_language_code(
        text_language
    )

    stopwords = get_stopwords(
        language=normalized_language,
        include_domain=True,
        preserve_negations=True,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    stop_words=stopwords,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(
        model_data["CleanReviewText"],
        model_data["RatingSentiment"],
    )

    return model


def _get_multiclass_coefficients(
    classifier: LogisticRegression,
) -> np.ndarray:
    """
    Zwraca współczynniki w układzie:
    liczba klas × liczba cech.

    Obsługuje również klasyfikację binarną, w której scikit-learn
    przechowuje tylko jeden wiersz współczynników.
    """
    coefficients = classifier.coef_

    if (
        len(classifier.classes_) == 2
        and coefficients.shape[0] == 1
    ):
        coefficients = np.vstack(
            [
                -coefficients[0],
                coefficients[0],
            ]
        )

    return coefficients


def extract_global_feature_importance(
    model: Pipeline,
    top_n: int = 20,
) -> dict:
    """
    Wyodrębnia słowa i frazy o największym wpływie globalnym.

    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = _get_multiclass_coefficients(
        classifier
    )

    supporting_rows = []
    opposing_rows = []

    for class_index, class_name in enumerate(
        classifier.classes_
    ):
        class_coefficients = coefficients[class_index]

        supporting_indices = np.argsort(
            class_coefficients
        )[-top_n:][::-1]

        opposing_indices = np.argsort(
            class_coefficients
        )[:top_n]

        for rank, feature_index in enumerate(
            supporting_indices,
            start=1,
        ):
            supporting_rows.append(
                {
                    "Sentiment": class_name,
                    "Rank": rank,
                    "Term": feature_names[feature_index],
                    "Coefficient": class_coefficients[
                        feature_index
                    ],
                    "Interpretation": (
                        "Wspiera przypisanie do klasy"
                    ),
                }
            )

        for rank, feature_index in enumerate(
            opposing_indices,
            start=1,
        ):
            opposing_rows.append(
                {
                    "Sentiment": class_name,
                    "Rank": rank,
                    "Term": feature_names[feature_index],
                    "Coefficient": class_coefficients[
                        feature_index
                    ],
                    "Interpretation": (
                        "Zmniejsza wynik klasy"
                    ),
                }
            )

    return {
        "supporting_terms": pd.DataFrame(
            supporting_rows
        ),
        "opposing_terms": pd.DataFrame(
            opposing_rows
        ),
    }


def explain_single_review(
    model: Pipeline,
    review_text: str,
    top_n: int = 15,
) -> dict:
    """
    Wyjaśnia pojedynczą predykcję modelu Logistic Regression.

    """
    review_text = str(review_text).strip()

    if not review_text:
        raise ValueError(
            "Tekst opinii nie może być pusty."
        )

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    transformed_review = vectorizer.transform(
        [review_text]
    )

    predicted_sentiment = classifier.predict(
        transformed_review
    )[0]

    probabilities = classifier.predict_proba(
        transformed_review
    )[0]

    classes = classifier.classes_

    predicted_class_index = int(
        np.where(
            classes == predicted_sentiment
        )[0][0]
    )

    coefficients = _get_multiclass_coefficients(
        classifier
    )

    predicted_class_coefficients = coefficients[
        predicted_class_index
    ]

    feature_names = vectorizer.get_feature_names_out()

    sparse_review = transformed_review.tocsr().getrow(0)

    contribution_rows = []

    for feature_index, tfidf_value in zip(
        sparse_review.indices,
        sparse_review.data,
    ):
        coefficient = predicted_class_coefficients[
            feature_index
        ]

        contribution = tfidf_value * coefficient

        contribution_rows.append(
            {
                "Term": feature_names[feature_index],
                "TFIDF": tfidf_value,
                "Coefficient": coefficient,
                "Contribution": contribution,
                "Direction": (
                    "Wspiera predykcję"
                    if contribution >= 0
                    else "Działa przeciw predykcji"
                ),
            }
        )

    contributions = pd.DataFrame(
        contribution_rows
    )

    if not contributions.empty:
        contributions["AbsoluteContribution"] = (
            contributions["Contribution"].abs()
        )

        contributions = (
            contributions
            .sort_values(
                by="AbsoluteContribution",
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    probability_data = pd.DataFrame(
        {
            "Sentiment": classes,
            "Probability": probabilities,
        }
    ).sort_values(
        by="Probability",
        ascending=False,
    )

    return {
        "predicted_sentiment": predicted_sentiment,
        "confidence": float(
            probabilities[predicted_class_index]
        ),
        "probabilities": probability_data,
        "contributions": contributions,
    }


def build_interpretability_analysis(
    data: pd.DataFrame,
    top_n: int = 20,
    text_language: str | None = "multilingual",
    random_state: int = 42,
    max_samples: int | None = None,
) -> dict:
    """
    Trenuje model interpretowalny i przygotowuje
    globalne wyniki jego interpretacji.
    """
    if (
        max_samples is not None
        and max_samples < 30
    ):
        raise ValueError(
            "Limit próbki dla interpretowalności "
            "musi wynosić co najmniej 30."
        )

    prepared_model_data = (
        _prepare_interpretability_data(
            data
        )
    )

    source_number_of_reviews = len(
        prepared_model_data
    )

    model_data = create_stratified_text_sample(
        data=prepared_model_data,
        max_samples=max_samples,
        target_column="RatingSentiment",
        random_state=random_state,
    )

    normalized_language = normalize_language_code(
        text_language
    )

    model = train_interpretable_sentiment_model(
        data=model_data,
        random_state=random_state,
        text_language=normalized_language,
    )

    feature_importance = (
        extract_global_feature_importance(
            model=model,
            top_n=top_n,
        )
    )

    return {
        "model": model,
        "supporting_terms": feature_importance[
            "supporting_terms"
        ],
        "opposing_terms": feature_importance[
            "opposing_terms"
        ],
        "classes": model.named_steps[
            "classifier"
        ].classes_.tolist(),
        "number_of_reviews": len(model_data),
        "source_number_of_reviews": (
            source_number_of_reviews
        ),
        "sampled": (
            len(model_data)
            < source_number_of_reviews
        ),
        "sample_limit": max_samples,
        "number_of_features": len(
            model.named_steps[
                "tfidf"
            ].get_feature_names_out()
        ),
        "text_language": normalized_language,
    }