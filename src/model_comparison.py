from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    # Dostępne w nowszych wersjach scikit-learn.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


def _create_tfidf_pipeline(classifier: Any) -> Pipeline:
    """
    Tworzy wspólny pipeline przetwarzania tekstu i klasyfikacji.

    Każdy model korzysta z takiej samej reprezentacji TF-IDF,
    dzięki czemu porównanie algorytmów jest bardziej uczciwe.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def _prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Waliduje i przygotowuje opinie do porównania modeli.
    """
    required_columns = [
        "CleanReviewText",
        "RatingSentiment",
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Brakuje kolumn wymaganych do porównania modeli: {missing_columns}"
        )

    model_data = data.dropna(
        subset=["CleanReviewText", "RatingSentiment"]
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
            "Do porównania modeli wymagane jest co najmniej 30 opinii."
        )

    class_counts = model_data["RatingSentiment"].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Do porównania modeli wymagane są co najmniej dwie klasy sentymentu."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Każda klasa sentymentu musi zawierać co najmniej dwa przykłady."
        )

    return model_data.reset_index(drop=True)


def _create_cross_validation(
    model_data: pd.DataFrame,
    requested_folds: int,
    random_state: int,
):
    # Tworzy strategię walidacji.
    
    class_counts = model_data["RatingSentiment"].value_counts()
    minimum_class_size = int(class_counts.min())

    unique_texts_per_class = (
        model_data
        .groupby("RatingSentiment")["CleanReviewText"]
        .nunique()
    )

    minimum_unique_texts = int(unique_texts_per_class.min())

    number_of_folds = min(
        requested_folds,
        minimum_class_size,
        minimum_unique_texts,
    )

    if number_of_folds < 2:
        raise ValueError(
            "Zbyt mało zróżnicowanych opinii do przeprowadzenia walidacji."
        )

    if StratifiedGroupKFold is not None:
        cross_validation = StratifiedGroupKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=random_state,
        )

        groups = model_data["CleanReviewText"]

        return (
            cross_validation,
            groups,
            number_of_folds,
            "StratifiedGroupKFold",
        )

    cross_validation = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_state,
    )

    return (
        cross_validation,
        None,
        number_of_folds,
        "StratifiedKFold",
    )


def compare_sentiment_models(
    data: pd.DataFrame,
    requested_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Porównuje kilka modeli klasyfikacji sentymentu.

    Modele:
    - Logistic Regression,
    - Linear SVM,
    - Multinomial Naive Bayes,
    - Complement Naive Bayes.

    Główną miarą wyboru najlepszego modelu jest macro F1.
    """
    model_data = _prepare_model_data(data)

    X = model_data["CleanReviewText"]
    y = model_data["RatingSentiment"]

    (
        cross_validation,
        groups,
        number_of_folds,
        validation_strategy,
    ) = _create_cross_validation(
        model_data=model_data,
        requested_folds=requested_folds,
        random_state=random_state,
    )

    models = {
        "Logistic Regression": _create_tfidf_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )
        ),
        "Linear SVM": _create_tfidf_pipeline(
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            )
        ),
        "Multinomial Naive Bayes": _create_tfidf_pipeline(
            MultinomialNB(
                alpha=0.5,
            )
        ),
        "Complement Naive Bayes": _create_tfidf_pipeline(
            ComplementNB(
                alpha=0.5,
            )
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
    }

    comparison_rows = []

    validation_arguments = {}

    if groups is not None:
        validation_arguments["groups"] = groups

    for model_name, pipeline in models.items():
        scores = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
            **validation_arguments,
        )

        comparison_rows.append(
            {
                "Model": model_name,
                "AccuracyMean": scores["test_accuracy"].mean(),
                "AccuracyStd": scores["test_accuracy"].std(),
                "PrecisionMacroMean": scores[
                    "test_precision_macro"
                ].mean(),
                "RecallMacroMean": scores[
                    "test_recall_macro"
                ].mean(),
                "F1MacroMean": scores["test_f1_macro"].mean(),
                "F1MacroStd": scores["test_f1_macro"].std(),
                "AverageFitTime": scores["fit_time"].mean(),
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)

    comparison_table = comparison_table.sort_values(
        by=["F1MacroMean", "AccuracyMean"],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = comparison_table.iloc[0]["Model"]
    best_pipeline = models[best_model_name]

    out_of_fold_predictions = cross_val_predict(
        estimator=best_pipeline,
        X=X,
        y=y,
        cv=cross_validation,
        n_jobs=-1,
        method="predict",
        **validation_arguments,
    )

    preferred_label_order = [
        "Negatywny",
        "Neutralny",
        "Pozytywny",
    ]

    available_labels = y.unique().tolist()

    labels = [
        label
        for label in preferred_label_order
        if label in available_labels
    ]

    labels.extend(
        sorted(
            label
            for label in available_labels
            if label not in labels
        )
    )

    confusion = confusion_matrix(
        y_true=y,
        y_pred=out_of_fold_predictions,
        labels=labels,
    )

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=[f"Rzeczywiste: {label}" for label in labels],
        columns=[f"Predykcja: {label}" for label in labels],
    )

    report = classification_report(
        y_true=y,
        y_pred=out_of_fold_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Class"})
    )

    prediction_columns = [
        column
        for column in [
            "ReviewID",
            "ProductID",
            "ProductName",
            "Rating",
            "RatingSentiment",
            "ReviewText",
            "CleanReviewText",
        ]
        if column in model_data.columns
    ]

    predictions_dataframe = model_data[
        prediction_columns
    ].copy()

    predictions_dataframe["PredictedSentiment"] = (
        out_of_fold_predictions
    )

    predictions_dataframe["CorrectPrediction"] = (
        predictions_dataframe["RatingSentiment"]
        == predictions_dataframe["PredictedSentiment"]
    )

    error_analysis = predictions_dataframe[
        ~predictions_dataframe["CorrectPrediction"]
    ].copy()

    class_distribution = (
        model_data["RatingSentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    # Po zakończeniu oceny najlepszy model jest uczony na wszystkich danych.
    # Ta wersja może później służyć do predykcji nowych opinii.
    best_pipeline.fit(X, y)

    return {
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "best_model": best_pipeline,
        "confusion_matrix": confusion_dataframe,
        "classification_report": report_dataframe,
        "predictions": predictions_dataframe,
        "errors": error_analysis,
        "class_distribution": class_distribution,
        "number_of_folds": number_of_folds,
        "validation_strategy": validation_strategy,
        "number_of_reviews": len(model_data),
        "labels": labels,
    }