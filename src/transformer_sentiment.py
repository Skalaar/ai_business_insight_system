from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_TRANSFORMER_MODEL = (
    "nlptown/bert-base-multilingual-uncased-sentiment"
)


def _prepare_transformer_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Waliduje i przygotowuje opinie do analizy transformerowej.
    """
    required_columns = [
        "ReviewText",
        "Rating",
        "RatingSentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych przez model transformerowy: "
            f"{missing_columns}"
        )

    model_data = data.dropna(
        subset=required_columns
    ).copy()

    model_data["ReviewText"] = (
        model_data["ReviewText"]
        .astype(str)
        .str.strip()
    )

    model_data["Rating"] = pd.to_numeric(
        model_data["Rating"],
        errors="coerce",
    )

    model_data = model_data.dropna(
        subset=["Rating"]
    )

    model_data = model_data[
        model_data["ReviewText"].str.len() > 0
    ]

    model_data["Rating"] = (
        model_data["Rating"]
        .round()
        .astype(int)
    )

    model_data = model_data[
        model_data["Rating"].between(1, 5)
    ]

    if len(model_data) < 10:
        raise ValueError(
            "Do oceny modelu transformerowego wymagane jest "
            "co najmniej 10 opinii."
        )

    if model_data["RatingSentiment"].nunique() < 2:
        raise ValueError(
            "Do oceny modelu wymagane są co najmniej "
            "dwie klasy sentymentu."
        )

    return model_data.reset_index(drop=True)


def _star_to_sentiment(
    number_of_stars: int,
) -> str:
    """
    Przekształca ocenę gwiazdkową na klasę sentymentu.
    """
    if number_of_stars <= 2:
        return "Negatywny"

    if number_of_stars == 3:
        return "Neutralny"

    return "Pozytywny"


def _extract_star_labels(
    model: Any,
    number_of_classes: int,
) -> list[int]:
    """
    Odczytuje przypisanie wyjść modelu do liczby gwiazdek.

    W razie braku czytelnych etykiet wykorzystuje kolejność 1–5.
    """
    star_labels = []

    for class_index in range(number_of_classes):
        raw_label = model.config.id2label.get(
            class_index,
            model.config.id2label.get(
                str(class_index),
                str(class_index + 1),
            ),
        )

        match = re.search(
            r"\b([1-5])\b",
            str(raw_label),
        )

        if match:
            star_labels.append(
                int(match.group(1))
            )
        else:
            star_labels.append(
                class_index + 1
            )

    return star_labels


def load_transformer_sentiment_resources(
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
) -> dict:
    """
    Ładuje tokenizer i model transformerowy.

    Model korzysta z GPU, jeśli PyTorch wykryje CUDA.
    W przeciwnym razie działa na procesorze.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    device_name = (
        "GPU — CUDA"
        if device.type == "cuda"
        else "CPU"
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "device_name": device_name,
        "model_name": model_name,
    }


def evaluate_transformer_sentiment(
    data: pd.DataFrame,
    resources: dict,
    batch_size: int = 16,
    max_length: int = 256,
) -> dict:
    """
    Przeprowadza predykcję sentymentu i ocenę modelu BERT.

    Model zwraca rozkład prawdopodobieństwa dla ocen 1–5.
    Predykowana ocena jest następnie przekształcana na
    negatywny, neutralny albo pozytywny sentyment.
    """
    if batch_size < 1:
        raise ValueError(
            "Rozmiar partii musi być większy od zera."
        )

    model_data = _prepare_transformer_data(
        data
    )

    tokenizer = resources["tokenizer"]
    model = resources["model"]
    device = resources["device"]

    review_texts = model_data[
        "ReviewText"
    ].tolist()

    probability_batches = []

    start_time = time.perf_counter()

    for batch_start in range(
        0,
        len(review_texts),
        batch_size,
    ):
        batch_texts = review_texts[
            batch_start:
            batch_start + batch_size
        ]

        encoded_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded_batch = {
            key: value.to(device)
            for key, value
            in encoded_batch.items()
        }

        with torch.inference_mode():
            outputs = model(
                **encoded_batch
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        probability_batches.append(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    all_probabilities = np.vstack(
        probability_batches
    )

    star_labels = _extract_star_labels(
        model=model,
        number_of_classes=(
            all_probabilities.shape[1]
        ),
    )

    predicted_class_indices = np.argmax(
        all_probabilities,
        axis=1,
    )

    predicted_stars = np.asarray(
        [
            star_labels[class_index]
            for class_index
            in predicted_class_indices
        ],
        dtype=int,
    )

    transformer_sentiments = [
        _star_to_sentiment(
            number_of_stars
        )
        for number_of_stars
        in predicted_stars
    ]

    prediction_confidences = (
        all_probabilities[
            np.arange(
                len(all_probabilities)
            ),
            predicted_class_indices,
        ]
    )

    output_columns = [
        column
        for column in [
            "ReviewID",
            "ProductID",
            "ProductName",
            "Rating",
            "RatingSentiment",
            "ReviewText",
        ]
        if column in model_data.columns
    ]

    predictions = model_data[
        output_columns
    ].copy()

    predictions["TransformerStars"] = (
        predicted_stars
    )

    predictions["TransformerSentiment"] = (
        transformer_sentiments
    )

    predictions["TransformerConfidence"] = (
        prediction_confidences
    )

    for class_index, number_of_stars in enumerate(
        star_labels
    ):
        predictions[
            f"Probability{number_of_stars}Star"
        ] = all_probabilities[
            :,
            class_index,
        ]

    negative_indices = [
        index
        for index, stars in enumerate(
            star_labels
        )
        if stars <= 2
    ]

    neutral_indices = [
        index
        for index, stars in enumerate(
            star_labels
        )
        if stars == 3
    ]

    positive_indices = [
        index
        for index, stars in enumerate(
            star_labels
        )
        if stars >= 4
    ]

    predictions["NegativeProbability"] = (
        all_probabilities[
            :,
            negative_indices,
        ].sum(axis=1)
    )

    predictions["NeutralProbability"] = (
        all_probabilities[
            :,
            neutral_indices,
        ].sum(axis=1)
    )

    predictions["PositiveProbability"] = (
        all_probabilities[
            :,
            positive_indices,
        ].sum(axis=1)
    )

    predictions["SentimentAgreement"] = (
        predictions["RatingSentiment"]
        == predictions[
            "TransformerSentiment"
        ]
    )

    predictions["RatingError"] = (
        predictions["TransformerStars"]
        - predictions["Rating"]
    )

    actual_sentiments = predictions[
        "RatingSentiment"
    ]

    predicted_sentiments = predictions[
        "TransformerSentiment"
    ]

    preferred_label_order = [
        "Negatywny",
        "Neutralny",
        "Pozytywny",
    ]

    available_labels = set(
        actual_sentiments.unique()
    ) | set(
        predicted_sentiments.unique()
    )

    labels = [
        label
        for label in preferred_label_order
        if label in available_labels
    ]

    confusion = confusion_matrix(
        y_true=actual_sentiments,
        y_pred=predicted_sentiments,
        labels=labels,
    )

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=[
            f"Rzeczywiste: {label}"
            for label in labels
        ],
        columns=[
            f"Predykcja: {label}"
            for label in labels
        ],
    )

    report = classification_report(
        y_true=actual_sentiments,
        y_pred=predicted_sentiments,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "Class",
            }
        )
    )

    sentiment_accuracy = accuracy_score(
        actual_sentiments,
        predicted_sentiments,
    )

    precision_macro = precision_score(
        actual_sentiments,
        predicted_sentiments,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        actual_sentiments,
        predicted_sentiments,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        actual_sentiments,
        predicted_sentiments,
        average="macro",
        zero_division=0,
    )

    exact_star_accuracy = float(
        (
            predictions["Rating"]
            == predictions[
                "TransformerStars"
            ]
        ).mean()
    )

    within_one_star_accuracy = float(
        (
            predictions[
                "RatingError"
            ].abs()
            <= 1
        ).mean()
    )

    star_mae = mean_absolute_error(
        predictions["Rating"],
        predictions[
            "TransformerStars"
        ],
    )

    overall_metrics = pd.DataFrame(
        [
            {
                "Model": resources[
                    "model_name"
                ],
                "SentimentAccuracy": (
                    sentiment_accuracy
                ),
                "PrecisionMacro": (
                    precision_macro
                ),
                "RecallMacro": recall_macro,
                "F1Macro": f1_macro,
                "StarMAE": float(star_mae),
                "ExactStarAccuracy": (
                    exact_star_accuracy
                ),
                "WithinOneStarAccuracy": (
                    within_one_star_accuracy
                ),
                "RuntimeSeconds": (
                    runtime_seconds
                ),
                "ReviewsProcessed": len(
                    predictions
                ),
                "Device": resources[
                    "device_name"
                ],
            }
        ]
    )

    errors = predictions[
        ~predictions[
            "SentimentAgreement"
        ]
    ].copy()

    predicted_distribution = (
        predictions[
            "TransformerSentiment"
        ]
        .value_counts()
        .rename_axis(
            "TransformerSentiment"
        )
        .reset_index(
            name="Reviews"
        )
    )

    if "ProductName" in predictions.columns:
        product_summary = (
            predictions
            .groupby(
                "ProductName",
                as_index=False,
            )
            .agg(
                Reviews=(
                    "TransformerSentiment",
                    "size",
                ),
                SentimentAccuracy=(
                    "SentimentAgreement",
                    "mean",
                ),
                AverageActualRating=(
                    "Rating",
                    "mean",
                ),
                AveragePredictedStars=(
                    "TransformerStars",
                    "mean",
                ),
                AverageConfidence=(
                    "TransformerConfidence",
                    "mean",
                ),
            )
            .sort_values(
                by="Reviews",
                ascending=False,
            )
            .reset_index(drop=True)
        )
    else:
        product_summary = pd.DataFrame()

    return {
        "metrics": overall_metrics,
        "predictions": predictions,
        "errors": errors,
        "confusion_matrix": (
            confusion_dataframe
        ),
        "classification_report": (
            report_dataframe
        ),
        "predicted_distribution": (
            predicted_distribution
        ),
        "product_summary": product_summary,
        "labels": labels,
        "model_name": resources[
            "model_name"
        ],
        "device_name": resources[
            "device_name"
        ],
        "number_of_reviews": len(
            predictions
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
    }