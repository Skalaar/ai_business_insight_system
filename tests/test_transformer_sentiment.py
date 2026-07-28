from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from src.transformer_sentiment import (
    evaluate_transformer_sentiment,
)


class FakeTokenizer:
    """
    Minimalny tokenizer używany wyłącznie w teście.
    Nie pobiera żadnych zasobów z internetu.
    """

    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        batch_size = len(texts)

        input_ids = torch.ones(
            (batch_size, 4),
            dtype=torch.long,
        )

        attention_mask = torch.ones(
            (batch_size, 4),
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class FakeModel:
    """
    Minimalny model zwracający pięć wyników
    odpowiadających ocenom od 1 do 5 gwiazdek.
    """

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            id2label={
                0: "Very Negative",
                1: "Negative",
                2: "Neutral",
                3: "Positive",
                4: "Very Positive",
            }
        )

    def __call__(self, **encoded_batch):
        batch_size = encoded_batch[
            "input_ids"
        ].shape[0]

        logits = torch.tensor(
            [
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ]
            ],
            dtype=torch.float32,
        ).repeat(
            batch_size,
            1,
        )

        return SimpleNamespace(
            logits=logits
        )


def test_transformer_evaluation_uses_sample() -> None:
    rows = []

    sentiment_configuration = [
        (
            "Negatywny",
            1,
            "opinia negatywna",
        ),
        (
            "Neutralny",
            3,
            "opinia neutralna",
        ),
        (
            "Pozytywny",
            5,
            "opinia pozytywna",
        ),
    ]

    review_id = 1

    for (
        sentiment,
        rating,
        review_text,
    ) in sentiment_configuration:
        for row_number in range(10):
            rows.append(
                {
                    "ReviewID": review_id,
                    "ProductID": (
                        f"PRODUCT-{row_number:03d}"
                    ),
                    "ProductName": (
                        f"Produkt {row_number}"
                    ),
                    "Rating": rating,
                    "RatingSentiment": sentiment,
                    "ReviewText": (
                        f"{review_text} {row_number}"
                    ),
                }
            )

            review_id += 1

    review_data = pd.DataFrame(
        rows
    )

    resources = {
        "tokenizer": FakeTokenizer(),
        "model": FakeModel(),
        "device": torch.device("cpu"),
        "device_name": "CPU",
        "model_name": "fake-transformer",
        "model_license": "Test license",
        "model_language": "Portuguese",
        "model_output": "5 sentiment classes",
    }

    results = evaluate_transformer_sentiment(
        data=review_data,
        resources=resources,
        batch_size=4,
        max_length=32,
        max_samples=15,
        random_state=42,
    )

    assert results[
        "source_number_of_reviews"
    ] == 30

    assert results[
        "number_of_reviews"
    ] == 15

    assert results[
        "sample_limit"
    ] == 15

    assert results[
        "sampled"
    ] is True

    assert len(
        results["predictions"]
    ) == 15

    assert int(
        results[
            "metrics"
        ].iloc[0][
            "ReviewsProcessed"
        ]
    ) == 15

    sample_distribution = (
        results[
            "predictions"
        ][
            "RatingSentiment"
        ]
        .value_counts()
        .to_dict()
    )

    assert sample_distribution == {
        "Negatywny": 5,
        "Neutralny": 5,
        "Pozytywny": 5,
    }

    assert results[
        "predictions"
    ][
        "TransformerStars"
    ].unique().tolist() == [5]

    assert results[
        "model_license"
    ] == "Test license"

    assert results[
        "model_language"
    ] == "Portuguese"