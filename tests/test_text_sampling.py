from __future__ import annotations

import pandas as pd
import pytest

from src.text_sampling import (
    create_stratified_text_sample,
)


@pytest.fixture
def imbalanced_text_data() -> pd.DataFrame:
    rows = []

    class_configuration = {
        "Pozytywny": 50,
        "Negatywny": 30,
        "Neutralny": 20,
    }

    review_id = 1

    for sentiment, number_of_rows in (
        class_configuration.items()
    ):
        for row_number in range(
            number_of_rows
        ):
            rows.append(
                {
                    "ReviewID": review_id,
                    "CleanReviewText": (
                        f"opinia {sentiment} "
                        f"{row_number}"
                    ),
                    "RatingSentiment": sentiment,
                }
            )

            review_id += 1

    return pd.DataFrame(
        rows
    )


def test_sample_preserves_expected_class_proportions(
    imbalanced_text_data: pd.DataFrame,
) -> None:
    sample = create_stratified_text_sample(
        data=imbalanced_text_data,
        max_samples=30,
        random_state=42,
    )

    class_counts = (
        sample["RatingSentiment"]
        .value_counts()
        .to_dict()
    )

    assert len(sample) == 30

    assert class_counts == {
        "Pozytywny": 15,
        "Negatywny": 9,
        "Neutralny": 6,
    }


def test_sampling_is_reproducible(
    imbalanced_text_data: pd.DataFrame,
) -> None:
    first_sample = create_stratified_text_sample(
        data=imbalanced_text_data,
        max_samples=30,
        random_state=42,
    )

    second_sample = create_stratified_text_sample(
        data=imbalanced_text_data,
        max_samples=30,
        random_state=42,
    )

    assert (
        first_sample["ReviewID"].tolist()
        == second_sample["ReviewID"].tolist()
    )


def test_full_dataset_is_returned_when_limit_is_larger(
    imbalanced_text_data: pd.DataFrame,
) -> None:
    sample = create_stratified_text_sample(
        data=imbalanced_text_data,
        max_samples=200,
        random_state=42,
    )

    assert len(sample) == len(
        imbalanced_text_data
    )

    assert sample[
        "ReviewID"
    ].tolist() == imbalanced_text_data[
        "ReviewID"
    ].tolist()


def test_random_sampling_without_target_column(
    imbalanced_text_data: pd.DataFrame,
) -> None:
    data_without_target = (
        imbalanced_text_data
        .drop(
            columns=[
                "RatingSentiment",
            ]
        )
    )

    sample = create_stratified_text_sample(
        data=data_without_target,
        max_samples=25,
        random_state=42,
    )

    assert len(sample) == 25
    assert sample["ReviewID"].nunique() == 25


def test_sample_limit_cannot_be_smaller_than_classes(
    imbalanced_text_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="liczbie klas",
    ):
        create_stratified_text_sample(
            data=imbalanced_text_data,
            max_samples=2,
            random_state=42,
        )