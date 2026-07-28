from __future__ import annotations

import pandas as pd

from src.drift_monitoring import (
    build_drift_monitoring_analysis,
)


def test_drift_monitoring_analysis(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
) -> None:
    results = build_drift_monitoring_analysis(
        sales_data=sales_data,
        review_data=review_data,
        window_weeks=8,
        top_terms=25,
        text_language="multilingual",
    )

    components = results[
        "drift_components"
    ]

    summary = results["summary"]

    assert len(components) == 9

    assert components[
        "DriftScore"
    ].between(
        0,
        1,
    ).all()

    assert components[
        "Severity"
    ].isin(
        [
            "Niski",
            "Średni",
            "Wysoki",
        ]
    ).all()

    assert (
        0
        <= summary["OverallDriftScore"]
        <= 1
    )

    assert summary[
        "OverallSeverity"
    ] in {
        "Niski",
        "Średni",
        "Wysoki",
    }

    assert summary[
        "ReferenceSalesRecords"
    ] > 0

    assert summary[
        "CurrentSalesRecords"
    ] > 0

    assert summary[
        "ReferenceReviews"
    ] > 0

    assert summary[
        "CurrentReviews"
    ] > 0

    assert results["text_language"] == (
        "multilingual"
    )

    vocabulary = results[
        "vocabulary_comparison"
    ]

    assert not vocabulary.empty

    for forbidden_term in [
        "the",
        "and",
        "que",
        "com",
        "produto",
    ]:
        assert forbidden_term not in set(
            vocabulary["Term"]
    )