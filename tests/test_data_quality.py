from __future__ import annotations

import pandas as pd

from src.data_quality import (
    build_data_quality_report,
)


def test_data_quality_report_for_valid_data(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
) -> None:
    results = build_data_quality_report(
        sales_data=sales_data,
        review_data=review_data,
        minimum_review_length=10,
    )

    summary = results["summary"]
    checks = results["quality_checks"]

    assert len(checks) >= 10

    assert (
        0
        <= summary["OverallQualityScore"]
        <= 100
    )

    assert summary["QualityStatus"] in {
        "Bardzo wysoka",
        "Dobra",
        "Ostrzegawcza",
        "Krytyczna",
    }

    assert checks[
        "IssueRate"
    ].between(
        0,
        1,
    ).all()

    assert checks[
        "PassRate"
    ].between(
        0,
        1,
    ).all()


def test_data_quality_detects_invalid_records(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
) -> None:
    broken_sales = sales_data.copy()
    broken_reviews = review_data.copy()

    broken_sales.loc[
        broken_sales.index[0],
        "Quantity",
    ] = -5

    broken_sales.loc[
        broken_sales.index[1],
        "UnitPrice",
    ] = 0

    broken_reviews.loc[
        broken_reviews.index[0],
        "Rating",
    ] = 8

    broken_reviews.loc[
        broken_reviews.index[1],
        "ReviewText",
    ] = ""

    results = build_data_quality_report(
        sales_data=broken_sales,
        review_data=broken_reviews,
        minimum_review_length=10,
    )

    assert results[
        "summary"
    ]["FailedChecks"] >= 4

    assert not results[
        "failed_checks"
    ].empty

    assert results[
        "summary"
    ]["OverallQualityScore"] < 100