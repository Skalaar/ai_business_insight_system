from __future__ import annotations

import pandas as pd

from src.advanced_recommendations import (
    build_advanced_decision_support,
)
from src.sales_forecasting import (
    build_sales_forecast_analysis,
)
from src.topic_modeling import (
    build_topic_analysis,
)


def test_advanced_decision_support(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
) -> None:
    topic_results = build_topic_analysis(
        data=review_data,
        number_of_topics=3,
        top_terms_per_topic=6,
        random_state=42,
    )

    forecast_results = (
        build_sales_forecast_analysis(
            data=sales_data,
            test_fraction=0.25,
            forecast_horizon=8,
        )
    )

    decision_results = (
        build_advanced_decision_support(
            sales_data=sales_data,
            review_data=review_data,
            topic_assignments=(
                topic_results[
                    "review_assignments"
                ]
            ),
            weekly_data=(
                forecast_results[
                    "weekly_data"
                ]
            ),
            future_forecast=(
                forecast_results[
                    "future_forecast"
                ]
            ),
            trend_window_weeks=8,
        )
    )

    product_matrix = decision_results[
        "product_matrix"
    ]

    recommendations = decision_results[
        "recommendations"
    ]

    assert len(product_matrix) == 6

    assert len(recommendations) == 6

    assert product_matrix[
        "ProductName"
    ].nunique() == 6

    assert product_matrix[
        "RiskScore"
    ].between(
        0,
        100,
    ).all()

    assert product_matrix[
        "OpportunityScore"
    ].between(
        0,
        100,
    ).all()

    assert recommendations[
        "Priority"
    ].isin(
        [
            "Wysoki",
            "Średni",
            "Niski",
        ]
    ).all()

    assert decision_results[
        "executive_summary"
    ]["ProductsAnalyzed"] == 6