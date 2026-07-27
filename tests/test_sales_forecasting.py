from __future__ import annotations

import pandas as pd

from src.sales_forecasting import (
    build_sales_forecast_analysis,
    prepare_weekly_sales_series,
)


def test_weekly_sales_preparation(
    sales_data: pd.DataFrame,
) -> None:
    weekly_data = prepare_weekly_sales_series(
        sales_data
    )

    assert not weekly_data.empty

    assert list(
        weekly_data.columns
    ) == [
        "PeriodEnd",
        "Revenue",
    ]

    assert len(weekly_data) >= 16

    assert (
        weekly_data["Revenue"] >= 0
    ).all()


def test_sales_forecasting_analysis(
    sales_data: pd.DataFrame,
) -> None:
    results = build_sales_forecast_analysis(
        data=sales_data,
        test_fraction=0.25,
        forecast_horizon=8,
    )

    metrics = results["metrics"]

    assert len(metrics) == 4

    assert set(
        metrics["Model"]
    ) == {
        "Prognoza naiwna",
        "Średnia ruchoma 4 tygodnie",
        "Trend liniowy",
        "Metoda Holta",
    }

    assert results[
        "best_model_name"
    ] in set(metrics["Model"])

    assert len(
        results["future_forecast"]
    ) == 8

    assert (
        results["future_forecast"][
            "ForecastRevenue"
        ]
        >= 0
    ).all()

    assert (
        metrics["MAE"] >= 0
    ).all()

    assert (
        metrics["RMSE"] >= 0
    ).all()