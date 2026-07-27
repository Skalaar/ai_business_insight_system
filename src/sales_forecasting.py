from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import Holt


MODEL_NAMES = [
    "Prognoza naiwna",
    "Średnia ruchoma 4 tygodnie",
    "Trend liniowy",
    "Metoda Holta",
]


def prepare_weekly_sales_series(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agreguje dane sprzedażowe do tygodniowego szeregu przychodów.

    Każdy rekord wynikowy reprezentuje przychód osiągnięty
    w tygodniu kończącym się w niedzielę.
    """
    required_columns = [
        "InvoiceDate",
        "TotalPrice",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych do prognozowania sprzedaży: "
            f"{missing_columns}"
        )

    prepared_data = data[
        required_columns
    ].dropna().copy()

    prepared_data["InvoiceDate"] = pd.to_datetime(
        prepared_data["InvoiceDate"],
        errors="coerce",
    )

    prepared_data["TotalPrice"] = pd.to_numeric(
        prepared_data["TotalPrice"],
        errors="coerce",
    )

    prepared_data = prepared_data.dropna(
        subset=["InvoiceDate", "TotalPrice"]
    )

    prepared_data = prepared_data[
        prepared_data["TotalPrice"] >= 0
    ]

    if prepared_data.empty:
        raise ValueError(
            "Po przygotowaniu danych nie pozostały obserwacje "
            "nadające się do prognozowania."
        )

    weekly_revenue = (
        prepared_data
        .set_index("InvoiceDate")["TotalPrice"]
        .resample("W-SUN")
        .sum()
        .astype(float)
    )

    weekly_revenue = weekly_revenue.asfreq(
        "W-SUN",
        fill_value=0.0,
    )

    weekly_data = (
        weekly_revenue
        .rename("Revenue")
        .reset_index()
        .rename(
            columns={
                "InvoiceDate": "PeriodEnd",
            }
        )
    )

    if len(weekly_data) < 16:
        raise ValueError(
            "Do walidacji prognoz wymagane jest co najmniej "
            "16 tygodni danych."
        )

    return weekly_data


def _naive_forecast(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Przyjmuje ostatnią znaną wartość jako prognozę
    dla kolejnych okresów.
    """
    if len(history) == 0:
        raise ValueError(
            "Historia szeregu czasowego nie może być pusta."
        )

    return np.repeat(
        float(history[-1]),
        horizon,
    )


def _moving_average_forecast(
    history: np.ndarray,
    horizon: int,
    window: int = 4,
) -> np.ndarray:
    """
    Prognozuje na podstawie średniej z ostatnich tygodni.

    Przy prognozie wielookresowej wartości są obliczane
    rekurencyjnie.
    """
    if len(history) == 0:
        raise ValueError(
            "Historia szeregu czasowego nie może być pusta."
        )

    recursive_history = [
        float(value)
        for value in history
    ]

    forecasts = []

    for _ in range(horizon):
        effective_window = min(
            window,
            len(recursive_history),
        )

        prediction = float(
            np.mean(
                recursive_history[
                    -effective_window:
                ]
            )
        )

        prediction = max(
            0.0,
            prediction,
        )

        forecasts.append(prediction)
        recursive_history.append(prediction)

    return np.asarray(
        forecasts,
        dtype=float,
    )


def _linear_trend_forecast(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Dopasowuje regresję liniową do numeru okresu
    i wartości sprzedaży.
    """
    if len(history) < 2:
        return _naive_forecast(
            history=history,
            horizon=horizon,
        )

    time_index = np.arange(
        len(history)
    ).reshape(-1, 1)

    model = LinearRegression()

    model.fit(
        time_index,
        history,
    )

    future_index = np.arange(
        len(history),
        len(history) + horizon,
    ).reshape(-1, 1)

    forecasts = model.predict(
        future_index
    )

    return np.maximum(
        forecasts,
        0.0,
    )


def _holt_forecast(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Prognozuje poziom i trend szeregu metodą Holta
    z trendem tłumionym.
    """
    if len(history) < 4:
        return _linear_trend_forecast(
            history=history,
            horizon=horizon,
        )

    try:
        fitted_model = Holt(
            history,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(
            optimized=True,
        )

        forecasts = fitted_model.forecast(
            horizon
        )

        return np.maximum(
            np.asarray(
                forecasts,
                dtype=float,
            ),
            0.0,
        )

    except Exception:
        return _linear_trend_forecast(
            history=history,
            horizon=horizon,
        )


def _get_forecasting_functions() -> dict[
    str,
    Callable[[np.ndarray, int], np.ndarray],
]:
    """
    Zwraca mapowanie nazw modeli na funkcje prognostyczne.
    """
    return {
        "Prognoza naiwna": _naive_forecast,
        "Średnia ruchoma 4 tygodnie": (
            _moving_average_forecast
        ),
        "Trend liniowy": _linear_trend_forecast,
        "Metoda Holta": _holt_forecast,
    }


def _calculate_forecast_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict:
    """
    Oblicza metryki błędu prognoz.

    Zwracane miary:
    - MAE,
    - RMSE,
    - MAPE,
    - sMAPE,
    - WAPE,
    - obciążenie prognozy.
    """
    actual_values = np.asarray(
        actual_values,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=float,
    )

    errors = predicted_values - actual_values

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_values,
                predicted_values,
            )
        )
    )

    non_zero_actuals = (
        np.abs(actual_values) > 1e-12
    )

    if non_zero_actuals.any():
        mape = float(
            np.mean(
                np.abs(
                    errors[non_zero_actuals]
                    / actual_values[
                        non_zero_actuals
                    ]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    smape_denominator = (
        np.abs(actual_values)
        + np.abs(predicted_values)
    )

    valid_smape = (
        smape_denominator > 1e-12
    )

    if valid_smape.any():
        smape = float(
            np.mean(
                2
                * np.abs(
                    errors[valid_smape]
                )
                / smape_denominator[
                    valid_smape
                ]
            )
            * 100
        )
    else:
        smape = 0.0

    total_actual = float(
        np.sum(
            np.abs(actual_values)
        )
    )

    if total_actual > 1e-12:
        wape = float(
            np.sum(
                np.abs(errors)
            )
            / total_actual
            * 100
        )
    else:
        wape = np.nan

    bias = float(
        np.mean(errors)
    )

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "WAPE": wape,
        "Bias": bias,
    }


def _create_walk_forward_validation(
    weekly_data: pd.DataFrame,
    test_fraction: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
    int,
]:
    """
    Przeprowadza walidację kroczącą z rozszerzającym się
    oknem treningowym.

    Dla każdego okresu testowego model korzysta wyłącznie
    z obserwacji wcześniejszych.
    """
    if not 0.10 <= test_fraction <= 0.50:
        raise ValueError(
            "Udział zbioru testowego powinien znajdować się "
            "w przedziale od 0,10 do 0,50."
        )

    values = weekly_data[
        "Revenue"
    ].to_numpy(
        dtype=float
    )

    number_of_observations = len(values)

    test_size = max(
        5,
        int(
            round(
                number_of_observations
                * test_fraction
            )
        ),
    )

    test_size = min(
        test_size,
        number_of_observations - 10,
    )

    initial_train_size = (
        number_of_observations
        - test_size
    )

    if initial_train_size < 10:
        raise ValueError(
            "Zbiór treningowy musi zawierać "
            "co najmniej 10 tygodni."
        )

    forecasting_functions = (
        _get_forecasting_functions()
    )

    prediction_rows = []
    metric_rows = []

    for model_name, forecast_function in (
        forecasting_functions.items()
    ):
        model_predictions = []
        model_actuals = []

        for test_position in range(
            initial_train_size,
            number_of_observations,
        ):
            training_history = values[
                :test_position
            ]

            prediction = float(
                forecast_function(
                    training_history,
                    1,
                )[0]
            )

            actual_value = float(
                values[test_position]
            )

            prediction_rows.append(
                {
                    "PeriodEnd": weekly_data.iloc[
                        test_position
                    ]["PeriodEnd"],
                    "Model": model_name,
                    "ActualRevenue": actual_value,
                    "ForecastRevenue": prediction,
                    "Error": (
                        prediction
                        - actual_value
                    ),
                    "AbsoluteError": abs(
                        prediction
                        - actual_value
                    ),
                }
            )

            model_predictions.append(
                prediction
            )

            model_actuals.append(
                actual_value
            )

        model_metrics = (
            _calculate_forecast_metrics(
                actual_values=np.asarray(
                    model_actuals,
                    dtype=float,
                ),
                predicted_values=np.asarray(
                    model_predictions,
                    dtype=float,
                ),
            )
        )

        metric_rows.append(
            {
                "Model": model_name,
                **model_metrics,
                "ValidationPeriods": test_size,
            }
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics = (
        metrics
        .sort_values(
            by=[
                "RMSE",
                "MAE",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    metrics.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(metrics) + 1,
        ),
    )

    return (
        predictions,
        metrics,
        initial_train_size,
        test_size,
    )


def _forecast_future_with_model(
    model_name: str,
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Generuje prognozę przyszłą za pomocą wskazanego modelu.
    """
    forecasting_functions = (
        _get_forecasting_functions()
    )

    if model_name not in forecasting_functions:
        raise ValueError(
            f"Nieznany model prognostyczny: {model_name}"
        )

    return forecasting_functions[
        model_name
    ](
        history,
        horizon,
    )


def build_sales_forecast_analysis(
    data: pd.DataFrame,
    test_fraction: float = 0.25,
    forecast_horizon: int = 8,
) -> dict:
    """
    Przygotowuje pełną analizę prognostyczną sprzedaży.

    Etapy:
    - agregacja tygodniowa,
    - walidacja krocząca,
    - porównanie czterech modeli,
    - wybór modelu o najniższym RMSE,
    - prognoza przyszłych tygodni.
    """
    if not 1 <= forecast_horizon <= 26:
        raise ValueError(
            "Horyzont prognozy powinien wynosić "
            "od 1 do 26 tygodni."
        )

    weekly_data = prepare_weekly_sales_series(
        data
    )

    (
        validation_predictions,
        metrics,
        initial_train_size,
        test_size,
    ) = _create_walk_forward_validation(
        weekly_data=weekly_data,
        test_fraction=test_fraction,
    )

    best_model_name = str(
        metrics.iloc[0]["Model"]
    )

    complete_history = weekly_data[
        "Revenue"
    ].to_numpy(
        dtype=float
    )

    future_values = (
        _forecast_future_with_model(
            model_name=best_model_name,
            history=complete_history,
            horizon=forecast_horizon,
        )
    )

    last_period = pd.Timestamp(
        weekly_data["PeriodEnd"].max()
    )

    future_dates = pd.date_range(
        start=last_period
        + pd.offsets.Week(weekday=6),
        periods=forecast_horizon,
        freq="W-SUN",
    )

    future_forecast = pd.DataFrame(
        {
            "PeriodEnd": future_dates,
            "ForecastRevenue": future_values,
            "Model": best_model_name,
        }
    )

    best_model_predictions = (
        validation_predictions[
            validation_predictions["Model"]
            == best_model_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    return {
        "weekly_data": weekly_data,
        "validation_predictions": (
            validation_predictions
        ),
        "best_model_predictions": (
            best_model_predictions
        ),
        "metrics": metrics,
        "best_model_name": best_model_name,
        "future_forecast": future_forecast,
        "number_of_observations": len(
            weekly_data
        ),
        "initial_train_size": (
            initial_train_size
        ),
        "test_size": test_size,
        "test_fraction": test_fraction,
        "forecast_horizon": forecast_horizon,
        "frequency": "Tygodniowa",
    }