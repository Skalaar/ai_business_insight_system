from __future__ import annotations

from collections import Counter
import re
import unicodedata

import numpy as np
import pandas as pd

from src.text_resources import (
    get_stopwords,
    normalize_language_code,
)


def _jensen_shannon_divergence(
    reference_values: np.ndarray,
    current_values: np.ndarray,
) -> float:
    """
    Oblicza dywergencję Jensena-Shannona pomiędzy
    dwoma rozkładami prawdopodobieństwa.

    Wynik mieści się w przedziale od 0 do 1:
    - 0 oznacza identyczne rozkłady,
    - wartości bliższe 1 oznaczają większą zmianę.
    """
    reference_values = np.asarray(
        reference_values,
        dtype=float,
    )

    current_values = np.asarray(
        current_values,
        dtype=float,
    )

    reference_sum = reference_values.sum()
    current_sum = current_values.sum()

    if reference_sum <= 0 or current_sum <= 0:
        return 0.0

    reference_distribution = (
        reference_values / reference_sum
    )

    current_distribution = (
        current_values / current_sum
    )

    mean_distribution = (
        reference_distribution
        + current_distribution
    ) / 2

    epsilon = 1e-12

    reference_safe = np.clip(
        reference_distribution,
        epsilon,
        None,
    )

    current_safe = np.clip(
        current_distribution,
        epsilon,
        None,
    )

    mean_safe = np.clip(
        mean_distribution,
        epsilon,
        None,
    )

    reference_kl = np.sum(
        reference_distribution
        * np.log2(
            reference_safe / mean_safe
        )
    )

    current_kl = np.sum(
        current_distribution
        * np.log2(
            current_safe / mean_safe
        )
    )

    divergence = (
        reference_kl + current_kl
    ) / 2

    return float(
        np.clip(
            divergence,
            0.0,
            1.0,
        )
    )


def _calculate_percentage_change(
    reference_value: float,
    current_value: float,
) -> float:
    """
    Oblicza procentową zmianę wartości.
    """
    reference_value = float(
        reference_value
    )

    current_value = float(
        current_value
    )

    if abs(reference_value) < 1e-12:
        if abs(current_value) < 1e-12:
            return 0.0

        return 100.0

    return (
        (
            current_value
            - reference_value
        )
        / abs(reference_value)
        * 100
    )


def _change_to_drift_score(
    percentage_change: float,
) -> float:
    """
    Przekształca bezwzględną zmianę procentową
    na wynik driftu od 0 do 1.
    """
    return float(
        np.clip(
            abs(percentage_change)
            / 100,
            0.0,
            1.0,
        )
    )


def _assign_drift_severity(
    drift_score: float,
) -> str:
    """
    Przypisuje poziom nasilenia driftu.
    """
    if drift_score >= 0.25:
        return "Wysoki"

    if drift_score >= 0.10:
        return "Średni"

    return "Niski"


def _create_equal_time_windows(
    data: pd.DataFrame,
    date_column: str,
    window_weeks: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Dzieli dane na dwa bezpośrednio następujące
    po sobie okresy o równej długości.
    """
    if window_weeks < 2:
        raise ValueError(
            "Okno analizy musi obejmować "
            "co najmniej dwa tygodnie."
        )

    prepared = data.copy()

    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[date_column]
    )

    if prepared.empty:
        raise ValueError(
            f"Brak poprawnych dat w kolumnie {date_column}."
        )

    current_end = prepared[
        date_column
    ].max()

    current_start = (
        current_end
        - pd.Timedelta(
            weeks=window_weeks
        )
    )

    reference_start = (
        current_start
        - pd.Timedelta(
            weeks=window_weeks
        )
    )

    reference_data = prepared[
        (
            prepared[date_column]
            > reference_start
        )
        & (
            prepared[date_column]
            <= current_start
        )
    ].copy()

    current_data = prepared[
        (
            prepared[date_column]
            > current_start
        )
        & (
            prepared[date_column]
            <= current_end
        )
    ].copy()

    window_information = {
        "ReferenceStart": reference_start,
        "ReferenceEnd": current_start,
        "CurrentStart": current_start,
        "CurrentEnd": current_end,
        "WindowWeeks": window_weeks,
    }

    return (
        reference_data,
        current_data,
        window_information,
    )


def _build_distribution_comparison(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    category_column: str,
    value_column: str | None = None,
) -> pd.DataFrame:
    """
    Tworzy tabelę porównania rozkładów kategorii.

    Jeżeli podano value_column, wartości są sumowane.
    W przeciwnym razie liczona jest liczba rekordów.
    """
    if value_column is None:
        reference_values = (
            reference_data[
                category_column
            ]
            .value_counts()
            .rename("ReferenceValue")
        )

        current_values = (
            current_data[
                category_column
            ]
            .value_counts()
            .rename("CurrentValue")
        )
    else:
        reference_values = (
            reference_data
            .groupby(category_column)[
                value_column
            ]
            .sum()
            .rename("ReferenceValue")
        )

        current_values = (
            current_data
            .groupby(category_column)[
                value_column
            ]
            .sum()
            .rename("CurrentValue")
        )

    comparison = pd.concat(
        [
            reference_values,
            current_values,
        ],
        axis=1,
    ).fillna(0.0)

    reference_total = comparison[
        "ReferenceValue"
    ].sum()

    current_total = comparison[
        "CurrentValue"
    ].sum()

    comparison["ReferenceShare"] = np.where(
        reference_total > 0,
        comparison["ReferenceValue"]
        / reference_total,
        0.0,
    )

    comparison["CurrentShare"] = np.where(
        current_total > 0,
        comparison["CurrentValue"]
        / current_total,
        0.0,
    )

    comparison["ShareChange"] = (
        comparison["CurrentShare"]
        - comparison["ReferenceShare"]
    )

    return (
        comparison
        .reset_index()
        .rename(
            columns={
                category_column: "Category",
                "index": "Category",
            }
        )
        .sort_values(
            by="CurrentValue",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _tokenize_text(
    text: str,
    stopwords: set[str],
) -> list[str]:
    """
    Przekształca tekst na tokeny z zachowaniem
    liter diakrytycznych i usuwa wskazane stopwords.
    """
    normalized_text = unicodedata.normalize(
        "NFKC",
        str(text).lower(),
    )

    tokens = re.findall(
        r"[^\W\d_]{3,}",
        normalized_text,
        flags=re.UNICODE,
    )

    return [
        token
        for token in tokens
        if token not in stopwords
    ]


def _build_vocabulary_comparison(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    text_column: str,
    top_terms: int,
    stopwords: set[str],
) -> pd.DataFrame:
    """
    Porównuje częstość występowania terminów
    w dwóch okresach.
    """
    reference_counter = Counter()

    current_counter = Counter()

    for text in reference_data[
        text_column
    ].dropna():
        reference_counter.update(
            _tokenize_text(
                text=text,
                stopwords=stopwords,
            )
        )

    for text in current_data[
        text_column
    ].dropna():
        current_counter.update(
            _tokenize_text(
                text=text,
                stopwords=stopwords,
            )
        )

    combined_counter = (
        reference_counter
        + current_counter
    )

    selected_terms = [
        term
        for term, _
        in combined_counter.most_common(
            top_terms
        )
    ]

    rows = []

    reference_total = sum(
        reference_counter.values()
    )

    current_total = sum(
        current_counter.values()
    )

    for term in selected_terms:
        reference_count = (
            reference_counter[term]
        )

        current_count = (
            current_counter[term]
        )

        reference_share = (
            reference_count
            / reference_total
            if reference_total > 0
            else 0.0
        )

        current_share = (
            current_count
            / current_total
            if current_total > 0
            else 0.0
        )

        rows.append(
            {
                "Term": term,
                "ReferenceCount": (
                    reference_count
                ),
                "CurrentCount": (
                    current_count
                ),
                "ReferenceShare": (
                    reference_share
                ),
                "CurrentShare": (
                    current_share
                ),
                "ShareChange": (
                    current_share
                    - reference_share
                ),
            }
        )

    return pd.DataFrame(rows)


def _calculate_average_order_value(
    data: pd.DataFrame,
) -> float:
    """
    Oblicza średnią wartość transakcji.
    """
    if (
        data.empty
        or "InvoiceNo" not in data.columns
    ):
        return 0.0

    order_values = (
        data
        .groupby("InvoiceNo")[
            "TotalPrice"
        ]
        .sum()
    )

    if order_values.empty:
        return 0.0

    return float(
        order_values.mean()
    )


def _calculate_weekly_revenue_average(
    data: pd.DataFrame,
) -> float:
    """
    Oblicza średni tygodniowy przychód.
    """
    if data.empty:
        return 0.0

    weekly_revenue = (
        data
        .set_index("InvoiceDate")[
            "TotalPrice"
        ]
        .resample("W-SUN")
        .sum()
    )

    if weekly_revenue.empty:
        return 0.0

    return float(
        weekly_revenue.mean()
    )


def _add_component(
    component_rows: list[dict],
    area: str,
    metric: str,
    drift_score: float,
    reference_value: float | str,
    current_value: float | str,
    change_percentage: float | None,
    interpretation: str,
) -> None:
    """
    Dodaje element do tabeli wyników driftu.
    """
    component_rows.append(
        {
            "Area": area,
            "Metric": metric,
            "DriftScore": float(
                np.clip(
                    drift_score,
                    0.0,
                    1.0,
                )
            ),
            "Severity": (
                _assign_drift_severity(
                    drift_score
                )
            ),
            "ReferenceValue": (
                reference_value
            ),
            "CurrentValue": current_value,
            "ChangePct": (
                change_percentage
            ),
            "Interpretation": (
                interpretation
            ),
        }
    )


def build_drift_monitoring_analysis(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
    window_weeks: int = 8,
    top_terms: int = 40,
    text_language: str | None = "multilingual",
) -> dict:
    """
    Buduje analizę driftu danych sprzedażowych i tekstowych.

    Porównywane są dwa kolejne okresy o tej samej długości.
    """

    normalized_language = normalize_language_code(
        text_language
    )

    text_stopwords = set(
        get_stopwords(
            language=normalized_language,
            include_domain=True,
            preserve_negations=True,
        )
    )

    required_sales_columns = [
        "InvoiceDate",
        "TotalPrice",
        "Description",
        "Country",
    ]

    missing_sales_columns = [
        column
        for column in required_sales_columns
        if column not in sales_data.columns
    ]

    if missing_sales_columns:
        raise ValueError(
            "Brakuje kolumn sprzedażowych: "
            f"{missing_sales_columns}"
        )

    required_review_columns = [
        "ReviewDate",
        "Rating",
        "RatingSentiment",
        "ReviewText",
    ]

    missing_review_columns = [
        column
        for column in required_review_columns
        if column not in review_data.columns
    ]

    if missing_review_columns:
        raise ValueError(
            "Brakuje kolumn opinii: "
            f"{missing_review_columns}"
        )

    prepared_sales = sales_data.copy()

    prepared_sales["InvoiceDate"] = (
        pd.to_datetime(
            prepared_sales["InvoiceDate"],
            errors="coerce",
        )
    )

    prepared_sales["TotalPrice"] = (
        pd.to_numeric(
            prepared_sales["TotalPrice"],
            errors="coerce",
        )
    )

    prepared_sales = (
        prepared_sales
        .dropna(
            subset=[
                "InvoiceDate",
                "TotalPrice",
                "Description",
                "Country",
            ]
        )
    )

    prepared_reviews = review_data.copy()

    prepared_reviews["ReviewDate"] = (
        pd.to_datetime(
            prepared_reviews["ReviewDate"],
            errors="coerce",
        )
    )

    prepared_reviews["Rating"] = pd.to_numeric(
        prepared_reviews["Rating"],
        errors="coerce",
    )

    prepared_reviews = (
        prepared_reviews
        .dropna(
            subset=[
                "ReviewDate",
                "Rating",
                "RatingSentiment",
                "ReviewText",
            ]
        )
    )

    (
        reference_sales,
        current_sales,
        sales_windows,
    ) = _create_equal_time_windows(
        data=prepared_sales,
        date_column="InvoiceDate",
        window_weeks=window_weeks,
    )

    (
        reference_reviews,
        current_reviews,
        review_windows,
    ) = _create_equal_time_windows(
        data=prepared_reviews,
        date_column="ReviewDate",
        window_weeks=window_weeks,
    )

    if (
        len(reference_sales) < 20
        or len(current_sales) < 20
    ):
        raise ValueError(
            "Każdy okres sprzedażowy powinien zawierać "
            "co najmniej 20 rekordów."
        )

    if (
        len(reference_reviews) < 10
        or len(current_reviews) < 10
    ):
        raise ValueError(
            "Każdy okres opinii powinien zawierać "
            "co najmniej 10 opinii."
        )

    product_distribution = (
        _build_distribution_comparison(
            reference_data=reference_sales,
            current_data=current_sales,
            category_column="Description",
            value_column="TotalPrice",
        )
    )

    country_distribution = (
        _build_distribution_comparison(
            reference_data=reference_sales,
            current_data=current_sales,
            category_column="Country",
            value_column="TotalPrice",
        )
    )

    sentiment_distribution = (
        _build_distribution_comparison(
            reference_data=reference_reviews,
            current_data=current_reviews,
            category_column="RatingSentiment",
        )
    )

    rating_distribution = (
        _build_distribution_comparison(
            reference_data=reference_reviews,
            current_data=current_reviews,
            category_column="Rating",
        )
    )

    vocabulary_comparison = (
        _build_vocabulary_comparison(
            reference_data=reference_reviews,
            current_data=current_reviews,
            text_column="ReviewText",
            top_terms=top_terms,
            stopwords=text_stopwords,
        )
    )

    component_rows = []

    product_divergence = (
        _jensen_shannon_divergence(
            product_distribution[
                "ReferenceShare"
            ].to_numpy(),
            product_distribution[
                "CurrentShare"
            ].to_numpy(),
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Sprzedaż",
        metric="Struktura przychodu według produktów",
        drift_score=product_divergence,
        reference_value="Rozkład referencyjny",
        current_value="Rozkład bieżący",
        change_percentage=None,
        interpretation=(
            "Zmiana udziału poszczególnych produktów "
            "w całkowitym przychodzie."
        ),
    )

    country_divergence = (
        _jensen_shannon_divergence(
            country_distribution[
                "ReferenceShare"
            ].to_numpy(),
            country_distribution[
                "CurrentShare"
            ].to_numpy(),
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Sprzedaż",
        metric="Struktura przychodu według krajów",
        drift_score=country_divergence,
        reference_value="Rozkład referencyjny",
        current_value="Rozkład bieżący",
        change_percentage=None,
        interpretation=(
            "Zmiana udziału poszczególnych krajów "
            "w całkowitym przychodzie."
        ),
    )

    reference_weekly_revenue = (
        _calculate_weekly_revenue_average(
            reference_sales
        )
    )

    current_weekly_revenue = (
        _calculate_weekly_revenue_average(
            current_sales
        )
    )

    weekly_revenue_change = (
        _calculate_percentage_change(
            reference_weekly_revenue,
            current_weekly_revenue,
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Sprzedaż",
        metric="Średni przychód tygodniowy",
        drift_score=(
            _change_to_drift_score(
                weekly_revenue_change
            )
        ),
        reference_value=(
            reference_weekly_revenue
        ),
        current_value=current_weekly_revenue,
        change_percentage=weekly_revenue_change,
        interpretation=(
            "Zmiana przeciętnego poziomu "
            "tygodniowych przychodów."
        ),
    )

    reference_order_value = (
        _calculate_average_order_value(
            reference_sales
        )
    )

    current_order_value = (
        _calculate_average_order_value(
            current_sales
        )
    )

    order_value_change = (
        _calculate_percentage_change(
            reference_order_value,
            current_order_value,
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Sprzedaż",
        metric="Średnia wartość zamówienia",
        drift_score=(
            _change_to_drift_score(
                order_value_change
            )
        ),
        reference_value=reference_order_value,
        current_value=current_order_value,
        change_percentage=order_value_change,
        interpretation=(
            "Zmiana przeciętnej wartości "
            "pojedynczej transakcji."
        ),
    )

    sentiment_divergence = (
        _jensen_shannon_divergence(
            sentiment_distribution[
                "ReferenceShare"
            ].to_numpy(),
            sentiment_distribution[
                "CurrentShare"
            ].to_numpy(),
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Opinie",
        metric="Rozkład sentymentu",
        drift_score=sentiment_divergence,
        reference_value="Rozkład referencyjny",
        current_value="Rozkład bieżący",
        change_percentage=None,
        interpretation=(
            "Zmiana udziału opinii pozytywnych, "
            "neutralnych i negatywnych."
        ),
    )

    rating_divergence = (
        _jensen_shannon_divergence(
            rating_distribution[
                "ReferenceShare"
            ].to_numpy(),
            rating_distribution[
                "CurrentShare"
            ].to_numpy(),
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Opinie",
        metric="Rozkład ocen gwiazdkowych",
        drift_score=rating_divergence,
        reference_value="Rozkład referencyjny",
        current_value="Rozkład bieżący",
        change_percentage=None,
        interpretation=(
            "Zmiana struktury ocen od jednej "
            "do pięciu gwiazdek."
        ),
    )

    reference_average_rating = float(
        reference_reviews["Rating"].mean()
    )

    current_average_rating = float(
        current_reviews["Rating"].mean()
    )

    rating_change = (
        _calculate_percentage_change(
            reference_average_rating,
            current_average_rating,
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Opinie",
        metric="Średnia ocena",
        drift_score=(
            _change_to_drift_score(
                rating_change
            )
        ),
        reference_value=(
            reference_average_rating
        ),
        current_value=current_average_rating,
        change_percentage=rating_change,
        interpretation=(
            "Zmiana przeciętnej oceny klientów."
        ),
    )

    reference_text_length = float(
        reference_reviews["ReviewText"]
        .astype(str)
        .str.len()
        .mean()
    )

    current_text_length = float(
        current_reviews["ReviewText"]
        .astype(str)
        .str.len()
        .mean()
    )

    text_length_change = (
        _calculate_percentage_change(
            reference_text_length,
            current_text_length,
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Opinie",
        metric="Średnia długość opinii",
        drift_score=(
            _change_to_drift_score(
                text_length_change
            )
        ),
        reference_value=(
            reference_text_length
        ),
        current_value=current_text_length,
        change_percentage=text_length_change,
        interpretation=(
            "Zmiana przeciętnej liczby znaków "
            "w opiniach klientów."
        ),
    )

    vocabulary_divergence = (
        _jensen_shannon_divergence(
            vocabulary_comparison[
                "ReferenceShare"
            ].to_numpy(),
            vocabulary_comparison[
                "CurrentShare"
            ].to_numpy(),
        )
    )

    _add_component(
        component_rows=component_rows,
        area="Opinie",
        metric="Rozkład słownictwa",
        drift_score=vocabulary_divergence,
        reference_value="Słownictwo referencyjne",
        current_value="Słownictwo bieżące",
        change_percentage=None,
        interpretation=(
            "Zmiana częstości występowania "
            "najważniejszych terminów."
        ),
    )

    drift_components = pd.DataFrame(
        component_rows
    )

    overall_drift_score = float(
        drift_components[
            "DriftScore"
        ].mean()
    )

    overall_severity = (
        _assign_drift_severity(
            overall_drift_score
        )
    )

    alerts = drift_components[
        drift_components["Severity"].isin(
            [
                "Średni",
                "Wysoki",
            ]
        )
    ].copy()

    severity_order = pd.CategoricalDtype(
        categories=[
            "Wysoki",
            "Średni",
            "Niski",
        ],
        ordered=True,
    )

    alerts["Severity"] = (
        alerts["Severity"]
        .astype(severity_order)
    )

    alerts = (
        alerts
        .sort_values(
            by=[
                "Severity",
                "DriftScore",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    alerts["Severity"] = (
        alerts["Severity"]
        .astype(str)
    )

    summary = {
        "OverallDriftScore": (
            overall_drift_score
        ),
        "OverallSeverity": overall_severity,
        "NumberOfAlerts": len(alerts),
        "HighSeverityAlerts": int(
            (
                drift_components["Severity"]
                == "Wysoki"
            ).sum()
        ),
        "MediumSeverityAlerts": int(
            (
                drift_components["Severity"]
                == "Średni"
            ).sum()
        ),
        "ReferenceSalesRecords": len(
            reference_sales
        ),
        "CurrentSalesRecords": len(
            current_sales
        ),
        "ReferenceReviews": len(
            reference_reviews
        ),
        "CurrentReviews": len(
            current_reviews
        ),
    }

    return {
        "summary": summary,
        "drift_components": drift_components,
        "alerts": alerts,
        "product_distribution": (
            product_distribution
        ),
        "country_distribution": (
            country_distribution
        ),
        "sentiment_distribution": (
            sentiment_distribution
        ),
        "rating_distribution": (
            rating_distribution
        ),
        "vocabulary_comparison": (
            vocabulary_comparison
        ),
        "sales_windows": sales_windows,
        "review_windows": review_windows,
        "text_language": normalized_language,
    }