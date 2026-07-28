from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SEVERITY_WEIGHTS = {
    "Wysoki": 3.0,
    "Średni": 2.0,
    "Niski": 1.0,
}


def _safe_issue_rate(
    issue_count: int,
    records_checked: int,
) -> float:
    if records_checked <= 0:
        return 0.0

    return float(
        np.clip(
            issue_count / records_checked,
            0.0,
            1.0,
        )
    )


def _add_quality_check(
    rows: list[dict[str, Any]],
    dataset: str,
    check_name: str,
    severity: str,
    records_checked: int,
    issue_count: int,
    interpretation: str,
    recommendation: str,
) -> None:
    issue_rate = _safe_issue_rate(
        issue_count=issue_count,
        records_checked=records_checked,
    )

    rows.append(
        {
            "Dataset": dataset,
            "Check": check_name,
            "Severity": severity,
            "RecordsChecked": int(records_checked),
            "Issues": int(issue_count),
            "IssueRate": issue_rate,
            "PassRate": 1.0 - issue_rate,
            "Status": (
                "Poprawny"
                if issue_count == 0
                else "Wymaga uwagi"
            ),
            "Interpretation": interpretation,
            "Recommendation": recommendation,
        }
    )


def _blank_text_mask(
    series: pd.Series,
) -> pd.Series:
    return (
        series.isna()
        | series.astype(str).str.strip().eq("")
        | series.astype(str).str.lower().eq("nan")
    )


def _prepare_sales_data(
    sales_data: pd.DataFrame,
) -> pd.DataFrame:
    prepared = sales_data.copy()

    for column in [
        "InvoiceDate",
    ]:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(
                prepared[column],
                errors="coerce",
            )

    for column in [
        "Quantity",
        "UnitPrice",
        "TotalPrice",
    ]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    return prepared


def _prepare_review_data(
    review_data: pd.DataFrame,
) -> pd.DataFrame:
    prepared = review_data.copy()

    if "ReviewDate" in prepared.columns:
        prepared["ReviewDate"] = pd.to_datetime(
            prepared["ReviewDate"],
            errors="coerce",
        )

    if "Rating" in prepared.columns:
        prepared["Rating"] = pd.to_numeric(
            prepared["Rating"],
            errors="coerce",
        )

    return prepared


def _normalize_product_name(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


def _calculate_quality_score(
    quality_checks: pd.DataFrame,
) -> float:
    if quality_checks.empty:
        return 0.0

    weights = quality_checks[
        "Severity"
    ].map(SEVERITY_WEIGHTS).fillna(1.0)

    weighted_pass_rate = (
        quality_checks["PassRate"]
        * weights
    ).sum()

    total_weight = weights.sum()

    if total_weight <= 0:
        return 0.0

    return float(
        np.clip(
            weighted_pass_rate
            / total_weight
            * 100,
            0.0,
            100.0,
        )
    )


def _assign_quality_status(
    quality_score: float,
) -> str:
    if quality_score >= 95:
        return "Bardzo wysoka"

    if quality_score >= 85:
        return "Dobra"

    if quality_score >= 70:
        return "Ostrzegawcza"

    return "Krytyczna"


def build_data_quality_report(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
    minimum_review_length: int = 10,
    value_tolerance: float = 0.01,
) -> dict:
    if sales_data is None or sales_data.empty:
        raise ValueError(
            "Brak danych sprzedażowych do kontroli jakości."
        )

    if review_data is None or review_data.empty:
        raise ValueError(
            "Brak danych opinii do kontroli jakości."
        )

    prepared_sales = _prepare_sales_data(
        sales_data
    )

    prepared_reviews = _prepare_review_data(
        review_data
    )

    checks: list[dict[str, Any]] = []

    sales_required_columns = [
        "InvoiceNo",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "Country",
    ]

    review_required_columns = [
        "ReviewID",
        "ProductName",
        "Rating",
        "ReviewDate",
        "ReviewText",
    ]

    missing_sales_columns = [
        column
        for column in sales_required_columns
        if column not in prepared_sales.columns
    ]

    missing_review_columns = [
        column
        for column in review_required_columns
        if column not in prepared_reviews.columns
    ]

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Kompletność wymaganych kolumn",
        severity="Wysoki",
        records_checked=len(
            sales_required_columns
        ),
        issue_count=len(
            missing_sales_columns
        ),
        interpretation=(
            "Sprawdzenie obecności kolumn niezbędnych "
            "do przeprowadzenia analiz sprzedażowych."
        ),
        recommendation=(
            "Uzupełnij brakujące kolumny: "
            + (
                ", ".join(missing_sales_columns)
                if missing_sales_columns
                else "brak"
            )
        ),
    )

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Kompletność wymaganych kolumn",
        severity="Wysoki",
        records_checked=len(
            review_required_columns
        ),
        issue_count=len(
            missing_review_columns
        ),
        interpretation=(
            "Sprawdzenie obecności kolumn wymaganych "
            "przez moduły analizy tekstowej."
        ),
        recommendation=(
            "Uzupełnij brakujące kolumny: "
            + (
                ", ".join(missing_review_columns)
                if missing_review_columns
                else "brak"
            )
        ),
    )

    sales_record_count = len(
        prepared_sales
    )

    review_record_count = len(
        prepared_reviews
    )

    sales_duplicate_count = int(
        prepared_sales.duplicated().sum()
    )

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Zduplikowane rekordy",
        severity="Średni",
        records_checked=sales_record_count,
        issue_count=sales_duplicate_count,
        interpretation=(
            "Duplikaty mogą prowadzić do zawyżenia "
            "przychodów, ilości i liczby transakcji."
        ),
        recommendation=(
            "Usuń pełne duplikaty po wcześniejszym "
            "sprawdzeniu ich pochodzenia."
        ),
    )

    review_duplicate_count = int(
        prepared_reviews.duplicated().sum()
    )

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Zduplikowane rekordy",
        severity="Średni",
        records_checked=review_record_count,
        issue_count=review_duplicate_count,
        interpretation=(
            "Powielone opinie mogą zniekształcać "
            "rozkład sentymentu i ocen."
        ),
        recommendation=(
            "Usuń powielone rekordy opinii lub pozostaw "
            "je tylko po potwierdzeniu ich autentyczności."
        ),
    )

    if "InvoiceDate" in prepared_sales.columns:
        invalid_sales_dates = int(
            prepared_sales[
                "InvoiceDate"
            ].isna().sum()
        )
    else:
        invalid_sales_dates = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Poprawność dat sprzedaży",
        severity="Wysoki",
        records_checked=sales_record_count,
        issue_count=invalid_sales_dates,
        interpretation=(
            "Nieprawidłowe daty uniemożliwiają analizę "
            "trendów i prognozowanie."
        ),
        recommendation=(
            "Ujednolić format dat i usunąć albo poprawić "
            "rekordy bez prawidłowej daty."
        ),
    )

    if "ReviewDate" in prepared_reviews.columns:
        invalid_review_dates = int(
            prepared_reviews[
                "ReviewDate"
            ].isna().sum()
        )
    else:
        invalid_review_dates = review_record_count

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Poprawność dat opinii",
        severity="Średni",
        records_checked=review_record_count,
        issue_count=invalid_review_dates,
        interpretation=(
            "Błędne daty utrudniają analizę zmian "
            "sentymentu i driftu."
        ),
        recommendation=(
            "Ujednolić format dat opinii i poprawić "
            "nieprawidłowe wartości."
        ),
    )

    if "Quantity" in prepared_sales.columns:
        invalid_quantity_count = int(
            (
                prepared_sales["Quantity"].isna()
                | (
                    prepared_sales["Quantity"]
                    <= 0
                )
            ).sum()
        )
    else:
        invalid_quantity_count = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Dodatnia ilość produktu",
        severity="Wysoki",
        records_checked=sales_record_count,
        issue_count=invalid_quantity_count,
        interpretation=(
            "Ilości zerowe lub ujemne wymagają "
            "oddzielnej interpretacji jako zwroty lub błędy."
        ),
        recommendation=(
            "Rozdziel sprzedaż i zwroty albo usuń błędne "
            "wartości przed obliczeniem KPI."
        ),
    )

    if "UnitPrice" in prepared_sales.columns:
        invalid_price_count = int(
            (
                prepared_sales["UnitPrice"].isna()
                | (
                    prepared_sales["UnitPrice"]
                    <= 0
                )
            ).sum()
        )
    else:
        invalid_price_count = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Dodatnia cena jednostkowa",
        severity="Wysoki",
        records_checked=sales_record_count,
        issue_count=invalid_price_count,
        interpretation=(
            "Cena zerowa lub ujemna może oznaczać "
            "błąd, rabat techniczny albo korektę."
        ),
        recommendation=(
            "Zweryfikuj rekordy z ceną niedodatnią "
            "i określ zasady obsługi korekt."
        ),
    )

    if {
        "Quantity",
        "UnitPrice",
        "TotalPrice",
    }.issubset(
        prepared_sales.columns
    ):
        expected_total = (
            prepared_sales["Quantity"]
            * prepared_sales["UnitPrice"]
        )

        difference = (
            prepared_sales["TotalPrice"]
            - expected_total
        ).abs()

        total_price_issue_mask = (
            prepared_sales["TotalPrice"].isna()
            | expected_total.isna()
            | (
                difference
                > value_tolerance
            )
        )

        inconsistent_total_count = int(
            total_price_issue_mask.sum()
        )
    else:
        inconsistent_total_count = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Spójność wartości sprzedaży",
        severity="Wysoki",
        records_checked=sales_record_count,
        issue_count=inconsistent_total_count,
        interpretation=(
            "TotalPrice powinno odpowiadać iloczynowi "
            "Quantity i UnitPrice."
        ),
        recommendation=(
            "Przelicz wartość sprzedaży albo wyjaśnij "
            "różnice wynikające z rabatów i korekt."
        ),
    )

    if "Description" in prepared_sales.columns:
        blank_sales_products = int(
            _blank_text_mask(
                prepared_sales["Description"]
            ).sum()
        )
    else:
        blank_sales_products = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Kompletność nazw produktów",
        severity="Wysoki",
        records_checked=sales_record_count,
        issue_count=blank_sales_products,
        interpretation=(
            "Brak nazwy produktu uniemożliwia analizę "
            "sprzedaży i połączenie danych z opiniami."
        ),
        recommendation=(
            "Uzupełnij nazwę produktu na podstawie "
            "StockCode albo kartoteki produktowej."
        ),
    )

    if "CustomerID" in prepared_sales.columns:
        missing_customer_count = int(
            prepared_sales[
                "CustomerID"
            ].isna().sum()
        )
    else:
        missing_customer_count = sales_record_count

    _add_quality_check(
        rows=checks,
        dataset="Sprzedaż",
        check_name="Kompletność identyfikatorów klientów",
        severity="Średni",
        records_checked=sales_record_count,
        issue_count=missing_customer_count,
        interpretation=(
            "Brak CustomerID ogranicza możliwość "
            "segmentacji klientów metodą RFM."
        ),
        recommendation=(
            "Uzupełnij identyfikatory klientów albo "
            "wyłącz anonimowe transakcje z segmentacji."
        ),
    )

    if "Rating" in prepared_reviews.columns:
        invalid_rating_count = int(
            (
                prepared_reviews["Rating"].isna()
                | ~prepared_reviews[
                    "Rating"
                ].between(
                    1,
                    5,
                )
            ).sum()
        )
    else:
        invalid_rating_count = review_record_count

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Ocena w zakresie 1–5",
        severity="Wysoki",
        records_checked=review_record_count,
        issue_count=invalid_rating_count,
        interpretation=(
            "Oceny spoza zakresu zaburzają etykiety "
            "sentymentu i metryki jakości produktów."
        ),
        recommendation=(
            "Przeskaluj oceny albo usuń rekordy "
            "niespełniające zakresu 1–5."
        ),
    )

    if "ReviewText" in prepared_reviews.columns:
        blank_review_mask = _blank_text_mask(
            prepared_reviews["ReviewText"]
        )

        blank_review_count = int(
            blank_review_mask.sum()
        )

        short_review_count = int(
            (
                ~blank_review_mask
                & (
                    prepared_reviews[
                        "ReviewText"
                    ]
                    .astype(str)
                    .str.strip()
                    .str.len()
                    < minimum_review_length
                )
            ).sum()
        )
    else:
        blank_review_count = review_record_count
        short_review_count = review_record_count

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Kompletność treści opinii",
        severity="Wysoki",
        records_checked=review_record_count,
        issue_count=blank_review_count,
        interpretation=(
            "Puste opinie nie mogą zostać wykorzystane "
            "w analizie tekstowej."
        ),
        recommendation=(
            "Usuń puste rekordy albo uzupełnij treść "
            "z systemu źródłowego."
        ),
    )

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name=(
            f"Minimalna długość opinii "
            f"({minimum_review_length} znaków)"
        ),
        severity="Niski",
        records_checked=review_record_count,
        issue_count=short_review_count,
        interpretation=(
            "Bardzo krótkie teksty mogą zawierać "
            "zbyt mało informacji dla modeli NLP."
        ),
        recommendation=(
            "Oznacz krótkie opinie i rozważ ich "
            "oddzielną analizę."
        ),
    )

    if "ProductName" in prepared_reviews.columns:
        blank_review_products = int(
            _blank_text_mask(
                prepared_reviews[
                    "ProductName"
                ]
            ).sum()
        )
    else:
        blank_review_products = review_record_count

    _add_quality_check(
        rows=checks,
        dataset="Opinie",
        check_name="Kompletność nazw produktów",
        severity="Wysoki",
        records_checked=review_record_count,
        issue_count=blank_review_products,
        interpretation=(
            "Brak produktu uniemożliwia integrację "
            "opinii z wynikami sprzedaży."
        ),
        recommendation=(
            "Uzupełnij ProductName lub ProductID "
            "na podstawie kartoteki produktu."
        ),
    )

    if {
        "Description",
    }.issubset(
        prepared_sales.columns
    ) and {
        "ProductName",
    }.issubset(
        prepared_reviews.columns
    ):
        sales_products = set(
            _normalize_product_name(
                prepared_sales["Description"]
            )
        )

        sales_products.discard("")

        normalized_review_products = (
            _normalize_product_name(
                prepared_reviews[
                    "ProductName"
                ]
            )
        )

        unmatched_review_mask = (
            ~normalized_review_products.isin(
                sales_products
            )
            & normalized_review_products.ne("")
        )

        unmatched_review_count = int(
            unmatched_review_mask.sum()
        )

        unmatched_products = sorted(
            normalized_review_products[
                unmatched_review_mask
            ].unique().tolist()
        )
    else:
        unmatched_review_count = review_record_count
        unmatched_products = []

    _add_quality_check(
        rows=checks,
        dataset="Integracja",
        check_name="Zgodność produktów między zbiorami",
        severity="Wysoki",
        records_checked=review_record_count,
        issue_count=unmatched_review_count,
        interpretation=(
            "Opinie powinny być możliwe do przypisania "
            "do produktów obecnych w danych sprzedażowych."
        ),
        recommendation=(
            "Ujednolić nazwy lub zastosować wspólny "
            "identyfikator produktu."
        ),
    )

    quality_checks = pd.DataFrame(
        checks
    )

    quality_score = _calculate_quality_score(
        quality_checks
    )

    quality_status = _assign_quality_status(
        quality_score
    )

    failed_checks = quality_checks[
        quality_checks["Issues"] > 0
    ].copy()

    severity_order = pd.CategoricalDtype(
        categories=[
            "Wysoki",
            "Średni",
            "Niski",
        ],
        ordered=True,
    )

    if not failed_checks.empty:
        failed_checks["Severity"] = (
            failed_checks["Severity"]
            .astype(severity_order)
        )

        failed_checks = (
            failed_checks
            .sort_values(
                by=[
                    "Severity",
                    "IssueRate",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        failed_checks["Severity"] = (
            failed_checks["Severity"]
            .astype(str)
        )

    recommendations = (
        failed_checks[
            [
                "Dataset",
                "Check",
                "Severity",
                "Issues",
                "IssueRate",
                "Recommendation",
            ]
        ].copy()
    )

    dataset_summary = (
        quality_checks
        .groupby(
            "Dataset",
            as_index=False,
        )
        .agg(
            Checks=("Check", "size"),
            FailedChecks=(
                "Issues",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            AveragePassRate=(
                "PassRate",
                "mean",
            ),
            TotalIssues=(
                "Issues",
                "sum",
            ),
        )
    )

    summary = {
        "OverallQualityScore": quality_score,
        "QualityStatus": quality_status,
        "TotalChecks": len(
            quality_checks
        ),
        "PassedChecks": int(
            (
                quality_checks["Issues"]
                == 0
            ).sum()
        ),
        "FailedChecks": int(
            (
                quality_checks["Issues"]
                > 0
            ).sum()
        ),
        "HighSeverityFailedChecks": int(
            (
                (
                    quality_checks["Severity"]
                    == "Wysoki"
                )
                & (
                    quality_checks["Issues"]
                    > 0
                )
            ).sum()
        ),
        "SalesRecords": sales_record_count,
        "ReviewRecords": review_record_count,
        "UnmatchedReviewProducts": len(
            unmatched_products
        ),
    }

    return {
        "summary": summary,
        "quality_checks": quality_checks,
        "failed_checks": failed_checks,
        "recommendations": recommendations,
        "dataset_summary": dataset_summary,
        "unmatched_products": pd.DataFrame(
            {
                "ProductName": unmatched_products,
            }
        ),
    }