from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "olist"
)

SALES_PATH = (
    DATA_DIR
    / "olist_sales_prepared.csv"
)

REVIEWS_PATH = (
    DATA_DIR
    / "olist_reviews_prepared.csv"
)

AUDIT_PATH = (
    DATA_DIR
    / "olist_audit_summary.json"
)

PRODUCT_COVERAGE_PATH = (
    DATA_DIR
    / "olist_product_review_coverage.csv"
)

CATEGORY_SUMMARY_PATH = (
    DATA_DIR
    / "olist_category_summary.csv"
)


def percentage(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator * 100
    )


def normalize_boolean(
    series: pd.Series,
) -> pd.Series:
    """
    Normalizuje wartości logiczne zapisane jako bool
    albo tekst True/False.
    """
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


def make_json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not SALES_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {SALES_PATH}"
        )

    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {REVIEWS_PATH}"
        )

    sales = pd.read_csv(
        SALES_PATH,
        low_memory=False,
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        low_memory=False,
    )

    return sales, reviews


def prepare_data_types(
    sales: pd.DataFrame,
    reviews: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    sales = sales.copy()
    reviews = reviews.copy()

    for column in [
        "InvoiceDate",
        "DeliveryDate",
        "EstimatedDeliveryDate",
    ]:
        if column in sales.columns:
            sales[column] = pd.to_datetime(
                sales[column],
                errors="coerce",
            )

    for column in [
        "Quantity",
        "UnitPrice",
        "TotalPrice",
        "FreightValue",
        "DeliveryDelayDays",
    ]:
        if column in sales.columns:
            sales[column] = pd.to_numeric(
                sales[column],
                errors="coerce",
            )

    for column in [
        "ReviewDate",
        "PurchaseDate",
        "ReviewAnswerDate",
    ]:
        if column in reviews.columns:
            reviews[column] = pd.to_datetime(
                reviews[column],
                errors="coerce",
            )

    if "Rating" in reviews.columns:
        reviews["Rating"] = pd.to_numeric(
            reviews["Rating"],
            errors="coerce",
        )

    if "DeliveryDelayDays" in reviews.columns:
        reviews["DeliveryDelayDays"] = pd.to_numeric(
            reviews["DeliveryDelayDays"],
            errors="coerce",
        )

    return sales, reviews


def build_product_coverage(
    sales: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    sales_summary = (
        sales
        .groupby(
            [
                "ProductID",
                "ProductName",
                "Category",
            ],
            as_index=False,
        )
        .agg(
            Revenue=(
                "TotalPrice",
                "sum",
            ),
            Quantity=(
                "Quantity",
                "sum",
            ),
            Orders=(
                "InvoiceNo",
                "nunique",
            ),
            Customers=(
                "CustomerID",
                "nunique",
            ),
        )
    )

    review_summary = (
        reviews
        .groupby(
            "ProductID",
            as_index=False,
        )
        .agg(
            Reviews=(
                "ReviewID",
                "size",
            ),
            AverageRating=(
                "Rating",
                "mean",
            ),
            PositiveReviews=(
                "RatingSentiment",
                lambda values: int(
                    (
                        values
                        == "Pozytywny"
                    ).sum()
                ),
            ),
            NeutralReviews=(
                "RatingSentiment",
                lambda values: int(
                    (
                        values
                        == "Neutralny"
                    ).sum()
                ),
            ),
            NegativeReviews=(
                "RatingSentiment",
                lambda values: int(
                    (
                        values
                        == "Negatywny"
                    ).sum()
                ),
            ),
        )
    )

    coverage = sales_summary.merge(
        review_summary,
        on="ProductID",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "Reviews",
        "PositiveReviews",
        "NeutralReviews",
        "NegativeReviews",
    ]

    for column in count_columns:
        coverage[column] = (
            coverage[column]
            .fillna(0)
            .astype(int)
        )

    coverage["HasReviews"] = (
        coverage["Reviews"] > 0
    )

    coverage["PositiveShare"] = np.where(
        coverage["Reviews"] > 0,
        coverage["PositiveReviews"]
        / coverage["Reviews"],
        np.nan,
    )

    coverage["NegativeShare"] = np.where(
        coverage["Reviews"] > 0,
        coverage["NegativeReviews"]
        / coverage["Reviews"],
        np.nan,
    )

    return (
        coverage
        .sort_values(
            by=[
                "Reviews",
                "Revenue",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_category_summary(
    product_coverage: pd.DataFrame,
) -> pd.DataFrame:
    return (
        product_coverage
        .groupby(
            "Category",
            as_index=False,
        )
        .agg(
            Products=(
                "ProductID",
                "nunique",
            ),
            ProductsWithReviews=(
                "HasReviews",
                "sum",
            ),
            Revenue=(
                "Revenue",
                "sum",
            ),
            Orders=(
                "Orders",
                "sum",
            ),
            Reviews=(
                "Reviews",
                "sum",
            ),
            AverageRating=(
                "AverageRating",
                "mean",
            ),
        )
        .sort_values(
            by="Revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    sales, reviews = load_data()

    sales, reviews = prepare_data_types(
        sales=sales,
        reviews=reviews,
    )

    product_coverage = build_product_coverage(
        sales=sales,
        reviews=reviews,
    )

    category_summary = build_category_summary(
        product_coverage=product_coverage,
    )

    expected_total = (
        sales["Quantity"]
        * sales["UnitPrice"]
    )

    inconsistent_total_mask = (
        sales["TotalPrice"].isna()
        | expected_total.isna()
        | (
            (
                sales["TotalPrice"]
                - expected_total
            )
            .abs()
            > 0.01
        )
    )

    blank_review_mask = (
        reviews["ReviewText"].isna()
        | reviews[
            "ReviewText"
        ]
        .astype(str)
        .str.strip()
        .eq("")
    )

    sales_product_ids = set(
        sales["ProductID"]
        .dropna()
        .astype(str)
    )

    review_product_ids = set(
        reviews["ProductID"]
        .dropna()
        .astype(str)
    )

    unmatched_review_products = (
        review_product_ids
        - sales_product_ids
    )

    review_counts = (
        product_coverage.loc[
            product_coverage["Reviews"] > 0,
            "Reviews",
        ]
    )

    revenue_total = float(
        product_coverage["Revenue"].sum()
    )

    revenue_with_reviews = float(
        product_coverage.loc[
            product_coverage["HasReviews"],
            "Revenue",
        ].sum()
    )

    sales_late = normalize_boolean(
        sales["DeliveredLate"]
    )

    review_late = normalize_boolean(
        reviews["DeliveredLate"]
    )

    late_review_analysis = (
        reviews
        .assign(
            DeliveredLateNormalized=review_late
        )
        .dropna(
            subset=[
                "DeliveredLateNormalized",
                "Rating",
            ]
        )
        .groupby(
            "DeliveredLateNormalized",
            as_index=False,
        )
        .agg(
            Reviews=(
                "ReviewID",
                "size",
            ),
            AverageRating=(
                "Rating",
                "mean",
            ),
            NegativeShare=(
                "RatingSentiment",
                lambda values: float(
                    (
                        values
                        == "Negatywny"
                    ).mean()
                ),
            ),
        )
    )

    rating_distribution = (
        reviews["Rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    sentiment_distribution = (
        reviews["RatingSentiment"]
        .value_counts()
        .to_dict()
    )

    product_thresholds = {}

    for threshold in [
        1,
        3,
        5,
        10,
        20,
        50,
    ]:
        product_thresholds[
            f"products_with_at_least_{threshold}_reviews"
        ] = int(
            (
                product_coverage["Reviews"]
                >= threshold
            ).sum()
        )

    core_sales_columns = [
        "InvoiceNo",
        "ProductID",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "TotalPrice",
    ]

    core_review_columns = [
        "ReviewID",
        "ProductID",
        "Rating",
        "ReviewDate",
        "ReviewText",
    ]

    sales_nulls = {
        column: int(
            sales[column].isna().sum()
        )
        for column in core_sales_columns
        if column in sales.columns
    }

    review_nulls = {
        column: int(
            reviews[column].isna().sum()
        )
        for column in core_review_columns
        if column in reviews.columns
    }

    summary = {
        "sales": {
            "rows": len(sales),
            "orders": int(
                sales["InvoiceNo"].nunique()
            ),
            "products": int(
                sales["ProductID"].nunique()
            ),
            "customers": int(
                sales["CustomerID"].nunique()
            ),
            "date_min": sales[
                "InvoiceDate"
            ].min(),
            "date_max": sales[
                "InvoiceDate"
            ].max(),
            "duplicate_rows": int(
                sales.duplicated().sum()
            ),
            "duplicate_order_product_pairs": int(
                sales.duplicated(
                    subset=[
                        "InvoiceNo",
                        "ProductID",
                    ]
                ).sum()
            ),
            "invalid_quantity": int(
                (
                    sales["Quantity"].isna()
                    | (
                        sales["Quantity"]
                        <= 0
                    )
                ).sum()
            ),
            "invalid_unit_price": int(
                (
                    sales["UnitPrice"].isna()
                    | (
                        sales["UnitPrice"]
                        <= 0
                    )
                ).sum()
            ),
            "inconsistent_total_price": int(
                inconsistent_total_mask.sum()
            ),
            "late_delivery_rows": int(
                sales_late.eq(True).sum()
            ),
            "late_delivery_share_pct": (
                percentage(
                    sales_late.eq(True).sum(),
                    sales_late.notna().sum(),
                )
            ),
            "core_nulls": sales_nulls,
        },
        "reviews": {
            "rows": len(reviews),
            "review_ids": int(
                reviews["ReviewID"].nunique()
            ),
            "orders": int(
                reviews["OrderID"].nunique()
            ),
            "products": int(
                reviews["ProductID"].nunique()
            ),
            "date_min": reviews[
                "ReviewDate"
            ].min(),
            "date_max": reviews[
                "ReviewDate"
            ].max(),
            "duplicate_rows": int(
                reviews.duplicated().sum()
            ),
            "duplicate_order_ids": int(
                reviews.duplicated(
                    subset=["OrderID"]
                ).sum()
            ),
            "duplicate_review_ids": int(
                reviews.duplicated(
                    subset=["ReviewID"]
                ).sum()
            ),
            "blank_review_texts": int(
                blank_review_mask.sum()
            ),
            "reviews_below_10_characters": int(
                (
                    ~blank_review_mask
                    & (
                        reviews["ReviewText"]
                        .astype(str)
                        .str.strip()
                        .str.len()
                        < 10
                    )
                ).sum()
            ),
            "invalid_ratings": int(
                (
                    reviews["Rating"].isna()
                    | ~reviews["Rating"].between(
                        1,
                        5,
                    )
                ).sum()
            ),
            "average_text_length": float(
                reviews["ReviewText"]
                .astype(str)
                .str.len()
                .mean()
            ),
            "median_text_length": float(
                reviews["ReviewText"]
                .astype(str)
                .str.len()
                .median()
            ),
            "rating_distribution": (
                rating_distribution
            ),
            "sentiment_distribution": (
                sentiment_distribution
            ),
            "unmatched_product_ids": len(
                unmatched_review_products
            ),
            "core_nulls": review_nulls,
        },
        "product_coverage": {
            "sales_products": int(
                product_coverage[
                    "ProductID"
                ].nunique()
            ),
            "products_with_reviews": int(
                product_coverage[
                    "HasReviews"
                ].sum()
            ),
            "product_coverage_pct": (
                percentage(
                    product_coverage[
                        "HasReviews"
                    ].sum(),
                    product_coverage[
                        "ProductID"
                    ].nunique(),
                )
            ),
            "revenue_covered_by_reviewed_products_pct": (
                percentage(
                    revenue_with_reviews,
                    revenue_total,
                )
            ),
            "median_reviews_per_reviewed_product": (
                float(
                    review_counts.median()
                )
                if not review_counts.empty
                else 0.0
            ),
            "mean_reviews_per_reviewed_product": (
                float(
                    review_counts.mean()
                )
                if not review_counts.empty
                else 0.0
            ),
            "maximum_reviews_for_one_product": (
                int(
                    review_counts.max()
                )
                if not review_counts.empty
                else 0
            ),
            **product_thresholds,
        },
        "delivery_and_reviews": (
            late_review_analysis
            .to_dict(
                orient="records"
            )
        ),
    }

    safe_summary = json.loads(
        json.dumps(
            summary,
            default=make_json_safe,
            ensure_ascii=False,
        )
    )

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            safe_summary,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    product_coverage.to_csv(
        PRODUCT_COVERAGE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_summary.to_csv(
        CATEGORY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("AUDYT DANYCH OLIST")
    print("=" * 65)

    print(
        f"Sprzedaż: {len(sales):,} wierszy, "
        f"{sales['InvoiceNo'].nunique():,} zamówień, "
        f"{sales['ProductID'].nunique():,} produktów"
    )

    print(
        f"Opinie: {len(reviews):,} wierszy, "
        f"{reviews['ProductID'].nunique():,} produktów"
    )

    print()
    print(
        "Duplikaty sprzedaży: "
        f"{summary['sales']['duplicate_rows']:,}"
    )

    print(
        "Niespójne TotalPrice: "
        f"{summary['sales']['inconsistent_total_price']:,}"
    )

    print(
        "Duplikaty OrderID w opiniach: "
        f"{summary['reviews']['duplicate_order_ids']:,}"
    )

    print(
        "Puste teksty opinii: "
        f"{summary['reviews']['blank_review_texts']:,}"
    )

    print(
        "Niepasujące produkty opinii: "
        f"{summary['reviews']['unmatched_product_ids']:,}"
    )

    print()
    print(
        "Pokrycie produktów opiniami: "
        f"{summary['product_coverage']['product_coverage_pct']:.2f}%"
    )

    print(
        "Pokrycie przychodu produktami z opiniami: "
        f"{summary['product_coverage']['revenue_covered_by_reviewed_products_pct']:.2f}%"
    )

    print(
        "Mediana opinii na oceniany produkt: "
        f"{summary['product_coverage']['median_reviews_per_reviewed_product']:.2f}"
    )

    print()
    print("Liczba produktów według progu opinii:")

    for threshold in [
        1,
        3,
        5,
        10,
        20,
        50,
    ]:
        key = (
            f"products_with_at_least_"
            f"{threshold}_reviews"
        )

        print(
            f"  >= {threshold:2d}: "
            f"{summary['product_coverage'][key]:,}"
        )

    print()
    print("Rozkład ocen:")

    for rating, count in sorted(
        rating_distribution.items()
    ):
        print(
            f"  {int(rating)} gwiazdek: "
            f"{int(count):,}"
        )

    print()
    print("Oceny a terminowość dostawy:")

    for row in late_review_analysis.to_dict(
        orient="records"
    ):
        delivery_label = (
            "opóźnione"
            if row[
                "DeliveredLateNormalized"
            ]
            else "terminowe"
        )

        print(
            f"  {delivery_label}: "
            f"{int(row['Reviews']):,} opinii, "
            f"średnia ocena "
            f"{row['AverageRating']:.3f}, "
            f"negatywne "
            f"{row['NegativeShare']:.2%}"
        )

    print()
    print(f"Podsumowanie JSON: {AUDIT_PATH}")
    print(
        "Pokrycie produktów: "
        f"{PRODUCT_COVERAGE_PATH}"
    )
    print(
        "Podsumowanie kategorii: "
        f"{CATEGORY_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()