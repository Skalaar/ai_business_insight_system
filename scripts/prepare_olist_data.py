from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "olist"
)

REQUIRED_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": (
        "product_category_name_translation.csv"
    ),
}


def validate_source_files() -> None:
    """
    Sprawdza obecność plików wymaganych do przygotowania
    danych sprzedażowych i tekstowych.
    """
    missing_files = [
        file_name
        for file_name in REQUIRED_FILES.values()
        if not (RAW_DIR / file_name).exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"- {file_name}"
            for file_name in missing_files
        )

        raise FileNotFoundError(
            "Brakuje następujących plików Olist:\n"
            f"{missing_text}\n\n"
            f"Oczekiwany folder: {RAW_DIR}"
        )


def read_source_file(
    source_name: str,
) -> pd.DataFrame:
    """
    Wczytuje wskazany plik źródłowy Olist.
    """
    file_path = (
        RAW_DIR
        / REQUIRED_FILES[source_name]
    )

    return pd.read_csv(
        file_path,
        low_memory=False,
    )


def prepare_product_lookup(
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tworzy słownik produktów z angielskimi kategoriami
    i czytelną nazwą wyświetlaną.

    Olist nie zawiera handlowych nazw produktów, dlatego
    nazwa składa się z kategorii i skrótu product_id.
    """
    product_lookup = products.merge(
        category_translation,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    category_name = (
        product_lookup[
            "product_category_name_english"
        ]
        .fillna(
            product_lookup[
                "product_category_name"
            ]
        )
        .fillna("uncategorized")
        .astype(str)
        .str.strip()
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.title()
    )

    product_lookup["Category"] = (
        category_name
    )

    product_lookup["ProductName"] = (
        product_lookup["Category"]
        + " | "
        + product_lookup[
            "product_id"
        ].astype(str).str[:8]
    )

    selected_columns = [
        "product_id",
        "ProductName",
        "Category",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in product_lookup.columns
    ]

    return (
        product_lookup[
            available_columns
        ]
        .drop_duplicates(
            subset=["product_id"]
        )
        .reset_index(drop=True)
    )


def prepare_sales_data(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
    product_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje dane sprzedażowe zgodne z formatem aplikacji.

    Analiza obejmuje wyłącznie zamówienia o statusie delivered.
    Pozycje tego samego produktu w jednym zamówieniu są
    agregowane, dzięki czemu Quantity oznacza liczbę sztuk.
    """
    prepared_orders = orders.copy()

    order_date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in order_date_columns:
        if column in prepared_orders.columns:
            prepared_orders[column] = (
                pd.to_datetime(
                    prepared_orders[column],
                    errors="coerce",
                )
            )

    prepared_orders = prepared_orders[
        prepared_orders["order_status"]
        .eq("delivered")
    ].copy()

    prepared_items = order_items.copy()

    for column in [
        "order_item_id",
        "price",
        "freight_value",
    ]:
        prepared_items[column] = pd.to_numeric(
            prepared_items[column],
            errors="coerce",
        )

    prepared_items = prepared_items.dropna(
        subset=[
            "order_id",
            "product_id",
            "price",
        ]
    )

    prepared_items = prepared_items[
        prepared_items["price"] > 0
    ].copy()

    aggregated_items = (
        prepared_items
        .groupby(
            [
                "order_id",
                "product_id",
            ],
            as_index=False,
        )
        .agg(
            Quantity=(
                "order_item_id",
                "size",
            ),
            TotalPrice=(
                "price",
                "sum",
            ),
            FreightValue=(
                "freight_value",
                "sum",
            ),
            SellerCount=(
                "seller_id",
                "nunique",
            ),
        )
    )

    sales = (
        aggregated_items
        .merge(
            prepared_orders,
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            customers,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            product_lookup,
            on="product_id",
            how="left",
            validate="many_to_one",
        )
    )

    sales["Category"] = (
        sales["Category"]
        .fillna("Uncategorized")
    )

    sales["ProductName"] = (
        sales["ProductName"]
        .fillna(
            "Uncategorized | "
            + sales["product_id"]
            .astype(str)
            .str[:8]
        )
    )

    sales["UnitPrice"] = (
        sales["TotalPrice"]
        / sales["Quantity"]
    )

    sales["DeliveryDelayDays"] = (
        (
            sales[
                "order_delivered_customer_date"
            ]
            - sales[
                "order_estimated_delivery_date"
            ]
        )
        .dt.total_seconds()
        .div(86400)
    )

    sales["DeliveredLate"] = (
        sales["DeliveryDelayDays"] > 0
    )

    sales["InvoiceNo"] = sales[
        "order_id"
    ].astype(str)

    sales["StockCode"] = sales[
        "product_id"
    ].astype(str)

    sales["Description"] = sales[
        "ProductName"
    ]

    sales["InvoiceDate"] = sales[
        "order_purchase_timestamp"
    ]

    sales["CustomerID"] = (
        sales["customer_unique_id"]
        .fillna(
            sales["customer_id"]
        )
        .astype(str)
    )

    sales["Country"] = "Brazil"

    sales["OrderStatus"] = sales[
        "order_status"
    ]

    sales["CustomerCity"] = sales[
        "customer_city"
    ]

    sales["CustomerState"] = sales[
        "customer_state"
    ]

    sales["DeliveryDate"] = sales[
        "order_delivered_customer_date"
    ]

    sales["EstimatedDeliveryDate"] = sales[
        "order_estimated_delivery_date"
    ]

    sales["ProductID"] = sales[
        "product_id"
    ].astype(str)

    output_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "TotalPrice",
        "FreightValue",
        "ProductID",
        "ProductName",
        "Category",
        "OrderStatus",
        "CustomerCity",
        "CustomerState",
        "DeliveryDate",
        "EstimatedDeliveryDate",
        "DeliveryDelayDays",
        "DeliveredLate",
        "SellerCount",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in sales.columns
    ]

    sales = (
        sales[available_columns]
        .sort_values(
            by=[
                "InvoiceDate",
                "InvoiceNo",
                "ProductID",
            ]
        )
        .reset_index(drop=True)
    )

    return sales


def combine_review_text(
    review_title: pd.Series,
    review_message: pd.Series,
) -> pd.Series:
    """
    Łączy tytuł i treść opinii bez tworzenia
    sztucznych wartości 'nan'.
    """
    title = (
        review_title
        .fillna("")
        .astype(str)
        .str.strip()
    )

    message = (
        review_message
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return pd.Series(
        np.select(
            condlist=[
                title.eq(""),
                message.eq(""),
            ],
            choicelist=[
                message,
                title,
            ],
            default=(
                title
                + ". "
                + message
            ),
        ),
        index=review_title.index,
        dtype="object",
    )


def prepare_review_data(
    order_reviews: pd.DataFrame,
    sales_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje opinie powiązane jednoznacznie z produktem.

    Zachowywane są wyłącznie:
    - opinie z tekstem,
    - oceny w zakresie 1–5,
    - dostarczone zamówienia,
    - zamówienia z jednym unikalnym produktem.

    Zamówienie może zawierać kilka sztuk tego samego produktu.
    """
    reviews = order_reviews.copy()

    for column in [
        "review_creation_date",
        "review_answer_timestamp",
    ]:
        reviews[column] = pd.to_datetime(
            reviews[column],
            errors="coerce",
        )

    reviews["review_score"] = pd.to_numeric(
        reviews["review_score"],
        errors="coerce",
    )

    reviews = (
        reviews
        .sort_values(
            by=[
                "order_id",
                "review_answer_timestamp",
            ]
        )
        .drop_duplicates(
            subset=["order_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    order_product_counts = (
        sales_data
        .groupby(
            "InvoiceNo",
            as_index=False,
        )
        .agg(
            UniqueProducts=(
                "ProductID",
                "nunique",
            ),
            ProductID=(
                "ProductID",
                "first",
            ),
            ProductName=(
                "ProductName",
                "first",
            ),
            Category=(
                "Category",
                "first",
            ),
            CustomerID=(
                "CustomerID",
                "first",
            ),
            PurchaseDate=(
                "InvoiceDate",
                "first",
            ),
            DeliveryDelayDays=(
                "DeliveryDelayDays",
                "first",
            ),
            DeliveredLate=(
                "DeliveredLate",
                "first",
            ),
        )
    )

    single_product_orders = (
        order_product_counts[
            order_product_counts[
                "UniqueProducts"
            ].eq(1)
        ]
        .copy()
    )

    reviews = reviews.merge(
        single_product_orders,
        left_on="order_id",
        right_on="InvoiceNo",
        how="inner",
        validate="one_to_one",
    )

    reviews["ReviewText"] = (
        combine_review_text(
            review_title=reviews[
                "review_comment_title"
            ],
            review_message=reviews[
                "review_comment_message"
            ],
        )
    )

    reviews = reviews[
        reviews["ReviewText"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    reviews = reviews[
        reviews["review_score"]
        .between(1, 5)
    ].copy()

    reviews["ReviewID"] = (
        reviews["review_id"]
        .fillna(
            reviews["order_id"]
        )
        .astype(str)
    )

    reviews["Rating"] = (
        reviews["review_score"]
        .round()
        .astype(int)
    )

    reviews["ReviewDate"] = reviews[
        "review_creation_date"
    ]

    reviews["ReviewTitle"] = reviews[
        "review_comment_title"
    ]

    reviews["ReviewMessage"] = reviews[
        "review_comment_message"
    ]

    reviews["ReviewAnswerDate"] = reviews[
        "review_answer_timestamp"
    ]

    reviews["OrderID"] = reviews[
        "order_id"
    ]

    reviews["ReviewLanguage"] = "pt"

    reviews["RatingSentiment"] = np.select(
        condlist=[
            reviews["Rating"] <= 2,
            reviews["Rating"] == 3,
        ],
        choicelist=[
            "Negatywny",
            "Neutralny",
        ],
        default="Pozytywny",
    )

    output_columns = [
        "ReviewID",
        "ProductID",
        "ProductName",
        "Rating",
        "ReviewDate",
        "ReviewText",
        "RatingSentiment",
        "Category",
        "OrderID",
        "CustomerID",
        "PurchaseDate",
        "DeliveryDelayDays",
        "DeliveredLate",
        "ReviewLanguage",
        "ReviewTitle",
        "ReviewMessage",
        "ReviewAnswerDate",
    ]

    reviews = (
        reviews[output_columns]
        .sort_values(
            by=[
                "ReviewDate",
                "ReviewID",
            ]
        )
        .reset_index(drop=True)
    )

    return reviews


def save_outputs(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
    raw_counts: dict[str, int],
) -> None:
    """
    Zapisuje przygotowane dane oraz podsumowanie procesu.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sales_path = (
        OUTPUT_DIR
        / "olist_sales_prepared.csv"
    )

    reviews_path = (
        OUTPUT_DIR
        / "olist_reviews_prepared.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "olist_preparation_summary.json"
    )

    sales_data.to_csv(
        sales_path,
        index=False,
        encoding="utf-8-sig",
    )

    review_data.to_csv(
        reviews_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        **raw_counts,
        "prepared_sales_rows": len(
            sales_data
        ),
        "prepared_sales_orders": int(
            sales_data["InvoiceNo"].nunique()
        ),
        "prepared_sales_products": int(
            sales_data["ProductID"].nunique()
        ),
        "prepared_sales_customers": int(
            sales_data["CustomerID"].nunique()
        ),
        "prepared_review_rows": len(
            review_data
        ),
        "prepared_review_products": int(
            review_data["ProductID"].nunique()
        ),
        "prepared_review_orders": int(
            review_data["OrderID"].nunique()
        ),
        "sales_date_min": str(
            sales_data["InvoiceDate"].min()
        ),
        "sales_date_max": str(
            sales_data["InvoiceDate"].max()
        ),
        "review_date_min": str(
            review_data["ReviewDate"].min()
        ),
        "review_date_max": str(
            review_data["ReviewDate"].max()
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Przygotowanie danych Olist zakończone.")
    print("-" * 55)
    print(
        f"Sprzedaż: {len(sales_data):,} wierszy, "
        f"{sales_data['InvoiceNo'].nunique():,} zamówień"
    )
    print(
        f"Opinie:   {len(review_data):,} wierszy, "
        f"{review_data['ProductID'].nunique():,} produktów"
    )
    print()
    print(f"Plik sprzedaży: {sales_path}")
    print(f"Plik opinii:    {reviews_path}")
    print(f"Podsumowanie:   {summary_path}")


def main() -> None:
    validate_source_files()

    customers = read_source_file(
        "customers"
    )

    orders = read_source_file(
        "orders"
    )

    order_items = read_source_file(
        "order_items"
    )

    order_reviews = read_source_file(
        "order_reviews"
    )

    products = read_source_file(
        "products"
    )

    category_translation = read_source_file(
        "category_translation"
    )

    raw_counts = {
        "raw_customers": len(customers),
        "raw_orders": len(orders),
        "raw_order_items": len(order_items),
        "raw_reviews": len(order_reviews),
        "raw_products": len(products),
        "raw_delivered_orders": int(
            orders["order_status"]
            .eq("delivered")
            .sum()
        ),
    }

    product_lookup = prepare_product_lookup(
        products=products,
        category_translation=(
            category_translation
        ),
    )

    sales_data = prepare_sales_data(
        orders=orders,
        order_items=order_items,
        customers=customers,
        product_lookup=product_lookup,
    )

    review_data = prepare_review_data(
        order_reviews=order_reviews,
        sales_data=sales_data,
    )

    if sales_data.empty:
        raise ValueError(
            "Po przygotowaniu nie pozostały "
            "żadne dane sprzedażowe."
        )

    if review_data.empty:
        raise ValueError(
            "Po przygotowaniu nie pozostały "
            "żadne opinie produktowe."
        )

    save_outputs(
        sales_data=sales_data,
        review_data=review_data,
        raw_counts=raw_counts,
    )


if __name__ == "__main__":
    main()