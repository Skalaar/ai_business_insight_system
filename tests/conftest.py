from __future__ import annotations

import pandas as pd
import pytest


PRODUCTS = [
    "PRODUCT QUALITY",
    "PRODUCT DELIVERY",
    "PRODUCT PRICE",
    "PRODUCT DESIGN",
    "PRODUCT PACKAGING",
    "PRODUCT DURABILITY",
]


POSITIVE_TEXTS = [
    "excellent quality product and durable material",
    "fast delivery and very good packaging",
    "good value for money and reasonable price",
    "beautiful design and excellent appearance",
    "reliable product with strong construction",
    "very satisfied with quality and performance",
    "useful product and positive purchase experience",
    "great design with practical functionality",
    "good packaging and product arrived safely",
    "durable material and excellent product quality",
]


NEUTRAL_TEXTS = [
    "average product with acceptable quality",
    "delivery was normal and packaging was standard",
    "price is acceptable but nothing special",
    "design is ordinary and product works normally",
    "mixed experience with quality and durability",
    "product is usable but not impressive",
    "acceptable purchase with average performance",
    "quality is standard for this price",
    "packaging was normal and delivery acceptable",
    "product works but could be improved",
]


NEGATIVE_TEXTS = [
    "poor quality product and weak material",
    "late delivery and damaged packaging",
    "bad value for money and excessive price",
    "disappointing design and poor appearance",
    "unreliable product with weak construction",
    "very disappointed with quality and performance",
    "product broke quickly and was not durable",
    "bad packaging and product arrived damaged",
    "cheap material and terrible product quality",
    "poor purchase experience and disappointing product",
]


@pytest.fixture
def sales_data() -> pd.DataFrame:
    """
    Tworzy deterministyczne dane sprzedażowe obejmujące 24 tygodnie.
    """
    rows = []

    dates = pd.date_range(
        start="2024-01-01",
        periods=168,
        freq="D",
    )

    countries = [
        "United Kingdom",
        "Germany",
        "France",
    ]

    for index, invoice_date in enumerate(dates):
        product_index = index % len(PRODUCTS)
        product_name = PRODUCTS[product_index]

        quantity = 1 + index % 6
        unit_price = 2.5 + product_index * 1.25

        seasonal_factor = (
            1.20
            if invoice_date.month in [3, 4]
            else 1.0
        )

        total_price = (
            quantity
            * unit_price
            * seasonal_factor
        )

        rows.append(
            {
                "InvoiceNo": f"INV-{index:05d}",
                "StockCode": f"STK-{product_index:03d}",
                "Description": product_name,
                "Quantity": quantity,
                "InvoiceDate": invoice_date,
                "UnitPrice": unit_price,
                "CustomerID": 10000 + index % 40,
                "Country": countries[
                    index % len(countries)
                ],
                "TotalPrice": total_price,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def review_data() -> pd.DataFrame:
    """
    Tworzy 120 opinii należących do trzech klas sentymentu.
    """
    rows = []

    sentiment_configuration = [
        (
            "Pozytywny",
            5,
            POSITIVE_TEXTS,
        ),
        (
            "Neutralny",
            3,
            NEUTRAL_TEXTS,
        ),
        (
            "Negatywny",
            1,
            NEGATIVE_TEXTS,
        ),
    ]

    review_dates = pd.date_range(
        start="2024-01-01",
        periods=120,
        freq="D",
    )

    for index, review_date in enumerate(
        review_dates
    ):
        (
            sentiment,
            rating,
            text_collection,
        ) = sentiment_configuration[
            index % len(sentiment_configuration)
        ]

        template_index = (
            index // len(sentiment_configuration)
        ) % len(text_collection)

        product_name = PRODUCTS[
            index % len(PRODUCTS)
        ]

        review_text = text_collection[
            template_index
        ]

        rows.append(
            {
                "ReviewID": index + 1,
                "ProductID": (
                    f"STK-{index % len(PRODUCTS):03d}"
                ),
                "ProductName": product_name,
                "Rating": rating,
                "ReviewDate": review_date,
                "ReviewText": review_text,
                "CleanReviewText": review_text.lower(),
                "RatingSentiment": sentiment,
            }
        )

    return pd.DataFrame(rows)