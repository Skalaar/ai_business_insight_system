import pandas as pd


def filter_sales_data(
    data: pd.DataFrame,
    date_range=None,
    countries=None,
    categories=None,
    products=None,
) -> pd.DataFrame:
    """
    Filtruje dane sprzedażowe według zakresu dat,
    krajów, kategorii i produktów.
    """
    filtered = data.copy()

    if (
        date_range is not None
        and len(date_range) == 2
    ):
        start_date, end_date = date_range

        filtered = filtered[
            (
                filtered["InvoiceDate"].dt.date
                >= start_date
            )
            & (
                filtered["InvoiceDate"].dt.date
                <= end_date
            )
        ]

    if countries:
        filtered = filtered[
            filtered["Country"].isin(
                countries
            )
        ]

    if (
        categories
        and "Category" in filtered.columns
    ):
        filtered = filtered[
            filtered["Category"].isin(
                categories
            )
        ]

    if products:
        filtered = filtered[
            filtered["Description"].isin(
                products
            )
        ]

    return filtered


def filter_review_data(
    data: pd.DataFrame,
    products=None,
    sentiments=None,
) -> pd.DataFrame:
    """
    Filtruje dane tekstowe według produktów i sentymentu.
    """
    filtered = data.copy()

    if products:
        filtered = filtered[filtered["ProductName"].isin(products)]

    if sentiments:
        filtered = filtered[filtered["RatingSentiment"].isin(sentiments)]

    return filtered