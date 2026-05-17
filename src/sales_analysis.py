import pandas as pd


def calculate_sales_kpis(data: pd.DataFrame) -> dict:
    """
    Oblicza podstawowe wskaźniki sprzedażowe.
    """
    total_revenue = data["TotalPrice"].sum()
    total_transactions = data["InvoiceNo"].nunique()
    total_customers = data["CustomerID"].nunique()
    total_products = data["Description"].nunique()

    average_order_value = (
        data.groupby("InvoiceNo")["TotalPrice"].sum().mean()
        if total_transactions > 0
        else 0
    )

    return {
        "total_revenue": total_revenue,
        "total_transactions": total_transactions,
        "total_customers": total_customers,
        "total_products": total_products,
        "average_order_value": average_order_value,
    }


def monthly_sales(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje sprzedaż według miesięcy.
    """
    result = (
        data.groupby("YearMonth", as_index=False)
        .agg(
            Revenue=("TotalPrice", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("YearMonth")
    )

    return result


def top_products(data: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Zwraca ranking najlepiej sprzedających się produktów według wartości sprzedaży.
    """
    result = (
        data.groupby("Description", as_index=False)
        .agg(
            Revenue=("TotalPrice", "sum"),
            Quantity=("Quantity", "sum"),
            Transactions=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(limit)
    )

    return result


def sales_by_country(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje sprzedaż według kraju.
    """
    result = (
        data.groupby("Country", as_index=False)
        .agg(
            Revenue=("TotalPrice", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )

    return result