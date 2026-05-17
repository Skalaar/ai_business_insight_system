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

def rfm_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wykonuje segmentację klientów metodą RFM.

    RFM:
    - Recency: liczba dni od ostatniego zakupu klienta,
    - Frequency: liczba unikalnych transakcji klienta,
    - Monetary: łączna wartość zakupów klienta.

    Zwraca:
        DataFrame z metrykami RFM, punktacją oraz segmentem klienta.
    """
    rfm_data = data.dropna(subset=["CustomerID"]).copy()

    if rfm_data.empty:
        return pd.DataFrame(
            columns=[
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "R_Score",
                "F_Score",
                "M_Score",
                "RFM_Score",
                "Segment",
            ]
        )

    reference_date = rfm_data["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        rfm_data.groupby("CustomerID", as_index=False)
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    )

    rfm["Recency"] = (reference_date - rfm["LastPurchaseDate"]).dt.days

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        4,
        labels=[4, 3, 2, 1],
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def assign_segment(row):
        if row["R_Score"] >= 3 and row["F_Score"] >= 3 and row["M_Score"] >= 3:
            return "Najlepsi klienci"
        if row["R_Score"] >= 3 and row["F_Score"] >= 3:
            return "Lojalni klienci"
        if row["R_Score"] >= 3 and row["F_Score"] <= 2:
            return "Nowi lub okazjonalni klienci"
        if row["R_Score"] <= 2 and row["F_Score"] >= 3:
            return "Klienci zagrożeni odejściem"
        if row["M_Score"] >= 3 and row["R_Score"] <= 2:
            return "Wartościowi nieaktywni klienci"
        return "Pozostali klienci"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    rfm = rfm[
        [
            "CustomerID",
            "LastPurchaseDate",
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score",
            "Segment",
        ]
    ].sort_values(["M_Score", "F_Score", "R_Score"], ascending=False)

    return rfm