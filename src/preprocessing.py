import pandas as pd


def clean_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści dane sprzedażowe i przygotowuje je do analizy.

    Operacje:
    - kopiuje dane wejściowe,
    - usuwa rekordy bez numeru faktury,
    - usuwa anulowane transakcje,
    - konwertuje datę faktury,
    - usuwa rekordy z niepoprawną ilością lub ceną,
    - dodaje kolumnę TotalPrice,
    - dodaje kolumny pomocnicze: Year, Month, YearMonth.
    """
    cleaned = data.copy()

    required_columns = [
        "InvoiceNo",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing_columns = [col for col in required_columns if col not in cleaned.columns]
    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn: {missing_columns}")

    cleaned = cleaned.dropna(subset=["InvoiceNo", "Description", "InvoiceDate", "UnitPrice", "Quantity"])

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)

    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C")]

    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned = cleaned.dropna(subset=["InvoiceDate"])

    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    cleaned = cleaned.dropna(subset=["Quantity", "UnitPrice"])

    cleaned = cleaned[cleaned["Quantity"] > 0]
    cleaned = cleaned[cleaned["UnitPrice"] > 0]

    cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    cleaned["Year"] = cleaned["InvoiceDate"].dt.year
    cleaned["Month"] = cleaned["InvoiceDate"].dt.month
    cleaned["YearMonth"] = cleaned["InvoiceDate"].dt.to_period("M").astype(str)

    return cleaned