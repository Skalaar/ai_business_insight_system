from pathlib import Path

import pandas as pd


def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """
    Wczytuje dane sprzedażowe z pliku CSV lub Excel.

    Parametry:
        file_path: ścieżka do pliku z danymi sprzedażowymi

    Zwraca:
        DataFrame z danymi sprzedażowymi
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

    if file_path.suffix.lower() == ".csv":
        data = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Obsługiwane są tylko pliki CSV, XLSX oraz XLS.")

    return data