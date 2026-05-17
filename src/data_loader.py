from pathlib import Path
from typing import BinaryIO

import pandas as pd


def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """
    Wczytuje dane sprzedażowe z pliku CSV lub Excel na podstawie ścieżki do pliku.

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


def load_uploaded_sales_data(uploaded_file: BinaryIO) -> pd.DataFrame:
    """
    Wczytuje dane sprzedażowe z pliku przesłanego przez interfejs Streamlit.

    Parametry:
        uploaded_file: plik przesłany przez komponent st.file_uploader

    Zwraca:
        DataFrame z danymi sprzedażowymi
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        data = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Obsługiwane są tylko pliki CSV, XLSX oraz XLS.")

    return data

def load_review_data(file_path: str | Path) -> pd.DataFrame:
    """
    Wczytuje dane tekstowe z opiniami klientów z pliku CSV lub Excel.
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


def load_uploaded_review_data(uploaded_file: BinaryIO) -> pd.DataFrame:
    """
    Wczytuje dane tekstowe z opiniami klientów przesłane przez interfejs Streamlit.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        data = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Obsługiwane są tylko pliki CSV, XLSX oraz XLS.")

    return data