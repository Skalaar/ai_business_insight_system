from pathlib import Path

import plotly.express as px
import streamlit as st

from src.data_loader import load_sales_data, load_uploaded_sales_data
from src.preprocessing import clean_sales_data
from src.sales_analysis import (
    calculate_sales_kpis,
    monthly_sales,
    sales_by_country,
    top_products,
)


st.set_page_config(
    page_title="AI Business Insight System",
    page_icon="📊",
    layout="wide",
)


st.title("AI Business Insight System")
st.subheader("Prototyp systemu wspierającego decyzje przedsiębiorstwa")


SAMPLE_DATA_PATH = Path("data/sample/sample_sales.csv")


@st.cache_data
def get_sample_sales_data():
    raw_data = load_sales_data(SAMPLE_DATA_PATH)
    cleaned_data = clean_sales_data(raw_data)
    return raw_data, cleaned_data


def process_uploaded_sales_file(uploaded_file):
    raw_data = load_uploaded_sales_data(uploaded_file)
    cleaned_data = clean_sales_data(raw_data)
    return raw_data, cleaned_data


st.sidebar.header("Źródło danych")

uploaded_file = st.sidebar.file_uploader(
    "Wgraj plik sprzedażowy CSV/XLSX",
    type=["csv", "xlsx", "xls"],
)

use_sample_data = st.sidebar.checkbox(
    "Użyj danych przykładowych",
    value=True,
)

try:
    if uploaded_file is not None:
        raw_sales_data, cleaned_sales_data = process_uploaded_sales_file(uploaded_file)
        data_source_description = f"Plik użytkownika: {uploaded_file.name}"
    elif use_sample_data:
        raw_sales_data, cleaned_sales_data = get_sample_sales_data()
        data_source_description = "Dane przykładowe: data/sample/sample_sales.csv"
    else:
        raw_sales_data = None
        cleaned_sales_data = None
        data_source_description = "Nie wybrano źródła danych"

except Exception as error:
    raw_sales_data = None
    cleaned_sales_data = None
    data_source_description = "Błąd wczytywania danych"
    st.sidebar.error(f"Błąd danych: {error}")


tab_intro, tab_data, tab_sales = st.tabs(
    [
        "Opis projektu",
        "Dane",
        "Analiza sprzedaży",
    ]
)


with tab_intro:
    st.header("Opis projektu")

    st.write(
        """
        Aplikacja stanowi prototyp systemu analitycznego wspierającego procesy
        decyzyjne przedsiębiorstwa. W obecnej wersji program wczytuje dane
        sprzedażowe, wykonuje ich podstawowe czyszczenie oraz prezentuje
        kluczowe wskaźniki i wizualizacje.
        """
    )

    st.info(
        """
        Ten etap projektu obejmuje podstawowy moduł analizy sprzedaży.
        W kolejnych etapach aplikacja zostanie rozszerzona o analizę opinii
        klientów, klasyfikację sentymentu oraz generowanie rekomendacji
        decyzyjnych.
        """
    )

    st.subheader("Aktualne źródło danych")
    st.write(data_source_description)

    st.subheader("Wymagany format danych sprzedażowych")
    st.write(
        """
        Plik sprzedażowy powinien zawierać następujące kolumny:
        """
    )

    st.code(
        "InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country",
        language="text",
    )


with tab_data:
    st.header("Dane sprzedażowe")

    if raw_sales_data is None or cleaned_sales_data is None:
        st.warning("Nie wczytano danych. Wgraj plik w panelu bocznym lub zaznacz użycie danych przykładowych.")
    else:
        st.success(f"Źródło danych: {data_source_description}")

        st.subheader("Podgląd danych surowych")
        st.dataframe(raw_sales_data, width="stretch")

        st.subheader("Podgląd danych po czyszczeniu")
        st.dataframe(cleaned_sales_data, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Liczba rekordów surowych", len(raw_sales_data))
        col2.metric("Liczba rekordów po czyszczeniu", len(cleaned_sales_data))
        col3.metric("Liczba kolumn", cleaned_sales_data.shape[1])


with tab_sales:
    st.header("Analiza sprzedaży")

    if cleaned_sales_data is None:
        st.warning("Brak danych do analizy. Wgraj plik sprzedażowy lub użyj danych przykładowych.")
    else:
        kpis = calculate_sales_kpis(cleaned_sales_data)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Przychód", f"{kpis['total_revenue']:,.2f}")
        col2.metric("Transakcje", f"{kpis['total_transactions']}")
        col3.metric("Klienci", f"{kpis['total_customers']}")
        col4.metric("Produkty", f"{kpis['total_products']}")
        col5.metric("Śr. wartość zamówienia", f"{kpis['average_order_value']:,.2f}")

        st.divider()

        st.subheader("Sprzedaż w czasie")
        monthly_data = monthly_sales(cleaned_sales_data)

        fig_monthly = px.line(
            monthly_data,
            x="YearMonth",
            y="Revenue",
            markers=True,
            title="Wartość sprzedaży według miesięcy",
        )

        st.plotly_chart(fig_monthly, width="stretch")
        st.dataframe(monthly_data, width="stretch")

        st.divider()

        st.subheader("Najlepiej sprzedające się produkty")
        top_products_data = top_products(cleaned_sales_data, limit=10)

        fig_products = px.bar(
            top_products_data,
            x="Revenue",
            y="Description",
            orientation="h",
            title="TOP 10 produktów według wartości sprzedaży",
        )

        st.plotly_chart(fig_products, width="stretch")
        st.dataframe(top_products_data, width="stretch")

        st.divider()

        st.subheader("Sprzedaż według kraju")
        country_data = sales_by_country(cleaned_sales_data)

        fig_country = px.bar(
            country_data,
            x="Country",
            y="Revenue",
            title="Sprzedaż według kraju",
        )

        st.plotly_chart(fig_country, width="stretch")
        st.dataframe(country_data, width="stretch")