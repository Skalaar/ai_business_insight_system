from pathlib import Path

import plotly.express as px
import streamlit as st

from src.data_loader import (
    load_review_data,
    load_sales_data,
    load_uploaded_review_data,
    load_uploaded_sales_data,
)
from src.preprocessing import clean_sales_data
from src.sales_analysis import (
    calculate_sales_kpis,
    monthly_sales,
    rfm_analysis,
    sales_by_country,
    top_products,
)
from src.recommendations import (
    generate_combined_product_recommendations,
    generate_sales_recommendations,
)
from src.text_analysis import (
    clean_review_data,
    product_review_summary,
    rating_distribution,
    review_kpis,
    sentiment_distribution,
    top_review_words,
)
from src.sentiment_models import (
    ml_sentiment_distribution,
    predict_sentiment_for_reviews,
    sentiment_comparison_summary,
    train_tfidf_logistic_regression_model,
)
from src.filters import filter_review_data, filter_sales_data

st.set_page_config(
    page_title="AI Business Insight System",
    page_icon="📊",
    layout="wide",
)


st.title("AI Business Insight System")
st.subheader("Prototyp systemu wspierającego decyzje przedsiębiorstwa")


# SAMPLE_DATA_PATH = Path("data/sample/sample_sales.csv")
# SAMPLE_REVIEWS_PATH = Path("data/sample/sample_reviews.csv")
SAMPLE_DATA_PATH = Path("data/sample/generated_sales.csv")
SAMPLE_REVIEWS_PATH = Path("data/sample/generated_reviews.csv")


@st.cache_data
def get_sample_sales_data():
    raw_data = load_sales_data(SAMPLE_DATA_PATH)
    cleaned_data = clean_sales_data(raw_data)
    return raw_data, cleaned_data

@st.cache_data
def get_sample_review_data():
    raw_data = load_review_data(SAMPLE_REVIEWS_PATH)
    cleaned_data = clean_review_data(raw_data)
    return raw_data, cleaned_data


def process_uploaded_sales_file(uploaded_file):
    raw_data = load_uploaded_sales_data(uploaded_file)
    cleaned_data = clean_sales_data(raw_data)
    return raw_data, cleaned_data

def process_uploaded_review_file(uploaded_file):
    raw_data = load_uploaded_review_data(uploaded_file)
    cleaned_data = clean_review_data(raw_data)
    return raw_data, cleaned_data

def convert_dataframe_to_csv(dataframe):
    """
    Konwertuje DataFrame do formatu CSV gotowego do pobrania w Streamlit.
    """
    return dataframe.to_csv(index=False).encode("utf-8-sig")

st.sidebar.header("Źródło danych")

uploaded_sales_file = st.sidebar.file_uploader(
    "Wgraj plik sprzedażowy CSV/XLSX",
    type=["csv", "xlsx", "xls"],
)

use_sample_sales_data = st.sidebar.checkbox(
    "Użyj przykładowych danych sprzedażowych",
    value=True,
)

st.sidebar.divider()

uploaded_review_file = st.sidebar.file_uploader(
    "Wgraj plik z opiniami CSV/XLSX",
    type=["csv", "xlsx", "xls"],
)

use_sample_review_data = st.sidebar.checkbox(
    "Użyj przykładowych opinii klientów",
    value=True,
)

try:
    if uploaded_sales_file is not None:
        raw_sales_data, filtered_sales_data = process_uploaded_sales_file(uploaded_sales_file)
        sales_data_source_description = f"Plik użytkownika: {uploaded_sales_file.name}"
    elif use_sample_sales_data:
        raw_sales_data, filtered_sales_data = get_sample_sales_data()
        # sales_data_source_description = "Dane przykładowe: data/sample/sample_sales.csv"
        sales_data_source_description = "Dane przykładowe: data/sample/generated_sales.csv"
    else:
        raw_sales_data = None
        filtered_sales_data = None
        sales_data_source_description = "Nie wybrano źródła danych sprzedażowych"

except Exception as error:
    raw_sales_data = None
    filtered_sales_data = None
    sales_data_source_description = "Błąd wczytywania danych sprzedażowych"
    st.sidebar.error(f"Błąd danych sprzedażowych: {error}")


try:
    if uploaded_review_file is not None:
        raw_review_data, filtered_review_data = process_uploaded_review_file(uploaded_review_file)
        review_data_source_description = f"Plik użytkownika: {uploaded_review_file.name}"
    elif use_sample_review_data:
        raw_review_data, filtered_review_data = get_sample_review_data()
        # review_data_source_description = "Dane przykładowe: data/sample/sample_reviews.csv"
        review_data_source_description = "Dane przykładowe: data/sample/generated_reviews.csv"
    else:
        raw_review_data = None
        filtered_review_data = None
        review_data_source_description = "Nie wybrano źródła danych tekstowych"

except Exception as error:
    raw_review_data = None
    filtered_review_data = None
    review_data_source_description = "Błąd wczytywania danych tekstowych"
    st.sidebar.error(f"Błąd danych tekstowych: {error}")

st.sidebar.divider()
st.sidebar.header("Filtry analizy")

filtered_sales_data = filtered_sales_data
filtered_review_data = filtered_review_data

if filtered_sales_data is not None:
    min_date = filtered_sales_data["InvoiceDate"].dt.date.min()
    max_date = filtered_sales_data["InvoiceDate"].dt.date.max()

    selected_date_range = st.sidebar.date_input(
        "Zakres dat sprzedaży",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    available_countries = sorted(filtered_sales_data["Country"].dropna().unique().tolist())

    selected_countries = st.sidebar.multiselect(
        "Kraje",
        options=available_countries,
        default=[],
    )

    available_products = sorted(filtered_sales_data["Description"].dropna().unique().tolist())

    selected_products = st.sidebar.multiselect(
        "Produkty",
        options=available_products,
        default=[],
    )

    filtered_sales_data = filter_sales_data(
        filtered_sales_data,
        date_range=selected_date_range,
        countries=selected_countries,
        products=selected_products,
    )

if filtered_review_data is not None:
    if filtered_sales_data is not None and not filtered_sales_data.empty:
        selected_review_products = sorted(
            filtered_sales_data["Description"].dropna().unique().tolist()
        )
    else:
        selected_review_products = None

    selected_sentiments = st.sidebar.multiselect(
        "Sentyment opinii",
        options=["Pozytywny", "Neutralny", "Negatywny"],
        default=[],
    )

    filtered_review_data = filter_review_data(
        filtered_review_data,
        products=selected_review_products,
        sentiments=selected_sentiments,
    )


tab_intro, tab_data, tab_sales, tab_rfm, tab_reviews, tab_ml, tab_recommendations, tab_export = st.tabs(
    [
        "Opis projektu",
        "Dane",
        "Analiza sprzedaży",
        "Segmentacja RFM",
        "Analiza opinii",
        "Model AI/ML",
        "Rekomendacje",
        "Eksport wyników",
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

    st.subheader("Aktualne źródła danych")
    st.write(f"**Dane sprzedażowe:** {sales_data_source_description}")
    st.write(f"**Dane tekstowe:** {review_data_source_description}")

    st.subheader("Wymagany format danych sprzedażowych")
    st.write(
        """
        Plik sprzedażowy powinien zawierać następujące kolumny:
        """
    )

    st.subheader("Wymagany format danych tekstowych")
    st.code(
        "ReviewID, ProductID, ProductName, Rating, ReviewDate, ReviewText",
        language="text",
    )

    st.code(
        "InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country",
        language="text",
    )


with tab_data:
    st.header("Dane sprzedażowe")

    if raw_sales_data is None or filtered_sales_data is None:
        st.warning("Nie wczytano danych. Wgraj plik w panelu bocznym lub zaznacz użycie danych przykładowych.")
    else:
        st.success(f"Źródło danych: {sales_data_source_description}")

        st.subheader("Podgląd danych surowych")
        st.dataframe(raw_sales_data, width="stretch")

        st.subheader("Podgląd danych po czyszczeniu")
        st.dataframe(filtered_sales_data, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Liczba rekordów surowych", len(raw_sales_data))
        col2.metric("Liczba rekordów po czyszczeniu", len(filtered_sales_data))
        col3.metric("Liczba kolumn", filtered_sales_data.shape[1])

        st.metric("Liczba rekordów sprzedażowych po zastosowaniu filtrów", len(filtered_sales_data))


with tab_sales:
    st.header("Analiza sprzedaży")

    if filtered_sales_data is None:
        st.warning("Brak danych do analizy. Wgraj plik sprzedażowy lub użyj danych przykładowych.")
    else:
        kpis = calculate_sales_kpis(filtered_sales_data)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Przychód", f"{kpis['total_revenue']:,.2f}")
        col2.metric("Transakcje", f"{kpis['total_transactions']}")
        col3.metric("Klienci", f"{kpis['total_customers']}")
        col4.metric("Produkty", f"{kpis['total_products']}")
        col5.metric("Śr. wartość zamówienia", f"{kpis['average_order_value']:,.2f}")

        st.divider()

        st.subheader("Sprzedaż w czasie")
        monthly_data = monthly_sales(filtered_sales_data)

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
        top_products_data = top_products(filtered_sales_data, limit=10)

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
        country_data = sales_by_country(filtered_sales_data)

        fig_country = px.bar(
            country_data,
            x="Country",
            y="Revenue",
            title="Sprzedaż według kraju",
        )

        st.plotly_chart(fig_country, width="stretch")
        st.dataframe(country_data, width="stretch")

        st.divider()

        st.subheader("Dane tekstowe — opinie klientów")

        if raw_review_data is None or filtered_review_data is None:
            st.warning("Nie wczytano danych tekstowych. Wgraj plik z opiniami lub użyj danych przykładowych.")
        else:
            st.success(f"Źródło danych tekstowych: {review_data_source_description}")

            st.write("Podgląd surowych opinii")
            st.dataframe(raw_review_data, width="stretch")

            st.write("Podgląd opinii po czyszczeniu")
            st.dataframe(filtered_review_data, width="stretch")

            col1, col2, col3 = st.columns(3)
            col1.metric("Liczba opinii surowych", len(raw_review_data))
            col2.metric("Liczba opinii po czyszczeniu", len(filtered_review_data))
            col3.metric("Liczba produktów z opiniami", filtered_review_data["ProductName"].nunique())

            st.metric("Liczba opinii po zastosowaniu filtrów", len(filtered_review_data))

with tab_rfm:
    st.header("Segmentacja klientów metodą RFM")

    if filtered_sales_data is None:
        st.warning("Brak danych do analizy RFM. Wgraj plik sprzedażowy lub użyj danych przykładowych.")
    else:
        st.write(
            """
            Segmentacja RFM pozwala ocenić klientów na podstawie aktualności zakupów,
            częstotliwości transakcji oraz łącznej wartości zakupów. Dzięki temu
            przedsiębiorstwo może identyfikować najcenniejszych klientów, klientów
            lojalnych oraz grupy wymagające działań retencyjnych.
            """
        )

        rfm_data = rfm_analysis(filtered_sales_data)

        if rfm_data.empty:
            st.warning("Nie udało się wykonać analizy RFM, ponieważ brakuje danych o klientach.")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Liczba klientów w RFM", rfm_data["CustomerID"].nunique())
            col2.metric("Średnia wartość klienta", f"{rfm_data['Monetary'].mean():,.2f}")
            col3.metric("Średnia liczba transakcji", f"{rfm_data['Frequency'].mean():,.2f}")

            st.divider()

            st.subheader("Liczba klientów w segmentach")

            segment_counts = (
                rfm_data.groupby("Segment", as_index=False)
                .agg(Customers=("CustomerID", "nunique"))
                .sort_values("Customers", ascending=False)
            )

            fig_segments = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                title="Liczba klientów według segmentów RFM",
            )

            st.plotly_chart(fig_segments, width="stretch")
            st.dataframe(segment_counts, width="stretch")

            st.divider()

            st.subheader("Wartość sprzedaży według segmentów")

            segment_value = (
                rfm_data.groupby("Segment", as_index=False)
                .agg(
                    Customers=("CustomerID", "nunique"),
                    TotalValue=("Monetary", "sum"),
                    AverageValue=("Monetary", "mean"),
                    AverageFrequency=("Frequency", "mean"),
                    AverageRecency=("Recency", "mean"),
                )
                .sort_values("TotalValue", ascending=False)
            )

            fig_segment_value = px.bar(
                segment_value,
                x="Segment",
                y="TotalValue",
                title="Łączna wartość zakupów według segmentów RFM",
            )

            st.plotly_chart(fig_segment_value, width="stretch")
            st.dataframe(segment_value, width="stretch")

            st.divider()

            st.subheader("Tabela klientów RFM")

            selected_segment = st.selectbox(
                "Wybierz segment klientów",
                options=["Wszystkie segmenty"] + sorted(rfm_data["Segment"].unique().tolist()),
            )

            if selected_segment != "Wszystkie segmenty":
                displayed_rfm_data = rfm_data[rfm_data["Segment"] == selected_segment]
            else:
                displayed_rfm_data = rfm_data

            st.dataframe(displayed_rfm_data, width="stretch")

            st.divider()

            st.subheader("Interpretacja biznesowa")

            st.write(
                """
                Wyniki segmentacji RFM mogą zostać wykorzystane do podejmowania decyzji
                dotyczących marketingu, sprzedaży i utrzymania klientów. Segment
                „Najlepsi klienci” może być objęty programem lojalnościowym, segment
                „Klienci zagrożeni odejściem” wymaga działań retencyjnych, natomiast
                „Nowi lub okazjonalni klienci” mogą być adresatami kampanii aktywizujących.
                """
            )        

with tab_reviews:
    st.header("Analiza danych tekstowych — opinie klientów")

    if filtered_review_data is None:
        st.warning("Brak danych tekstowych do analizy. Wgraj plik z opiniami lub użyj danych przykładowych.")
    else:
        st.write(
            """
            Ta sekcja prezentuje analizę opinii klientów. Program wykonuje
            podstawowe czyszczenie tekstu, analizuje rozkład ocen, klasyfikuje
            sentyment opinii na podstawie ocen gwiazdkowych oraz identyfikuje
            najczęściej występujące słowa.
            """
        )

        text_kpis = review_kpis(filtered_review_data)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Liczba opinii", text_kpis["total_reviews"])
        col2.metric("Średnia ocena", f"{text_kpis['average_rating']:.2f}")
        col3.metric("Śr. długość opinii", f"{text_kpis['average_review_length']:.0f} znaków")
        col4.metric("Produkty z opiniami", text_kpis["unique_products"])

        st.divider()

        st.subheader("Rozkład ocen klientów")

        rating_data = rating_distribution(filtered_review_data)

        fig_ratings = px.bar(
            rating_data,
            x="Rating",
            y="Reviews",
            title="Liczba opinii według ocen gwiazdkowych",
        )

        st.plotly_chart(fig_ratings, width="stretch")
        st.dataframe(rating_data, width="stretch")

        st.divider()

        st.subheader("Rozkład sentymentu opinii")

        sentiment_data = sentiment_distribution(filtered_review_data)

        fig_sentiment = px.pie(
            sentiment_data,
            names="RatingSentiment",
            values="Reviews",
            title="Udział opinii pozytywnych, neutralnych i negatywnych",
        )

        st.plotly_chart(fig_sentiment, width="stretch")
        st.dataframe(sentiment_data, width="stretch")

        st.divider()

        st.subheader("Najczęściej występujące słowa w opiniach")

        top_words_data = top_review_words(filtered_review_data, limit=20)

        if top_words_data.empty:
            st.warning("Brak słów do wyświetlenia.")
        else:
            fig_words = px.bar(
                top_words_data,
                x="Count",
                y="Word",
                orientation="h",
                title="Najczęściej występujące słowa w opiniach",
            )

            st.plotly_chart(fig_words, width="stretch")
            st.dataframe(top_words_data, width="stretch")

        st.divider()

        st.subheader("Podsumowanie opinii według produktów")

        product_reviews_data = product_review_summary(filtered_review_data)

        st.dataframe(product_reviews_data, width="stretch")

        st.divider()

        st.subheader("Przykładowe opinie")

        selected_sentiment = st.selectbox(
            "Filtruj opinie według sentymentu",
            options=["Wszystkie", "Pozytywny", "Neutralny", "Negatywny"],
        )

        if selected_sentiment != "Wszystkie":
            displayed_reviews = filtered_review_data[
                filtered_review_data["RatingSentiment"] == selected_sentiment
            ]
        else:
            displayed_reviews = filtered_review_data

        st.dataframe(
            displayed_reviews[
                [
                    "ProductName",
                    "Rating",
                    "RatingSentiment",
                    "ReviewText",
                    "CleanReviewText",
                ]
            ],
            width="stretch",
        )

with tab_ml:
    st.header("Model AI/ML — klasyfikacja sentymentu opinii")

    if filtered_review_data is None:
        st.warning("Brak danych tekstowych do trenowania modelu. Wgraj plik z opiniami lub użyj danych przykładowych.")
    else:
        st.write(
            """
            W tej sekcji zastosowano model uczenia maszynowego do klasyfikacji
            sentymentu opinii klientów. Tekst opinii został przekształcony do
            reprezentacji liczbowej metodą TF-IDF, a następnie sklasyfikowany
            za pomocą modelu Logistic Regression.
            """
        )

        st.info(
            """
            Etykietą uczącą modelu jest sentyment wyznaczony na podstawie oceny
            gwiazdkowej. Oznacza to, że oceny 1–2 traktowane są jako negatywne,
            ocena 3 jako neutralna, a oceny 4–5 jako pozytywne.
            """
        )

        try:
            model_results = train_tfidf_logistic_regression_model(filtered_review_data)

            trained_model = model_results["model"]
            predicted_reviews = predict_sentiment_for_reviews(
                filtered_review_data,
                trained_model,
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Liczba opinii treningowych", model_results["train_size"])
            col2.metric("Liczba opinii testowych", model_results["test_size"])
            col3.metric("Liczba klas", len(model_results["classes"]))
            col4.metric("Dokładność modelu", f"{model_results['accuracy']:.2%}")

            st.divider()

            st.subheader("Macierz pomyłek")

            st.write(
                """
                Macierz pomyłek pokazuje, ile opinii z danej klasy rzeczywistej
                zostało przypisanych przez model do poszczególnych klas predykcji.
                """
            )

            st.dataframe(model_results["confusion_matrix"], width="stretch")

            st.divider()

            st.subheader("Raport klasyfikacji")

            st.write(
                """
                Raport klasyfikacji prezentuje podstawowe miary jakości modelu,
                takie jak precision, recall i f1-score dla poszczególnych klas.
                """
            )

            st.dataframe(model_results["classification_report"], width="stretch")

            st.divider()

            st.subheader("Rozkład sentymentu przewidzianego przez model ML")

            ml_distribution = ml_sentiment_distribution(predicted_reviews)

            fig_ml_sentiment = px.pie(
                ml_distribution,
                names="MLSentiment",
                values="Reviews",
                title="Udział klas sentymentu przewidzianych przez model ML",
            )

            st.plotly_chart(fig_ml_sentiment, width="stretch")
            st.dataframe(ml_distribution, width="stretch")

            st.divider()

            st.subheader("Porównanie metody bazowej i modelu ML")

            comparison_data = sentiment_comparison_summary(predicted_reviews)

            fig_comparison = px.bar(
                comparison_data,
                x="RatingSentiment",
                y="Reviews",
                color="MLSentiment",
                barmode="group",
                title="Porównanie sentymentu z oceny gwiazdkowej i modelu ML",
            )

            st.plotly_chart(fig_comparison, width="stretch")
            st.dataframe(comparison_data, width="stretch")

            st.divider()

            st.subheader("Przykładowe predykcje modelu")

            st.dataframe(
                predicted_reviews[
                    [
                        "ProductName",
                        "Rating",
                        "RatingSentiment",
                        "MLSentiment",
                        "SentimentAgreement",
                        "ReviewText",
                    ]
                ],
                width="stretch",
            )

            st.info(
                """
                Wyniki modelu należy interpretować ostrożnie, szczególnie przy małym
                zbiorze danych. W przypadku większej liczby opinii model może lepiej
                nauczyć się zależności między treścią recenzji a sentymentem.
                """
            )

        except Exception as error:
            st.error(f"Nie udało się wytrenować modelu ML: {error}")

with tab_recommendations:
    st.header("Rekomendacje decyzyjne")

    if filtered_sales_data is None:
        st.warning("Brak danych sprzedażowych do wygenerowania rekomendacji. Wgraj plik sprzedażowy lub użyj danych przykładowych.")
    else:
        st.write(
            """
            Ta sekcja prezentuje automatycznie wygenerowane rekomendacje biznesowe
            na podstawie wyników analizy sprzedaży, rankingu produktów, sprzedaży
            według krajów oraz segmentacji klientów metodą RFM.
            """
        )

        kpis = calculate_sales_kpis(filtered_sales_data)
        top_products_data = top_products(filtered_sales_data, limit=10)
        country_data = sales_by_country(filtered_sales_data)
        rfm_data = rfm_analysis(filtered_sales_data)

        recommendations_data = generate_sales_recommendations(
            kpis=kpis,
            top_products_data=top_products_data,
            country_data=country_data,
            rfm_data=rfm_data,
        )

        high_priority_count = len(
            recommendations_data[recommendations_data["Priorytet"] == "Wysoki"]
        )
        medium_priority_count = len(
            recommendations_data[recommendations_data["Priorytet"] == "Średni"]
        )
        low_priority_count = len(
            recommendations_data[recommendations_data["Priorytet"] == "Niski"]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Liczba rekomendacji", len(recommendations_data))
        col2.metric("Wysoki priorytet", high_priority_count)
        col3.metric("Średni priorytet", medium_priority_count)
        col4.metric("Niski priorytet", low_priority_count)

        st.divider()

        selected_priority = st.selectbox(
            "Filtruj według priorytetu",
            options=["Wszystkie", "Wysoki", "Średni", "Niski"],
        )

        if selected_priority != "Wszystkie":
            displayed_recommendations = recommendations_data[
                recommendations_data["Priorytet"] == selected_priority
            ]
        else:
            displayed_recommendations = recommendations_data

        for index, row in displayed_recommendations.iterrows():
            with st.container(border=True):
                st.subheader(f"{row['Obszar']} — priorytet: {row['Priorytet']}")
                st.write(f"**Wniosek:** {row['Wniosek']}")
                st.write(f"**Rekomendacja:** {row['Rekomendacja']}")

        st.divider()

        st.subheader("Tabela rekomendacji sprzedażowych")

        st.dataframe(recommendations_data, width="stretch")

        st.divider()

        st.subheader("Rekomendacje łączące sprzedaż i opinie klientów")

        if filtered_review_data is None:
            st.warning(
                "Brak danych tekstowych. Aby wygenerować rekomendacje łączone, wgraj plik z opiniami lub użyj danych przykładowych."
            )
        else:
            product_reviews_data = product_review_summary(filtered_review_data)

            combined_recommendations_data = generate_combined_product_recommendations(
                top_products_data=top_products_data,
                product_reviews_data=product_reviews_data,
            )

            combined_high_priority_count = len(
                combined_recommendations_data[
                    combined_recommendations_data["Priorytet"] == "Wysoki"
                ]
            )

            col1, col2 = st.columns(2)

            col1.metric("Liczba rekomendacji łączonych", len(combined_recommendations_data))
            col2.metric("Rekomendacje wysokiego priorytetu", combined_high_priority_count)

            selected_combined_priority = st.selectbox(
                "Filtruj rekomendacje łączone według priorytetu",
                options=["Wszystkie", "Wysoki", "Średni", "Niski"],
            )

            if selected_combined_priority != "Wszystkie":
                displayed_combined_recommendations = combined_recommendations_data[
                    combined_recommendations_data["Priorytet"] == selected_combined_priority
                ]
            else:
                displayed_combined_recommendations = combined_recommendations_data

            for _, row in displayed_combined_recommendations.iterrows():
                with st.container(border=True):
                    st.subheader(f"{row['Produkt']} — {row['Obszar']}")
                    st.write(f"**Priorytet:** {row['Priorytet']}")
                    st.write(f"**Wniosek:** {row['Wniosek']}")
                    st.write(f"**Rekomendacja:** {row['Rekomendacja']}")

            st.subheader("Tabela rekomendacji łączonych")
            st.dataframe(combined_recommendations_data, width="stretch")

        st.info(
            """
            Rekomendacje mają charakter wspierający i powinny być interpretowane
            przez analityka lub menedżera w kontekście rzeczywistej sytuacji
            przedsiębiorstwa. System nie zastępuje decyzji biznesowej, lecz
            wskazuje obszary wymagające uwagi.
            """
        )

with tab_export:
    st.header("Eksport wyników analizy")

    if filtered_sales_data is None:
        st.warning("Brak danych do eksportu. Wgraj plik sprzedażowy lub użyj danych przykładowych.")
    else:
        st.write(
            """
            Ta sekcja umożliwia pobranie wyników analizy w formacie CSV.
            Wyeksportowane dane mogą zostać wykorzystane do dalszego raportowania,
            dokumentacji badania lub dodatkowej analizy poza aplikacją.
            """
        )

        monthly_data = monthly_sales(filtered_sales_data)
        top_products_data = top_products(filtered_sales_data, limit=10)
        country_data = sales_by_country(filtered_sales_data)
        rfm_data = rfm_analysis(filtered_sales_data)
        kpis = calculate_sales_kpis(filtered_sales_data)
        recommendations_data = generate_sales_recommendations(
            kpis=kpis,
            top_products_data=top_products_data,
            country_data=country_data,
            rfm_data=rfm_data,
        )  

        if filtered_review_data is not None:
            product_reviews_data = product_review_summary(filtered_review_data)
            combined_recommendations_data = generate_combined_product_recommendations(
                top_products_data=top_products_data,
                product_reviews_data=product_reviews_data,
            )
        else:
            combined_recommendations_data = None

        st.subheader("Dane po czyszczeniu")

        st.download_button(
            label="Pobierz dane po czyszczeniu CSV",
            data=convert_dataframe_to_csv(filtered_sales_data),
            file_name="filtered_sales_data.csv",
            mime="text/csv",
        )
        if combined_recommendations_data is not None:
            st.download_button(
                label="Pobierz rekomendacje łączące sprzedaż i opinie",
                data=convert_dataframe_to_csv(combined_recommendations_data),
                file_name="combined_product_recommendations.csv",
                mime="text/csv",
            )

        st.subheader("Wyniki analizy sprzedaży")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="Pobierz sprzedaż miesięczną",
                data=convert_dataframe_to_csv(monthly_data),
                file_name="monthly_sales.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="Pobierz TOP produkty",
                data=convert_dataframe_to_csv(top_products_data),
                file_name="top_products.csv",
                mime="text/csv",
            )

        with col3:
            st.download_button(
                label="Pobierz sprzedaż według kraju",
                data=convert_dataframe_to_csv(country_data),
                file_name="sales_by_country.csv",
                mime="text/csv",
            )

        st.subheader("Wyniki segmentacji klientów")

        if rfm_data.empty:
            st.warning("Brak danych RFM do eksportu.")
        else:
            st.download_button(
                label="Pobierz segmentację RFM",
                data=convert_dataframe_to_csv(rfm_data),
                file_name="rfm_segmentation.csv",
                mime="text/csv",
            )

            st.subheader("Rekomendacje decyzyjne")

        st.download_button(
            label="Pobierz rekomendacje decyzyjne",
            data=convert_dataframe_to_csv(recommendations_data),
            file_name="business_recommendations.csv",
            mime="text/csv",
        )

        st.divider()

        st.info(
            """
            Funkcja eksportu wyników zwiększa praktyczną użyteczność prototypu,
            ponieważ pozwala przenieść rezultaty analizy do arkusza kalkulacyjnego
            lub wykorzystać je jako załączniki do raportu biznesowego.
            """
        )            