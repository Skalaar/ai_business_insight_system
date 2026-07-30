from pathlib import Path

import plotly.express as px
import streamlit as st
import pandas as pd

from src.data_loader import (
    load_review_data,
    load_sales_data,
    load_uploaded_review_data,
    load_uploaded_sales_data,
)
from src.preprocessing import clean_sales_data
from src.sales_forecasting import (
    build_sales_forecast_analysis,
)
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
from src.advanced_recommendations import (
    build_advanced_decision_support,
)
from src.filters import filter_review_data, filter_sales_data
from src.model_comparison import compare_sentiment_models
from src.model_interpretability import (
    build_interpretability_analysis,
    explain_single_review,
)
from src.topic_modeling import build_topic_analysis
from src.transformer_sentiment import (
    DEFAULT_TRANSFORMER_MODEL,
    evaluate_transformer_sentiment,
    load_transformer_sentiment_resources,
)
from src.drift_monitoring import (
    build_drift_monitoring_analysis,
)
from src.app_config import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_PAGE_ICON,
    APP_VERSION,
    DATA_DISCLAIMER,
)
from src.ui_components import (
    inject_global_styles,
    render_app_header,
    render_footer,
    render_sidebar_status,
)
from src.data_quality import (
    build_data_quality_report,
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            f"{APP_NAME}\n\n"
            f"Wersja: {APP_VERSION}\n\n"
            "Prototyp przygotowany na potrzeby pracy magisterskiej."
        ),
    },
)
inject_global_styles()

render_app_header(
    app_name=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


st.subheader("Prototyp systemu wspierającego decyzje przedsiębiorstwa")


# SAMPLE_DATA_PATH = Path("data/sample/sample_sales.csv")
# SAMPLE_REVIEWS_PATH = Path("data/sample/sample_reviews.csv")
SAMPLE_DATA_PATH = Path("data/sample/generated_sales.csv")
SAMPLE_REVIEWS_PATH = Path("data/sample/generated_reviews.csv")
OLIST_SALES_PATH = Path(
    "data/processed/olist/olist_sales_prepared.csv"
)

OLIST_REVIEWS_PATH = Path(
    "data/processed/olist/olist_reviews_prepared.csv"
)
MODEL_COMPARISON_SAMPLE_LIMIT = 10_000
INTERPRETABILITY_SAMPLE_LIMIT = 10_000
TOPIC_MODEL_SAMPLE_LIMIT = 15_000
TRANSFORMER_SAMPLE_LIMIT = 1_000

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

@st.cache_data(show_spinner=False)
def get_olist_sales_data():
    """
    Wczytuje przygotowane rzeczywiste dane sprzedażowe Olist.
    """
    raw_data = load_sales_data(
        OLIST_SALES_PATH
    )

    cleaned_data = clean_sales_data(
        raw_data
    )

    return raw_data, cleaned_data

@st.cache_data(show_spinner=False)
def get_olist_review_data():
    """
    Wczytuje przygotowane opinie klientów Olist.
    """
    raw_data = load_review_data(
        OLIST_REVIEWS_PATH
    )

    cleaned_data = clean_review_data(
        raw_data
    )

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

@st.cache_resource(show_spinner=False)
def get_model_comparison_results(
    review_data,
    requested_folds: int,
    max_samples: int = (
        MODEL_COMPARISON_SAMPLE_LIMIT
    ),
):
    """
    Uruchamia porównanie modeli na kontrolowanej,
    reprezentatywnej próbce opinii.
    """
    return compare_sentiment_models(
        data=review_data,
        requested_folds=requested_folds,
        random_state=42,
        text_language="multilingual",
        max_samples=max_samples,
    )

@st.cache_resource(show_spinner=False)
def get_interpretability_results(
    review_data,
    top_n: int,
):
    """
    Trenuje i przechowuje model interpretowalny
    oraz jego globalne wyjaśnienia.
    """
    return build_interpretability_analysis(
        data=review_data,
        top_n=top_n,
        text_language="multilingual",
        random_state=42,
        max_samples=(
            INTERPRETABILITY_SAMPLE_LIMIT
        ),
    )

@st.cache_resource(show_spinner=False)
def get_topic_analysis_results(
    review_data,
    number_of_topics: int,
    top_terms_per_topic: int,
):
    """
    Uruchamia modelowanie tematów na kontrolowanej
    próbce i przechowuje wynik w pamięci podręcznej.
    """
    return build_topic_analysis(
        data=review_data,
        number_of_topics=number_of_topics,
        top_terms_per_topic=top_terms_per_topic,
        random_state=42,
        text_language="multilingual",
        max_samples=TOPIC_MODEL_SAMPLE_LIMIT,
    )

@st.cache_data(show_spinner=False)
def get_sales_forecast_results(
    sales_data,
    test_fraction: float,
    forecast_horizon: int,
):
    """
    Uruchamia analizę prognostyczną i przechowuje
    jej wyniki w pamięci podręcznej Streamlit.
    """
    return build_sales_forecast_analysis(
        data=sales_data,
        test_fraction=test_fraction,
        forecast_horizon=forecast_horizon,
    )

@st.cache_data(show_spinner=False)
def get_advanced_decision_results(
    sales_data,
    review_data,
    topic_assignments,
    weekly_data,
    future_forecast,
):
    """
    Integruje sprzedaż, sentyment, tematy opinii
    oraz prognozę sprzedaży.
    """
    return build_advanced_decision_support(
        sales_data=sales_data,
        review_data=review_data,
        topic_assignments=topic_assignments,
        weekly_data=weekly_data,
        future_forecast=future_forecast,
        trend_window_weeks=8,
    )

@st.cache_resource(show_spinner=False)
def get_transformer_resources(
    model_name: str,
):
    """
    Ładuje i przechowuje model transformerowy.
    """
    return load_transformer_sentiment_resources(
        model_name=model_name,
    )

@st.cache_data(show_spinner=False)
def get_transformer_evaluation_results(
    review_data,
    model_name: str,
    batch_size: int,
):
    """
    Przeprowadza i przechowuje predykcje
    modelu transformerowego.
    """

    resources = get_transformer_resources(
        model_name=model_name,
    )

    return evaluate_transformer_sentiment(
        data=review_data,
        resources=resources,
        batch_size=batch_size,
        max_length=256,
        max_samples=TRANSFORMER_SAMPLE_LIMIT,
        random_state=42,
    )

@st.cache_data(show_spinner=False)
def get_drift_monitoring_results(
    sales_data,
    review_data,
    window_weeks: int,
):
    """
    Porównuje dwa kolejne okresy sprzedaży i opinii.
    """
    return build_drift_monitoring_analysis(
        sales_data=sales_data,
        review_data=review_data,
        window_weeks=window_weeks,
        top_terms=40,
    )

@st.cache_data(show_spinner=False)
def get_data_quality_results(
    sales_data,
    review_data,
    minimum_review_length: int,
):
    return build_data_quality_report(
        sales_data=sales_data,
        review_data=review_data,
        minimum_review_length=minimum_review_length,
    )

st.sidebar.header("Źródło danych")

selected_data_source = st.sidebar.radio(
    "Wybierz zestaw danych",
    options=[
        "Olist — dane rzeczywiste",
        "Dane syntetyczne",
        "Własne pliki",
    ],
    index=0,
    help=(
        "Olist jest głównym rzeczywistym zbiorem "
        "wykorzystywanym w części badawczej."
    ),
)

uploaded_sales_file = None
uploaded_review_file = None

if selected_data_source == "Własne pliki":
    uploaded_sales_file = st.sidebar.file_uploader(
        "Wgraj plik sprzedażowy CSV/XLSX",
        type=["csv", "xlsx", "xls"],
        key="uploaded_sales_file",
    )

    uploaded_review_file = st.sidebar.file_uploader(
        "Wgraj plik z opiniami CSV/XLSX",
        type=["csv", "xlsx", "xls"],
        key="uploaded_review_file",
    )


try:
    if selected_data_source == "Olist — dane rzeczywiste":
        with st.spinner(
            "Trwa wczytywanie rzeczywistych danych Olist..."
        ):
            (
                raw_sales_data,
                filtered_sales_data,
            ) = get_olist_sales_data()

        sales_data_source_description = (
            "Olist — rzeczywiste dane e-commerce: "
            "data/processed/olist/olist_sales_prepared.csv"
        )

    elif selected_data_source == "Dane syntetyczne":
        (
            raw_sales_data,
            filtered_sales_data,
        ) = get_sample_sales_data()

        sales_data_source_description = (
            "Dane syntetyczne: "
            "data/sample/generated_sales.csv"
        )

    elif uploaded_sales_file is not None:
        (
            raw_sales_data,
            filtered_sales_data,
        ) = process_uploaded_sales_file(
            uploaded_sales_file
        )

        sales_data_source_description = (
            f"Plik użytkownika: "
            f"{uploaded_sales_file.name}"
        )

    else:
        raw_sales_data = None
        filtered_sales_data = None

        sales_data_source_description = (
            "Nie wgrano pliku sprzedażowego"
        )

except Exception as error:
    raw_sales_data = None
    filtered_sales_data = None

    sales_data_source_description = (
        "Błąd wczytywania danych sprzedażowych"
    )

    st.sidebar.error(
        f"Błąd danych sprzedażowych: {error}"
    )

try:
    if selected_data_source == "Olist — dane rzeczywiste":
        with st.spinner(
            "Trwa wczytywanie opinii klientów Olist..."
        ):
            (
                raw_review_data,
                filtered_review_data,
            ) = get_olist_review_data()

        review_data_source_description = (
            "Olist — rzeczywiste opinie klientów: "
            "data/processed/olist/olist_reviews_prepared.csv"
        )

    elif selected_data_source == "Dane syntetyczne":
        (
            raw_review_data,
            filtered_review_data,
        ) = get_sample_review_data()

        review_data_source_description = (
            "Dane syntetyczne: "
            "data/sample/generated_reviews.csv"
        )

    elif uploaded_review_file is not None:
        (
            raw_review_data,
            filtered_review_data,
        ) = process_uploaded_review_file(
            uploaded_review_file
        )

        review_data_source_description = (
            f"Plik użytkownika: "
            f"{uploaded_review_file.name}"
        )

    else:
        raw_review_data = None
        filtered_review_data = None

        review_data_source_description = (
            "Nie wgrano pliku z opiniami"
        )

except Exception as error:
    raw_review_data = None
    filtered_review_data = None

    review_data_source_description = (
        "Błąd wczytywania danych tekstowych"
    )

    st.sidebar.error(
        f"Błąd danych tekstowych: {error}"
    )

st.sidebar.divider()

st.sidebar.header("Filtry analizy")

if (
    filtered_sales_data is not None
    and not filtered_sales_data.empty
):
    min_date = (
        filtered_sales_data["InvoiceDate"]
        .dt.date
        .min()
    )

    max_date = (
        filtered_sales_data["InvoiceDate"]
        .dt.date
        .max()
    )

    selected_date_range = st.sidebar.date_input(
        "Zakres dat sprzedaży",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    available_countries = sorted(
        filtered_sales_data["Country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_countries = st.sidebar.multiselect(
        "Kraje",
        options=available_countries,
        default=[],
    )

    selected_categories = []

    if "Category" in filtered_sales_data.columns:
        available_categories = sorted(
            filtered_sales_data["Category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_categories = st.sidebar.multiselect(
            "Kategorie produktów",
            options=available_categories,
            default=[],
        )

    product_option_data = filtered_sales_data

    if selected_categories:
        product_option_data = (
            product_option_data[
                product_option_data[
                    "Category"
                ].isin(
                    selected_categories
                )
            ]
        )

    number_of_available_products = int(
        product_option_data[
            "Description"
        ].nunique()
    )

    if (
        number_of_available_products > 2000
        and not selected_categories
    ):
        available_products = []

        st.sidebar.caption(
            "Zbiór zawiera ponad 2000 produktów. "
            "Wybierz najpierw kategorię, aby włączyć "
            "filtr konkretnych produktów."
        )

    else:
        available_products = sorted(
            product_option_data[
                "Description"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_products = st.sidebar.multiselect(
        "Produkty",
        options=available_products,
        default=[],
        disabled=not available_products,
    )

    filtered_sales_data = filter_sales_data(
        filtered_sales_data,
        date_range=selected_date_range,
        countries=selected_countries,
        categories=selected_categories,
        products=selected_products,
    )

if (
    filtered_review_data is not None
    and not filtered_review_data.empty
):
    if (
        filtered_sales_data is not None
        and not filtered_sales_data.empty
    ):
        selected_review_products = sorted(
            filtered_sales_data[
                "Description"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:
        selected_review_products = None

    selected_sentiments = st.sidebar.multiselect(
        "Sentyment opinii",
        options=[
            "Pozytywny",
            "Neutralny",
            "Negatywny",
        ],
        default=[],
    )

    filtered_review_data = filter_review_data(
        filtered_review_data,
        products=selected_review_products,
        sentiments=selected_sentiments,
    )

render_sidebar_status(
    sales_data=filtered_sales_data,
    review_data=filtered_review_data,
    version=APP_VERSION,
)

(
    tab_intro,
    tab_data,
    tab_quality,
    tab_sales,
    tab_forecasting,
    tab_rfm,
    tab_reviews,
    tab_ml,
    tab_model_comparison,
    tab_transformer,
    tab_interpretability,
    tab_topics,
    tab_decision_center,
    tab_monitoring,
    tab_recommendations,
    tab_summary,
    tab_export,
) = st.tabs(
    [
        'Opis projektu',
        'Dane',
        'Jakość danych',
        'Analiza sprzedaży',
        'Prognozowanie',
        'Segmentacja RFM',
        'Analiza opinii',
        'Model bazowy',
        'Porównanie modeli',
        'Model transformerowy',
        'Interpretowalność AI',
        'Tematy opinii',
        'Centrum decyzji',
        'Monitoring i drift',
        'Rekomendacje',
        'Podsumowanie badania',
        'Eksport wyników',
    ],
    default="Opis projektu",
    key="main_analysis_tabs",
    on_change="rerun",
)

if tab_intro.open:
    with tab_intro:
        st.header("Opis projektu")

        st.write(
            """
            Aplikacja stanowi prototyp zintegrowanego systemu
            analitycznego wspierającego procesy decyzyjne
            przedsiębiorstwa. System łączy analizę danych
            sprzedażowych i tekstowych z metodami uczenia
            maszynowego, przetwarzania języka naturalnego,
            prognozowania oraz analityki biznesowej.
            """
        )

        st.info(
            """
            Wersja 3.0 obejmuje analizę sprzedaży,
            segmentację klientów RFM, klasyfikację sentymentu,
            porównanie modeli ML, model transformerowy,
            interpretowalność AI, modelowanie tematów,
            prognozowanie sprzedaży, monitoring driftu
            oraz zintegrowane centrum wspomagania decyzji.
            """
        )

        st.subheader("Aktualne źródła danych")
        st.write(f"**Dane sprzedażowe:** {sales_data_source_description}")
        st.write(f"**Dane tekstowe:** {review_data_source_description}")

        st.subheader("Wymagany format danych sprzedażowych")
        st.write("Plik sprzedażowy powinien zawierać następujące kolumny:")
        st.code(
            "InvoiceNo, StockCode, Description, Quantity, "
            "InvoiceDate, UnitPrice, CustomerID, Country",
            language=None,
        )

        st.subheader("Wymagany format danych tekstowych")
        st.write("Plik opinii powinien zawierać następujące kolumny:")
        st.code(
            "ReviewID, ProductID, ProductName, Rating, "
            "ReviewDate, ReviewText",
            language=None,
        )

if tab_data.open:
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

if tab_quality.open:
    with tab_quality:
        st.header("Kontrola jakości danych")

        st.write(
            """
            Moduł automatycznie ocenia kompletność, poprawność
            i spójność danych sprzedażowych oraz opinii klientów.
            Wynik jakości uwzględnia wagę poszczególnych kontroli.
            """
        )

        minimum_review_length = st.slider(
            "Minimalna długość opinii",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

        if (
            filtered_sales_data is None
            or filtered_sales_data.empty
            or filtered_review_data is None
            or filtered_review_data.empty
        ):
            st.warning(
                "Kontrola jakości wymaga danych sprzedażowych "
                "oraz danych opinii."
            )
        else:
            try:
                quality_results = (
                    get_data_quality_results(
                        sales_data=filtered_sales_data,
                        review_data=filtered_review_data,
                        minimum_review_length=(
                            minimum_review_length
                        ),
                    )
                )

                quality_summary = quality_results[
                    "summary"
                ]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Ogólny wynik jakości",
                    (
                        f"{quality_summary['OverallQualityScore']:.2f}%"
                    ),
                )

                col2.metric(
                    "Ocena jakości",
                    quality_summary[
                        "QualityStatus"
                    ],
                )

                col3.metric(
                    "Kontrole zakończone poprawnie",
                    (
                        f"{quality_summary['PassedChecks']} "
                        f"/ {quality_summary['TotalChecks']}"
                    ),
                )

                col4.metric(
                    "Nieudane kontrole wysokiej wagi",
                    quality_summary[
                        "HighSeverityFailedChecks"
                    ],
                )

                col5, col6, col7 = st.columns(3)

                col5.metric(
                    "Rekordy sprzedażowe",
                    quality_summary[
                        "SalesRecords"
                    ],
                )

                col6.metric(
                    "Opinie klientów",
                    quality_summary[
                        "ReviewRecords"
                    ],
                )

                col7.metric(
                    "Niepowiązane produkty opinii",
                    quality_summary[
                        "UnmatchedReviewProducts"
                    ],
                )

                st.divider()

                st.subheader(
                    "Podsumowanie według zbiorów"
                )

                st.dataframe(
                    quality_results[
                        "dataset_summary"
                    ].style.format(
                        {
                            "AveragePassRate": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Wyniki wszystkich kontroli"
                )

                quality_checks = quality_results[
                    "quality_checks"
                ]

                st.dataframe(
                    quality_checks.style.format(
                        {
                            "IssueRate": "{:.2%}",
                            "PassRate": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                if (
                    quality_checks["IssueRate"] > 0
                ).any():
                    quality_chart = px.bar(
                        quality_checks.sort_values(
                            by="IssueRate",
                            ascending=True,
                        ),
                        x="IssueRate",
                        y="Check",
                        color="Severity",
                        orientation="h",
                        title=(
                            "Odsetek problematycznych rekordów "
                            "według kontroli"
                        ),
                        range_x=[0, 1],
                    )

                    st.plotly_chart(
                        quality_chart,
                        width="stretch",
                    )
                else:
                    st.success(
                        "Wszystkie kontrole zakończyły się powodzeniem. "
                        "Nie wykryto problemów do przedstawienia na wykresie."
                    )

                st.divider()

                st.subheader(
                    "Problemy wymagające uwagi"
                )

                if quality_results[
                    "failed_checks"
                ].empty:
                    st.success(
                        "Nie wykryto problemów jakościowych."
                    )
                else:
                    st.dataframe(
                        quality_results[
                            "failed_checks"
                        ].style.format(
                            {
                                "IssueRate": "{:.2%}",
                                "PassRate": "{:.2%}",
                            }
                        ),
                        width="stretch",
                    )

                st.divider()

                st.subheader(
                    "Zalecenia naprawcze"
                )

                if quality_results[
                    "recommendations"
                ].empty:
                    st.success(
                        "Nie są wymagane działania naprawcze."
                    )
                else:
                    st.dataframe(
                        quality_results[
                            "recommendations"
                        ].style.format(
                            {
                                "IssueRate": "{:.2%}",
                            }
                        ),
                        width="stretch",
                    )

                if not quality_results[
                    "unmatched_products"
                ].empty:
                    st.subheader(
                        "Produkty opinii nieobecne w sprzedaży"
                    )

                    st.dataframe(
                        quality_results[
                            "unmatched_products"
                        ],
                        width="stretch",
                    )

                st.info(
                    """
                    Wynik jakości jest wskaźnikiem pomocniczym.
                    Każdy problem powinien zostać oceniony
                    z uwzględnieniem znaczenia biznesowego,
                    sposobu pozyskania danych oraz zasad
                    przetwarzania zwrotów i korekt.
                    """
                )

            except Exception as error:
                st.error(
                    "Nie udało się przygotować raportu jakości: "
                    f"{error}"
                )

if tab_sales.open:
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

if tab_forecasting.open:
    with tab_forecasting:
        st.header("Prognozowanie sprzedaży")

        st.write(
            """
            Moduł porównuje cztery metody prognozowania tygodniowego
            przychodu. Ocena jest prowadzona za pomocą walidacji kroczącej
            z rozszerzającym się oknem treningowym. W każdym kroku model
            korzysta wyłącznie z informacji dostępnych przed prognozowanym
            tygodniem.
            """
        )

        st.info(
            """
            Najlepszy model wybierany jest na podstawie najniższej wartości
            RMSE. Dodatkowo prezentowane są MAE, MAPE, sMAPE, WAPE
            oraz średnie obciążenie prognozy.
            """
        )

        if filtered_sales_data is None or filtered_sales_data.empty:
            st.warning(
                "Brak danych sprzedażowych do prognozowania."
            )
        else:
            col1, col2 = st.columns(2)

            with col1:
                selected_test_percent = st.slider(
                    "Udział okresów przeznaczonych do walidacji",
                    min_value=20,
                    max_value=40,
                    value=25,
                    step=5,
                    help=(
                        "Końcowa część szeregu jest prognozowana "
                        "w trybie kroczącym i nie jest używana "
                        "do trenowania wcześniejszych prognoz."
                    ),
                )

            with col2:
                selected_forecast_horizon = st.slider(
                    "Horyzont przyszłej prognozy w tygodniach",
                    min_value=4,
                    max_value=16,
                    value=8,
                    step=1,
                )

            try:
                with st.spinner(
                    "Trwa walidacja i porównywanie modeli prognostycznych..."
                ):
                    forecast_results = get_sales_forecast_results(
                        sales_data=filtered_sales_data,
                        test_fraction=(
                            selected_test_percent
                            / 100
                        ),
                        forecast_horizon=(
                            selected_forecast_horizon
                        ),
                    )

                removed_start_period = bool(
                    forecast_results[
                        "removed_incomplete_start_period"
                    ]
                )

                removed_end_period = bool(
                    forecast_results[
                        "removed_incomplete_end_period"
                    ]
                )

                if (
                    removed_start_period
                    or removed_end_period
                ):
                    removed_period_names = []

                    if removed_start_period:
                        removed_period_names.append(
                            "pierwszy tydzień"
                        )

                    if removed_end_period:
                        removed_period_names.append(
                            "ostatni tydzień"
                        )

                    st.info(
                        "Przed utworzeniem szeregu prognostycznego "
                        "pominięto niepełne okresy brzegowe: "
                        f"**{' i '.join(removed_period_names)}**. "
                        "Zapobiega to traktowaniu fragmentu tygodnia "
                        "jako pełnej obserwacji sprzedażowej."
                    )

                forecast_metrics = forecast_results[
                    "metrics"
                ]

                best_forecast_row = forecast_metrics.iloc[
                    0
                ]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Najlepszy model",
                    forecast_results[
                        "best_model_name"
                    ],
                )

                col2.metric(
                    "RMSE",
                    f"{best_forecast_row['RMSE']:,.2f}",
                )

                col3.metric(
                    "MAE",
                    f"{best_forecast_row['MAE']:,.2f}",
                )

                col4.metric(
                    "sMAPE",
                    f"{best_forecast_row['sMAPE']:.2f}%",
                )

                col5, col6, col7, col8 = st.columns(4)

                col5.metric(
                    "Liczba tygodni",
                    forecast_results[
                        "number_of_observations"
                    ],
                )

                col6.metric(
                    "Początkowy zbiór treningowy",
                    forecast_results[
                        "initial_train_size"
                    ],
                )

                col7.metric(
                    "Okresy walidacyjne",
                    forecast_results[
                        "test_size"
                    ],
                )

                col8.metric(
                    "Horyzont przyszły",
                    (
                        f"{forecast_results['forecast_horizon']} tyg."
                    ),
                )

                st.divider()

                st.subheader(
                    "Ranking modeli prognostycznych"
                )

                st.dataframe(
                    forecast_metrics.style.format(
                        {
                            "MAE": "{:,.2f}",
                            "RMSE": "{:,.2f}",
                            "MAPE": "{:.2f}%",
                            "sMAPE": "{:.2f}%",
                            "WAPE": "{:.2f}%",
                            "Bias": "{:,.2f}",
                            "ValidationPeriods": "{:.0f}",
                        }
                    ),
                    width="stretch",
                )

                ranking_figure = px.bar(
                    forecast_metrics.sort_values(
                        by="RMSE",
                        ascending=False,
                    ),
                    x="RMSE",
                    y="Model",
                    orientation="h",
                    title=(
                        "Porównanie modeli według RMSE "
                        "— niższa wartość jest lepsza"
                    ),
                    text="RMSE",
                )

                ranking_figure.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                )

                st.plotly_chart(
                    ranking_figure,
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Walidacja krocząca modeli"
                )

                selected_forecast_model = st.selectbox(
                    "Wybierz model do wyświetlenia",
                    options=forecast_metrics[
                        "Model"
                    ].tolist(),
                    index=0,
                    key="forecast_validation_model",
                )

                selected_validation_predictions = (
                    forecast_results[
                        "validation_predictions"
                    ][
                        forecast_results[
                            "validation_predictions"
                        ]["Model"]
                        == selected_forecast_model
                    ]
                    .copy()
                )

                actual_history_plot = (
                    forecast_results[
                        "weekly_data"
                    ][
                        [
                            "PeriodEnd",
                            "Revenue",
                        ]
                    ]
                    .rename(
                        columns={
                            "Revenue": "Value",
                        }
                    )
                )

                actual_history_plot["Series"] = (
                    "Sprzedaż rzeczywista"
                )

                validation_forecast_plot = (
                    selected_validation_predictions[
                        [
                            "PeriodEnd",
                            "ForecastRevenue",
                        ]
                    ]
                    .rename(
                        columns={
                            "ForecastRevenue": "Value",
                        }
                    )
                )

                validation_forecast_plot["Series"] = (
                    f"Prognoza: {selected_forecast_model}"
                )

                validation_plot_data = pd.concat(
                    [
                        actual_history_plot,
                        validation_forecast_plot,
                    ],
                    ignore_index=True,
                )

                validation_figure = px.line(
                    validation_plot_data,
                    x="PeriodEnd",
                    y="Value",
                    color="Series",
                    markers=True,
                    title=(
                        "Sprzedaż rzeczywista i prognozy "
                        "jednookresowe"
                    ),
                )

                st.plotly_chart(
                    validation_figure,
                    width="stretch",
                )

                st.dataframe(
                    selected_validation_predictions.style.format(
                        {
                            "ActualRevenue": "{:,.2f}",
                            "ForecastRevenue": "{:,.2f}",
                            "Error": "{:,.2f}",
                            "AbsoluteError": "{:,.2f}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Prognoza przyszłej sprzedaży"
                )

                future_forecast = forecast_results[
                    "future_forecast"
                ]

                recent_history = (
                    forecast_results[
                        "weekly_data"
                    ]
                    .tail(20)[
                        [
                            "PeriodEnd",
                            "Revenue",
                        ]
                    ]
                    .rename(
                        columns={
                            "Revenue": "Value",
                        }
                    )
                )

                recent_history["Series"] = (
                    "Sprzedaż historyczna"
                )

                future_plot = (
                    future_forecast[
                        [
                            "PeriodEnd",
                            "ForecastRevenue",
                        ]
                    ]
                    .rename(
                        columns={
                            "ForecastRevenue": "Value",
                        }
                    )
                )

                future_plot["Series"] = (
                    "Prognoza przyszła"
                )

                future_plot_data = pd.concat(
                    [
                        recent_history,
                        future_plot,
                    ],
                    ignore_index=True,
                )

                future_figure = px.line(
                    future_plot_data,
                    x="PeriodEnd",
                    y="Value",
                    color="Series",
                    markers=True,
                    title=(
                        "Prognoza tygodniowego przychodu "
                        f"— {forecast_results['best_model_name']}"
                    ),
                )

                st.plotly_chart(
                    future_figure,
                    width="stretch",
                )

                st.dataframe(
                    future_forecast.style.format(
                        {
                            "ForecastRevenue": "{:,.2f}",
                        }
                    ),
                    width="stretch",
                )

                if (
                    forecast_results["best_model_name"]
                    == "Prognoza naiwna"
                ):
                    st.info(
                        "Najlepszym modelem w walidacji kroczącej "
                        "okazała się prognoza naiwna. Model ten "
                        "przyjmuje ostatnią znaną wartość jako "
                        "prognozę dla każdego przyszłego tygodnia. "
                        "Stały przebieg prognozy jest więc zamierzonym "
                        "wynikiem metody, a nie błędem obliczeniowym."
                    )

                st.divider()

                st.subheader("Interpretacja metryk")

                st.write(
                    """
                    **MAE** określa przeciętną bezwzględną różnicę pomiędzy
                    prognozą i wynikiem rzeczywistym. **RMSE** silniej
                    penalizuje duże błędy. **MAPE**, **sMAPE** i **WAPE**
                    przedstawiają błędy w ujęciu procentowym. **Bias**
                    większy od zera oznacza przeciętne zawyżanie prognoz,
                    natomiast wartość ujemna wskazuje na ich zaniżanie.
                    """
                )

                if selected_data_source == "Dane syntetyczne":
                    st.warning(
                        """
                        Prognoza korzysta z danych syntetycznych, których
                        sprzedaż została wygenerowana losowo. Wyniki służą
                        przede wszystkim do weryfikacji poprawności procedury
                        prognostycznej.
                        """
                    )
                else:
                    st.info(
                        """
                        Prognoza została wyznaczona na podstawie aktualnie
                        wybranego źródła danych. Jej interpretacja powinna
                        uwzględniać długość szeregu, sezonowość, kompletność
                        końcowych okresów oraz błędy uzyskane w walidacji
                        kroczącej.
                        """
                    )

            except Exception as error:
                st.error(
                    "Nie udało się przeprowadzić prognozowania: "
                    f"{error}"
                )

if tab_rfm.open:
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

if tab_reviews.open:
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

if tab_ml.open:
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

if tab_model_comparison.open:
    with tab_model_comparison:
        st.header("Porównanie modeli klasyfikacji sentymentu")

        st.write(
            """
            W tej części porównywane są cztery klasyczne algorytmy uczenia
            maszynowego. Wszystkie modele korzystają z tej samej reprezentacji
            TF-IDF, dzięki czemu różnice wyników można przypisać przede wszystkim
            zastosowanemu klasyfikatorowi.
            """
        )

        st.info(
            """
            Podstawą wyboru najlepszego modelu jest macro F1, które nadaje
            jednakowe znaczenie każdej klasie sentymentu. Accuracy pozostaje
            miarą pomocniczą.
            """
        )

        if filtered_review_data is None or filtered_review_data.empty:
            st.warning(
                "Brak danych tekstowych do porównania modeli."
            )
        else:
            requested_folds = st.slider(
                "Liczba części walidacji krzyżowej",
                min_value=3,
                max_value=5,
                value=5,
                step=1,
                help=(
                    "Dane zostaną podzielone na kilka części. "
                    "Każda część zostanie kolejno wykorzystana do testowania."
                ),
            )

            try:
                with st.spinner(
                    "Trwa trenowanie i porównywanie modeli..."
                ):
                    benchmark_results = get_model_comparison_results(
                        review_data=filtered_review_data,
                        requested_folds=requested_folds,
                    )

                comparison_table = benchmark_results[
                    "comparison_table"
                ]

                best_model_row = comparison_table.iloc[0]

                benchmark_source_reviews = int(
                    benchmark_results[
                        "source_number_of_reviews"
                    ]
                )

                benchmark_reviews = int(
                    benchmark_results[
                        "number_of_reviews"
                    ]
                )

                benchmark_sample_share = (
                    benchmark_reviews
                    / benchmark_source_reviews
                    if benchmark_source_reviews > 0
                    else 0.0
                )

                formatted_source_reviews = (
                    f"{benchmark_source_reviews:,}"
                    .replace(",", " ")
                )

                formatted_benchmark_reviews = (
                    f"{benchmark_reviews:,}"
                    .replace(",", " ")
                )

                if benchmark_results["sampled"]:
                    st.info(
                        "Benchmark wykonano na reprezentatywnej, "
                        "stratyfikowanej próbce "
                        f"**{formatted_benchmark_reviews}** spośród "
                        f"**{formatted_source_reviews}** poprawnych opinii "
                        f"({benchmark_sample_share:.2%}). "
                        "Próbka zachowuje przybliżone proporcje klas "
                        "sentymentu i została wybrana deterministycznie."
                    )
                else:
                    st.success(
                        "Benchmark wykonano na pełnym zbiorze "
                        f"**{formatted_benchmark_reviews}** opinii."
                    )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Najlepszy model",
                    benchmark_results["best_model_name"],
                )

                col2.metric(
                    "Macro F1",
                    f"{best_model_row['F1MacroMean']:.2%}",
                )

                col3.metric(
                    "Accuracy",
                    f"{best_model_row['AccuracyMean']:.2%}",
                )

                col4.metric(
                    "Liczba części walidacji",
                    benchmark_results["number_of_folds"],
                )

                st.caption(
                    "Strategia walidacji: "
                    f"{benchmark_results['validation_strategy']} | "
                    "Język przetwarzania: "
                    f"{benchmark_results['text_language']} | "
                    "Ziarno losowania: 42"
                )

                sample_col1, sample_col2, sample_col3 = (
                    st.columns(3)
                )

                sample_col1.metric(
                    "Dostępne opinie",
                    formatted_source_reviews,
                )

                sample_col2.metric(
                    "Opinie w benchmarku",
                    formatted_benchmark_reviews,
                )

                sample_col3.metric(
                    "Udział wykorzystanych danych",
                    f"{benchmark_sample_share:.2%}",
                )

                with st.expander(
                    "Metadane próbki wykorzystanej w benchmarku"
                ):
                    st.write(
                        """
                        Próbkowanie przeprowadzono oddzielnie dla każdej
                        klasy sentymentu. Pozwala to zachować strukturę
                        zbioru źródłowego i ograniczyć koszt obliczeniowy
                        walidacji krzyżowej.
                        """
                    )

                    st.dataframe(
                        benchmark_results[
                            "class_distribution"
                        ],
                        width="stretch",
                    )

                st.divider()

                st.subheader("Ranking modeli")

                st.dataframe(
                    comparison_table.style.format(
                        {
                            "AccuracyMean": "{:.2%}",
                            "AccuracyStd": "{:.2%}",
                            "PrecisionMacroMean": "{:.2%}",
                            "RecallMacroMean": "{:.2%}",
                            "F1MacroMean": "{:.2%}",
                            "F1MacroStd": "{:.2%}",
                            "AverageFitTime": "{:.4f}",
                        }
                    ),
                    width="stretch",
                )

                plot_data = comparison_table.melt(
                    id_vars=["Model"],
                    value_vars=[
                        "AccuracyMean",
                        "PrecisionMacroMean",
                        "RecallMacroMean",
                        "F1MacroMean",
                    ],
                    var_name="Metric",
                    value_name="Score",
                )

                metric_names = {
                    "AccuracyMean": "Accuracy",
                    "PrecisionMacroMean": "Precision macro",
                    "RecallMacroMean": "Recall macro",
                    "F1MacroMean": "F1 macro",
                }

                plot_data["Metric"] = plot_data[
                    "Metric"
                ].replace(metric_names)

                comparison_figure = px.bar(
                    plot_data,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Porównanie jakości modeli",
                    range_y=[0, 1],
                )

                st.plotly_chart(
                    comparison_figure,
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Macierz pomyłek najlepszego modelu"
                )

                confusion_data = benchmark_results[
                    "confusion_matrix"
                ]

                confusion_figure = px.imshow(
                    confusion_data,
                    text_auto=True,
                    aspect="auto",
                    title=(
                        "Macierz pomyłek — "
                        f"{benchmark_results['best_model_name']}"
                    ),
                )

                confusion_figure.update_xaxes(
                    side="top"
                )

                st.plotly_chart(
                    confusion_figure,
                    width="stretch",
                )

                st.dataframe(
                    confusion_data,
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Raport klasyfikacji najlepszego modelu"
                )

                st.dataframe(
                    benchmark_results[
                        "classification_report"
                    ].style.format(
                        {
                            "precision": "{:.3f}",
                            "recall": "{:.3f}",
                            "f1-score": "{:.3f}",
                            "support": "{:.0f}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader("Analiza błędów")

                number_of_errors = len(
                    benchmark_results["errors"]
                )

                total_predictions = len(
                    benchmark_results["predictions"]
                )

                error_rate = (
                    number_of_errors / total_predictions
                    if total_predictions > 0
                    else 0
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Liczba predykcji",
                    total_predictions,
                )

                col2.metric(
                    "Liczba błędów",
                    number_of_errors,
                )

                col3.metric(
                    "Odsetek błędów",
                    f"{error_rate:.2%}",
                )

                if benchmark_results["errors"].empty:
                    st.success(
                        "Nie wykryto błędnych predykcji."
                    )
                else:
                    error_columns = [
                        column
                        for column in [
                            "ProductName",
                            "Rating",
                            "RatingSentiment",
                            "PredictedSentiment",
                            "ReviewText",
                        ]
                        if column
                        in benchmark_results[
                            "errors"
                        ].columns
                    ]

                    st.dataframe(
                        benchmark_results[
                            "errors"
                        ][error_columns],
                        width="stretch",
                    )

                st.info(
                    """
                    Analiza błędów pozwala ustalić, jakie typy opinii są
                    najtrudniejsze dla modelu. Szczególnie istotne są recenzje
                    mieszane, niejednoznaczne oraz takie, w których tekst nie
                    odpowiada bezpośrednio ocenie gwiazdkowej.
                    """
                )

            except Exception as error:
                st.error(
                    "Nie udało się przeprowadzić porównania modeli: "
                    f"{error}"
                )

if tab_transformer.open:
    with tab_transformer:
        st.header(
            "Transformerowy model analizy sentymentu"
        )

        st.write(
            """
            Moduł wykorzystuje wielojęzyczny model DistilBERT,
            który obejmuje język portugalski. Model przewiduje
            jedną z pięciu uporządkowanych klas sentymentu:
            od bardzo negatywnej do bardzo pozytywnej.
            Klasy są następnie mapowane na skalę od 1 do 5.
            """
        )

        st.info(
            """
            Model nie jest trenowany na danych Olist. Został
            wcześniej dostrojony na zewnętrznych, syntetycznych
            danych wielojęzycznych. Wyniki na rzeczywistych
            portugalskich opiniach Olist stanowią niezależną
            ocenę jego zdolności generalizacji.
            """
        )

        if (
            filtered_review_data is None
            or filtered_review_data.empty
        ):
            st.warning(
                "Brak danych tekstowych do analizy transformerowej."
            )
        else:
            selected_transformer_batch_size = st.slider(
                "Rozmiar partii przetwarzanych opinii",
                min_value=4,
                max_value=32,
                value=16,
                step=4,
                help=(
                    "Mniejsza wartość zużywa mniej pamięci, "
                    "ale analiza może potrwać dłużej."
                ),
            )

            run_transformer_analysis = st.checkbox(
                "Uruchom model transformerowy",
                value=False,
                help=(
                    "Pierwsze uruchomienie wymaga pobrania "
                    "pliku modelu i może potrwać kilka minut."
                ),
            )

            if not run_transformer_analysis:
                st.info(
                    "Zaznacz pole powyżej, aby rozpocząć analizę."
                )
            else:
                try:
                    with st.spinner(
                        "Trwa ładowanie modelu transformerowego "
                        "i analiza opinii..."
                    ):
                        transformer_results = (
                            get_transformer_evaluation_results(
                                review_data=(
                                    filtered_review_data
                                ),
                                model_name=(
                                    DEFAULT_TRANSFORMER_MODEL
                                ),
                                batch_size=(
                                    selected_transformer_batch_size
                                ),
                            )
                        )

                        classical_results = (
                            get_model_comparison_results(
                                review_data=(
                                    filtered_review_data
                                ),
                                requested_folds=5,
                                max_samples=(
                                    TRANSFORMER_SAMPLE_LIMIT
                                ),
                            )
                        )

                    transformer_metrics = (
                        transformer_results[
                            "metrics"
                        ].iloc[0]
                    )

                    transformer_source_count = int(
                        transformer_results[
                            "source_number_of_reviews"
                        ]
                    )

                    transformer_sample_count = int(
                        transformer_results[
                            "number_of_reviews"
                        ]
                    )

                    transformer_sample_share = (
                        transformer_sample_count
                        / transformer_source_count
                        if transformer_source_count > 0
                        else 0.0
                    )

                    formatted_transformer_source = (
                        f"{transformer_source_count:,}"
                        .replace(",", " ")
                    )

                    formatted_transformer_sample = (
                        f"{transformer_sample_count:,}"
                        .replace(",", " ")
                    )

                    if transformer_results["sampled"]:
                        st.info(
                            "Analizę transformerową wykonano na "
                            "reprezentatywnej, stratyfikowanej próbce "
                            f"**{formatted_transformer_sample}** spośród "
                            f"**{formatted_transformer_source}** opinii "
                            f"({transformer_sample_share:.2%}). "
                            "Do porównania klasycznego wykorzystano "
                            "ten sam limit liczebności oraz ziarno losowania 42."
                        )
                    else:
                        st.success(
                            "Analizę transformerową wykonano na pełnym "
                            f"zbiorze **{formatted_transformer_sample}** opinii."
                        )

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Accuracy sentymentu",
                        (
                            f"{transformer_metrics['SentimentAccuracy']:.2%}"
                        ),
                    )

                    col2.metric(
                        "Macro F1",
                        (
                            f"{transformer_metrics['F1Macro']:.2%}"
                        ),
                    )

                    col3.metric(
                        "MAE liczby gwiazdek",
                        (
                            f"{transformer_metrics['StarMAE']:.3f}"
                        ),
                    )

                    col4.metric(
                        "Urządzenie",
                        transformer_results[
                            "device_name"
                        ],
                    )

                    col5, col6, col7, col8 = st.columns(4)

                    col5.metric(
                        "Precision macro",
                        (
                            f"{transformer_metrics['PrecisionMacro']:.2%}"
                        ),
                    )

                    col6.metric(
                        "Recall macro",
                        (
                            f"{transformer_metrics['RecallMacro']:.2%}"
                        ),
                    )

                    col7.metric(
                        "Dokładna ocena gwiazdkowa",
                        (
                            f"{transformer_metrics['ExactStarAccuracy']:.2%}"
                        ),
                    )

                    col8.metric(
                        "Czas analizy",
                        (
                            f"{transformer_metrics['RuntimeSeconds']:.2f} s"
                        ),
                    )

                    col9, col10 = st.columns(2)

                    col9.metric(
                        "Ocena w granicy ±1 gwiazdki",
                        (
                            f"{transformer_metrics['WithinOneStarAccuracy']:.2%}"
                        ),
                    )

                    col10.metric(
                        "Przetworzone opinie",
                        int(
                            transformer_metrics["ReviewsProcessed"]
                        ),
                    )

                    st.caption(
                        "Model: "
                        f"{transformer_results['model_name']} | "
                        "Język: "
                        f"{transformer_results['model_language']} | "
                        "Wyjście: "
                        f"{transformer_results['model_output']} | "
                        "Licencja: "
                        f"{transformer_results['model_license']}"
                    )

                    st.divider()

                    st.subheader(
                        "Porównanie modelu klasycznego "
                        "i modelu transformerowego"
                    )

                    best_classical_row = (
                        classical_results[
                            "comparison_table"
                        ].iloc[0]
                    )

                    transformer_comparison = pd.DataFrame(
                        [
                            {
                                "Model": (
                                    classical_results[
                                        "best_model_name"
                                    ]
                                ),
                                "ModelType": (
                                    "Klasyczny TF-IDF"
                                ),
                                "Accuracy": (
                                    best_classical_row[
                                        "AccuracyMean"
                                    ]
                                ),
                                "F1Macro": (
                                    best_classical_row[
                                        "F1MacroMean"
                                    ]
                                ),
                                "Evaluation": (
                                    classical_results[
                                        "validation_strategy"
                                    ]
                                ),
                            },
                            {
                                "Model": "Multilingual DistilBERT",
                                "ModelType": (
                                    "Transformer"
                                ),
                                "Accuracy": (
                                    transformer_metrics[
                                        "SentimentAccuracy"
                                    ]
                                ),
                                "F1Macro": (
                                    transformer_metrics[
                                        "F1Macro"
                                    ]
                                ),
                                "Evaluation": (
                                    "Predykcja modelem "
                                    "wytrenowanym zewnętrznie"
                                ),
                            },
                        ]
                    )

                    st.dataframe(
                        transformer_comparison.style.format(
                            {
                                "Accuracy": "{:.2%}",
                                "F1Macro": "{:.2%}",
                            }
                        ),
                        width="stretch",
                    )

                    classical_accuracy = float(
                        best_classical_row[
                            "AccuracyMean"
                        ]
                    )

                    classical_f1_macro = float(
                        best_classical_row[
                            "F1MacroMean"
                        ]
                    )

                    transformer_accuracy = float(
                        transformer_metrics[
                            "SentimentAccuracy"
                        ]
                    )

                    transformer_f1_macro = float(
                        transformer_metrics[
                            "F1Macro"
                        ]
                    )

                    accuracy_difference_pp = (
                        classical_accuracy
                        - transformer_accuracy
                    ) * 100

                    f1_difference_pp = (
                        classical_f1_macro
                        - transformer_f1_macro
                    ) * 100

                    if (
                        accuracy_difference_pp > 0
                        and f1_difference_pp > 0
                    ):
                        st.warning(
                            "W zastosowanych procedurach oceny model klasyczny "
                            f"uzyskał Accuracy wyższe o "
                            f"**{accuracy_difference_pp:.2f} p.p.** oraz "
                            f"Macro F1 wyższe o "
                            f"**{f1_difference_pp:.2f} p.p.** "
                            "Wynik wskazuje, że model uczony na danych Olist "
                            "lepiej dopasował się do słownictwa i charakteru "
                            "analizowanych opinii niż model transformerowy "
                            "dostrojony wcześniej na danych zewnętrznych."
                        )

                    elif (
                        accuracy_difference_pp < 0
                        and f1_difference_pp < 0
                    ):
                        st.success(
                            "W zastosowanych procedurach oceny model "
                            "transformerowy uzyskał wyższe wyniki zarówno "
                            "dla Accuracy, jak i Macro F1."
                        )

                    else:
                        st.info(
                            "Modele uzyskały niejednoznaczne wyniki: "
                            "jeden z nich osiągnął wyższe Accuracy, "
                            "natomiast drugi wyższe Macro F1."
                        )

                    comparison_plot_data = (
                        transformer_comparison.melt(
                            id_vars=[
                                "Model",
                                "ModelType",
                            ],
                            value_vars=[
                                "Accuracy",
                                "F1Macro",
                            ],
                            var_name="Metric",
                            value_name="Score",
                        )
                    )

                    transformer_comparison_figure = px.bar(
                        comparison_plot_data,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        range_y=[0, 1],
                        title=(
                            "Porównanie jakości modeli"
                        ),
                    )

                    st.plotly_chart(
                        transformer_comparison_figure,
                        width="stretch",
                    )

                    st.caption(
                        """
                        Procedury oceny modeli różnią się. Model klasyczny
                        jest oceniany przez walidację krzyżową na bieżącym
                        zbiorze, natomiast model transformerowy został
                        wcześniej dostrojony na zewnętrznych danych
                        i wykonuje bezpośrednią predykcję.
                        """
                    )

                    st.divider()

                    st.subheader(
                        "Macierz pomyłek modelu transformerowego"
                    )

                    transformer_confusion = (
                        transformer_results[
                            "confusion_matrix"
                        ]
                    )

                    transformer_confusion_figure = px.imshow(
                        transformer_confusion,
                        text_auto=True,
                        aspect="auto",
                        title=(
                            "Rzeczywisty i przewidywany sentyment"
                        ),
                    )

                    transformer_confusion_figure.update_xaxes(
                        side="top"
                    )

                    st.plotly_chart(
                        transformer_confusion_figure,
                        width="stretch",
                    )

                    st.dataframe(
                        transformer_confusion,
                        width="stretch",
                    )

                    st.subheader(
                        "Skuteczność rozpoznawania poszczególnych klas"
                    )

                    class_diagnostic_rows = []

                    for sentiment_label in transformer_results[
                        "labels"
                    ]:
                        actual_row_name = (
                            f"Rzeczywiste: {sentiment_label}"
                        )

                        predicted_column_name = (
                            f"Predykcja: {sentiment_label}"
                        )

                        actual_class_count = int(
                            transformer_confusion
                            .loc[
                                actual_row_name
                            ]
                            .sum()
                        )

                        correct_class_predictions = int(
                            transformer_confusion.loc[
                                actual_row_name,
                                predicted_column_name,
                            ]
                        )

                        class_recall = (
                            correct_class_predictions
                            / actual_class_count
                            if actual_class_count > 0
                            else 0.0
                        )

                        class_diagnostic_rows.append(
                            {
                                "Klasa": sentiment_label,
                                "Liczba rzeczywistych opinii": (
                                    actual_class_count
                                ),
                                "Poprawne predykcje": (
                                    correct_class_predictions
                                ),
                                "Recall": class_recall,
                            }
                        )

                    class_diagnostics = pd.DataFrame(
                        class_diagnostic_rows
                    )

                    st.dataframe(
                        class_diagnostics.style.format(
                            {
                                "Recall": "{:.2%}",
                            }
                        ),
                        width="stretch",
                    )

                    weakest_class = (
                        class_diagnostics
                        .sort_values(
                            by="Recall",
                            ascending=True,
                        )
                        .iloc[0]
                    )

                    st.warning(
                        "Najtrudniejszą klasą dla modelu transformerowego "
                        f"jest **{weakest_class['Klasa']}**. "
                        "Model poprawnie rozpoznał "
                        f"**{int(weakest_class['Poprawne predykcje'])}** "
                        "spośród "
                        f"**{int(weakest_class['Liczba rzeczywistych opinii'])}** "
                        "opinii tej klasy, co odpowiada wartości recall "
                        f"**{weakest_class['Recall']:.2%}**."
                    )

                    st.divider()

                    st.subheader(
                        "Raport klasyfikacji"
                    )

                    st.dataframe(
                        transformer_results[
                            "classification_report"
                        ].style.format(
                            {
                                "precision": "{:.3f}",
                                "recall": "{:.3f}",
                                "f1-score": "{:.3f}",
                                "support": "{:.0f}",
                            }
                        ),
                        width="stretch",
                    )

                    st.divider()

                    st.subheader(
                        "Rozkład predykcji transformerowych"
                    )

                    transformer_distribution_figure = px.bar(
                        transformer_results[
                            "predicted_distribution"
                        ],
                        x="TransformerSentiment",
                        y="Reviews",
                        title=(
                            "Liczba opinii według "
                            "przewidzianego sentymentu"
                        ),
                        text="Reviews",
                    )

                    st.plotly_chart(
                        transformer_distribution_figure,
                        width="stretch",
                    )

                    st.divider()

                    st.subheader(
                        "Analiza błędnych predykcji"
                    )

                    transformer_errors = (
                        transformer_results["errors"]
                    )

                    error_count = len(
                        transformer_errors
                    )

                    error_share = (
                        error_count
                        / transformer_results[
                            "number_of_reviews"
                        ]
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Przetworzone opinie",
                        transformer_results[
                            "number_of_reviews"
                        ],
                    )

                    col2.metric(
                        "Błędne sentymenty",
                        error_count,
                    )

                    col3.metric(
                        "Odsetek błędów",
                        f"{error_share:.2%}",
                    )

                    transformer_error_columns = [
                        column
                        for column in [
                            "ProductName",
                            "Rating",
                            "RatingSentiment",
                            "TransformerStars",
                            "TransformerSentiment",
                            "TransformerConfidence",
                            "ReviewText",
                        ]
                        if column
                        in transformer_errors.columns
                    ]

                    st.dataframe(
                        transformer_errors[
                            transformer_error_columns
                        ].style.format(
                            {
                                "TransformerConfidence": (
                                    "{:.2%}"
                                ),
                            }
                        ),
                        width="stretch",
                    )

                    st.divider()

                    st.subheader(
                        "Wszystkie predykcje modelu"
                    )

                    transformer_prediction_columns = [
                        column
                        for column in [
                            "ProductName",
                            "Rating",
                            "RatingSentiment",
                            "TransformerStars",
                            "TransformerSentiment",
                            "TransformerConfidence",
                            "NegativeProbability",
                            "NeutralProbability",
                            "PositiveProbability",
                            "SentimentAgreement",
                            "ReviewText",
                        ]
                        if column
                        in transformer_results[
                            "predictions"
                        ].columns
                    ]

                    st.dataframe(
                        transformer_results[
                            "predictions"
                        ][
                            transformer_prediction_columns
                        ].style.format(
                            {
                                "TransformerConfidence": (
                                    "{:.2%}"
                                ),
                                "NegativeProbability": (
                                    "{:.2%}"
                                ),
                                "NeutralProbability": (
                                    "{:.2%}"
                                ),
                                "PositiveProbability": (
                                    "{:.2%}"
                                ),
                            }
                        ),
                        width="stretch",
                    )

                except Exception as error:
                    st.error(
                        "Nie udało się przeprowadzić "
                        f"analizy transformerowej: {error}"
                    )

if tab_interpretability.open:
    with tab_interpretability:
        st.header("Interpretowalność modelu klasyfikacji sentymentu")

        st.write(
            """
            Sekcja przedstawia globalne i lokalne wyjaśnienia działania
            modelu TF-IDF + Logistic Regression. Model interpretowalny
            nie zastępuje najlepszego modelu wybranego w benchmarku.
            Pozwala jednak ustalić, jakie słowa i frazy wpływają
            na klasyfikację sentymentu.
            """
        )

        if filtered_review_data is None or filtered_review_data.empty:
            st.warning(
                "Brak danych tekstowych do interpretacji modelu."
            )
        else:
            try:
                with st.spinner(
                    "Trwa przygotowywanie wyjaśnień modelu..."
                ):
                    interpretability_results = (
                        get_interpretability_results(
                            review_data=filtered_review_data,
                            top_n=20,
                        )
                    )

                    benchmark_for_interpretability = (
                        get_model_comparison_results(
                            review_data=filtered_review_data,
                            requested_folds=5,
                        )
                    )

                if interpretability_results["sampled"]:
                    interpretability_source_count = int(
                        interpretability_results[
                            "source_number_of_reviews"
                        ]
                    )

                    interpretability_sample_count = int(
                        interpretability_results[
                            "number_of_reviews"
                        ]
                    )

                    st.info(
                        "Model interpretowalny wytrenowano na "
                        f"reprezentatywnej próbce "
                        f"**{interpretability_sample_count:,}** "
                        f"spośród "
                        f"**{interpretability_source_count:,}** "
                        "poprawnych opinii."
                        .replace(",", " ")
                    )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Najlepszy model predykcyjny",
                    benchmark_for_interpretability[
                        "best_model_name"
                    ],
                )

                col2.metric(
                    "Model interpretowalny",
                    "Logistic Regression",
                )

                col3.metric(
                    "Liczba opinii",
                    interpretability_results[
                        "number_of_reviews"
                    ],
                )

                col4.metric(
                    "Liczba cech TF-IDF",
                    interpretability_results[
                        "number_of_features"
                    ],
                )

                st.info(
                    """
                    Dodatni współczynnik oznacza, że wystąpienie terminu
                    zwiększa wynik danej klasy. Im większa wartość
                    współczynnika, tym silniejsze globalne powiązanie
                    terminu z daną klasą sentymentu.
                    """
                )

                st.divider()

                st.subheader("Globalna interpretacja modelu")

                selected_interpretability_class = st.selectbox(
                    "Wybierz klasę sentymentu",
                    options=interpretability_results[
                        "classes"
                    ],
                    key="interpretability_class",
                )

                selected_supporting_terms = (
                    interpretability_results[
                        "supporting_terms"
                    ][
                        interpretability_results[
                            "supporting_terms"
                        ]["Sentiment"]
                        == selected_interpretability_class
                    ]
                    .sort_values(
                        by="Coefficient",
                        ascending=True,
                    )
                )

                supporting_figure = px.bar(
                    selected_supporting_terms,
                    x="Coefficient",
                    y="Term",
                    orientation="h",
                    title=(
                        "Terminy najsilniej wspierające klasę: "
                        f"{selected_interpretability_class}"
                    ),
                )

                st.plotly_chart(
                    supporting_figure,
                    width="stretch",
                )

                st.dataframe(
                    selected_supporting_terms.sort_values(
                        by="Rank"
                    ).style.format(
                        {
                            "Coefficient": "{:.4f}",
                        }
                    ),
                    width="stretch",
                )

                with st.expander(
                    "Pokaż terminy działające przeciw tej klasie"
                ):
                    selected_opposing_terms = (
                        interpretability_results[
                            "opposing_terms"
                        ][
                            interpretability_results[
                                "opposing_terms"
                            ]["Sentiment"]
                            == selected_interpretability_class
                        ]
                        .sort_values(
                            by="Coefficient",
                            ascending=False,
                        )
                    )

                    opposing_figure = px.bar(
                        selected_opposing_terms,
                        x="Coefficient",
                        y="Term",
                        orientation="h",
                        title=(
                            "Terminy zmniejszające wynik klasy: "
                            f"{selected_interpretability_class}"
                        ),
                    )

                    st.plotly_chart(
                        opposing_figure,
                        width="stretch",
                    )

                    st.dataframe(
                        selected_opposing_terms.sort_values(
                            by="Rank"
                        ).style.format(
                            {
                                "Coefficient": "{:.4f}",
                            }
                        ),
                        width="stretch",
                    )

                st.divider()

                st.subheader(
                    "Lokalne wyjaśnienie pojedynczej opinii"
                )

                review_options = (
                    filtered_review_data
                    .reset_index(drop=True)
                )

                selected_review_position = st.selectbox(
                    "Wybierz opinię do wyjaśnienia",
                    options=range(len(review_options)),
                    format_func=lambda position: (
                        f"{review_options.iloc[position].get('ProductName', 'Produkt')}"
                        " — "
                        f"{str(review_options.iloc[position].get('ReviewText', ''))[:90]}"
                    ),
                    key="interpretability_review",
                )

                selected_review = review_options.iloc[
                    selected_review_position
                ]

                local_explanation = explain_single_review(
                    model=interpretability_results["model"],
                    review_text=selected_review[
                        "CleanReviewText"
                    ],
                    top_n=15,
                )

                actual_sentiment = selected_review[
                    "RatingSentiment"
                ]

                predicted_sentiment = local_explanation[
                    "predicted_sentiment"
                ]

                prediction_agreement = (
                    actual_sentiment
                    == predicted_sentiment
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Sentyment rzeczywisty",
                    actual_sentiment,
                )

                col2.metric(
                    "Predykcja modelu",
                    predicted_sentiment,
                )

                col3.metric(
                    "Pewność modelu",
                    f"{local_explanation['confidence']:.2%}",
                )

                col4.metric(
                    "Zgodność",
                    "Tak" if prediction_agreement else "Nie",
                )

                st.write("**Oryginalna treść opinii:**")

                st.write(
                    selected_review.get(
                        "ReviewText",
                        selected_review["CleanReviewText"],
                    )
                )

                st.subheader(
                    "Prawdopodobieństwo poszczególnych klas"
                )

                probability_figure = px.bar(
                    local_explanation["probabilities"],
                    x="Sentiment",
                    y="Probability",
                    title="Rozkład prawdopodobieństw modelu",
                    range_y=[0, 1],
                )

                st.plotly_chart(
                    probability_figure,
                    width="stretch",
                )

                st.subheader(
                    "Wpływ terminów na predykcję"
                )

                if local_explanation[
                    "contributions"
                ].empty:
                    st.warning(
                        "W opinii nie znaleziono terminów "
                        "obecnych w słowniku modelu."
                    )
                else:
                    contribution_data = (
                        local_explanation[
                            "contributions"
                        ]
                        .sort_values(
                            by="Contribution",
                            ascending=True,
                        )
                    )

                    contribution_figure = px.bar(
                        contribution_data,
                        x="Contribution",
                        y="Term",
                        orientation="h",
                        title=(
                            "Lokalny wpływ terminów na klasę: "
                            f"{predicted_sentiment}"
                        ),
                    )

                    st.plotly_chart(
                        contribution_figure,
                        width="stretch",
                    )

                    st.dataframe(
                        local_explanation[
                            "contributions"
                        ].style.format(
                            {
                                "TFIDF": "{:.4f}",
                                "Coefficient": "{:.4f}",
                                "Contribution": "{:.4f}",
                                "AbsoluteContribution": "{:.4f}",
                            }
                        ),
                        width="stretch",
                    )

                st.caption(
                    """
                    Lokalne wyjaśnienie dotyczy wyłącznie wybranej opinii.
                    Wpływ terminu jest obliczany jako iloczyn wartości
                    TF-IDF i współczynnika przypisanego do przewidzianej klasy.
                    """
                )

            except Exception as error:
                st.error(
                    "Nie udało się przygotować interpretacji modelu: "
                    f"{error}"
                )

if tab_topics.open:
    with tab_topics:
        st.header("Automatyczne wykrywanie tematów w opiniach")

        st.write(
            """
            Model NMF analizuje reprezentację TF-IDF opinii i identyfikuje
            ukryte grupy współwystępujących słów oraz fraz. Każda opinia
            zostaje przypisana do tematu, który ma w niej największy udział.
            """
        )

        st.info(
            """
            Nazwy tematów są generowane automatycznie na podstawie trzech
            terminów o największych wagach. Powinny być traktowane jako
            pomoc analityczna, a nie jako ostateczne, obiektywne etykiety.
            """
        )

        if filtered_review_data is None or filtered_review_data.empty:
            st.warning(
                "Brak danych tekstowych do modelowania tematów."
            )
        else:
            col1, col2 = st.columns(2)

            with col1:
                selected_number_of_topics = st.slider(
                    "Liczba wykrywanych tematów",
                    min_value=3,
                    max_value=8,
                    value=5,
                    step=1,
                    help=(
                        "Większa liczba tematów daje bardziej szczegółowy "
                        "podział, ale może prowadzić do podobnych lub "
                        "trudnych do interpretacji tematów."
                    ),
                )

            with col2:
                selected_top_terms = st.slider(
                    "Liczba terminów opisujących temat",
                    min_value=5,
                    max_value=15,
                    value=10,
                    step=1,
                )

            try:
                with st.spinner(
                    "Trwa wykrywanie tematów w opiniach..."
                ):
                    topic_results = get_topic_analysis_results(
                        review_data=filtered_review_data,
                        number_of_topics=(
                            selected_number_of_topics
                        ),
                        top_terms_per_topic=(
                            selected_top_terms
                        ),
                    )

                if topic_results["sampled"]:
                    topic_source_count = int(
                        topic_results[
                            "source_number_of_reviews"
                        ]
                    )

                    topic_fit_count = int(
                        topic_results[
                            "fit_number_of_reviews"
                        ]
                    )

                    topic_assignment_count = int(
                        topic_results[
                            "number_of_reviews"
                        ]
                    )

                    topic_sample_share = (
                        topic_fit_count
                        / topic_source_count
                        if topic_source_count > 0
                        else 0.0
                    )

                    formatted_topic_source_count = (
                        f"{topic_source_count:,}"
                        .replace(",", " ")
                    )

                    formatted_topic_fit_count = (
                        f"{topic_fit_count:,}"
                        .replace(",", " ")
                    )

                    formatted_topic_assignment_count = (
                        f"{topic_assignment_count:,}"
                        .replace(",", " ")
                    )

                    st.info(
                        "Model TF-IDF + NMF dopasowano na "
                        "reprezentatywnej, stratyfikowanej próbce "
                        f"**{formatted_topic_fit_count}** spośród "
                        f"**{formatted_topic_source_count}** opinii "
                        f"({topic_sample_share:.2%}). "
                        "Następnie wytrenowany model wykorzystano "
                        "do przypisania tematów wszystkim "
                        f"**{formatted_topic_assignment_count}** "
                        "poprawnym opiniom."
                    )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Opinie z przypisanym tematem",
                    topic_results["number_of_reviews"],
                )

                col2.metric(
                    "Liczba tematów",
                    topic_results["number_of_topics"],
                )

                col3.metric(
                    "Liczba cech TF-IDF",
                    topic_results["number_of_features"],
                )

                col4.metric(
                    "Śr. dominacja tematu",
                    (
                        f"{topic_results['average_topic_dominance']:.2%}"
                    ),
                )

                col5, col6 = st.columns(2)

                col5.metric(
                    "Błąd rekonstrukcji NMF",
                    (
                        f"{topic_results['reconstruction_error']:.4f}"
                    ),
                )

                col6.metric(
                    "Różnorodność tematów",
                    (
                        f"{topic_results['topic_diversity']:.2%}"
                    ),
                )

                st.caption(
                    """
                    Dominacja tematu informuje, jak dużą część wszystkich wag
                    tematycznych opinii stanowi temat dominujący. Różnorodność
                    określa udział unikalnych terminów w zestawie najważniejszych
                    terminów wszystkich tematów.
                    """
                )

                st.divider()

                st.subheader("Przegląd wykrytych tematów")

                topic_overview_display = (
                    topic_results["topic_overview"]
                    .merge(
                        topic_results[
                            "topic_distribution"
                        ][
                            [
                                "TopicIndex",
                                "Reviews",
                                "ReviewShare",
                                "AverageDominance",
                            ]
                        ],
                        on="TopicIndex",
                        how="left",
                    )
                )

                st.dataframe(
                    topic_overview_display.style.format(
                        {
                            "ReviewShare": "{:.2%}",
                            "AverageDominance": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                topic_distribution_figure = px.bar(
                    topic_results["topic_distribution"],
                    x="TopicLabel",
                    y="Reviews",
                    title="Liczba opinii przypisana do tematów",
                    text="Reviews",
                )

                topic_distribution_figure.update_xaxes(
                    tickangle=-25
                )

                st.plotly_chart(
                    topic_distribution_figure,
                    width="stretch",
                )

                st.divider()

                st.subheader("Szczegóły wybranego tematu")

                selected_topic_label = st.selectbox(
                    "Wybierz temat",
                    options=topic_results[
                        "topic_overview"
                    ]["TopicLabel"].tolist(),
                    key="selected_topic_label",
                )

                selected_topic_terms = (
                    topic_results["topic_terms"][
                        topic_results["topic_terms"][
                            "TopicLabel"
                        ]
                        == selected_topic_label
                    ]
                    .sort_values(
                        by="Weight",
                        ascending=True,
                    )
                )

                topic_terms_figure = px.bar(
                    selected_topic_terms,
                    x="Weight",
                    y="Term",
                    orientation="h",
                    title=(
                        "Najważniejsze terminy — "
                        f"{selected_topic_label}"
                    ),
                )

                st.plotly_chart(
                    topic_terms_figure,
                    width="stretch",
                )

                st.dataframe(
                    selected_topic_terms.sort_values(
                        by="Rank"
                    ).style.format(
                        {
                            "Weight": "{:.4f}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Sentyment opinii w wybranym temacie"
                )

                selected_topic_sentiment = (
                    topic_results[
                        "topic_sentiment_summary"
                    ][
                        topic_results[
                            "topic_sentiment_summary"
                        ]["TopicLabel"]
                        == selected_topic_label
                    ]
                )

                sentiment_topic_figure = px.bar(
                    selected_topic_sentiment,
                    x="RatingSentiment",
                    y="Reviews",
                    color="RatingSentiment",
                    title=(
                        "Rozkład sentymentu — "
                        f"{selected_topic_label}"
                    ),
                    text="Reviews",
                )

                st.plotly_chart(
                    sentiment_topic_figure,
                    width="stretch",
                )

                st.dataframe(
                    selected_topic_sentiment.style.format(
                        {
                            "SentimentShare": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Przykładowe opinie przypisane do tematu"
                )

                selected_topic_reviews = (
                    topic_results[
                        "review_assignments"
                    ][
                        topic_results[
                            "review_assignments"
                        ]["TopicLabel"]
                        == selected_topic_label
                    ]
                    .sort_values(
                        by="DominantTopicShare",
                        ascending=False,
                    )
                )

                review_display_columns = [
                    column
                    for column in [
                        "ProductName",
                        "Rating",
                        "RatingSentiment",
                        "DominantTopicShare",
                        "ReviewText",
                    ]
                    if column
                    in selected_topic_reviews.columns
                ]

                st.dataframe(
                    selected_topic_reviews[
                        review_display_columns
                    ]
                    .head(30)
                    .style.format(
                        {
                            "DominantTopicShare": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader("Tematy według produktów")

                if topic_results[
                    "product_topic_summary"
                ].empty:
                    st.warning(
                        "Brak nazw produktów potrzebnych "
                        "do analizy produktowej."
                    )
                else:
                    product_topic_pivot = (
                        topic_results[
                            "product_topic_summary"
                        ]
                        .pivot_table(
                            index="ProductName",
                            columns="TopicLabel",
                            values="Reviews",
                            aggfunc="sum",
                            fill_value=0,
                        )
                    )

                    product_topic_figure = px.imshow(
                        product_topic_pivot,
                        text_auto=True,
                        aspect="auto",
                        title=(
                            "Liczba opinii według produktu "
                            "i dominującego tematu"
                        ),
                    )

                    st.plotly_chart(
                        product_topic_figure,
                        width="stretch",
                    )

                    st.dataframe(
                        product_topic_pivot,
                        width="stretch",
                    )

                if selected_data_source == "Dane syntetyczne":
                    st.warning(
                        """
                        Dane syntetyczne korzystają z ograniczonej liczby
                        szablonów opinii. Wykryte tematy służą przede wszystkim
                        do przetestowania działania modułu.
                        """
                    )
                else:
                    st.info(
                        """
                        Tematy zostały wykryte na podstawie aktualnie wybranego
                        zbioru opinii. Automatycznie wygenerowane nazwy tematów
                        wymagają interpretacji analitycznej i nie powinny być
                        traktowane jako jednoznaczne etykiety biznesowe.
                        """
                    )

            except Exception as error:
                st.error(
                    "Nie udało się przeprowadzić modelowania tematów: "
                    f"{error}"
                )

if tab_decision_center.open:
    with tab_decision_center:
        st.header("Zaawansowane centrum wspomagania decyzji")

        st.write(
            """
            Moduł integruje wyniki analizy sprzedaży, trendów produktowych,
            sentymentu klientów, modelowania tematów oraz prognozowania.
            Produkty są klasyfikowane według ich znaczenia sprzedażowego
            i bilansu opinii klientów.
            """
        )

        if (
            filtered_sales_data is None
            or filtered_sales_data.empty
            or filtered_review_data is None
            or filtered_review_data.empty
        ):
            st.warning(
                "Centrum decyzji wymaga jednocześnie danych "
                "sprzedażowych i opinii klientów."
            )
        else:
            try:
                with st.spinner(
                    "Trwa integrowanie wyników analiz..."
                ):
                    decision_topic_results = (
                        get_topic_analysis_results(
                            review_data=filtered_review_data,
                            number_of_topics=5,
                            top_terms_per_topic=10,
                        )
                    )

                    decision_forecast_results = (
                        get_sales_forecast_results(
                            sales_data=filtered_sales_data,
                            test_fraction=0.25,
                            forecast_horizon=8,
                        )
                    )

                    decision_results = (
                        get_advanced_decision_results(
                            sales_data=filtered_sales_data,
                            review_data=filtered_review_data,
                            topic_assignments=(
                                decision_topic_results[
                                    "review_assignments"
                                ]
                            ),
                            weekly_data=(
                                decision_forecast_results[
                                    "weekly_data"
                                ]
                            ),
                            future_forecast=(
                                decision_forecast_results[
                                    "future_forecast"
                                ]
                            ),
                        )
                    )

                executive_summary = decision_results[
                    "executive_summary"
                ]

                product_matrix = decision_results[
                    "product_matrix"
                ]

                advanced_recommendations = (
                    decision_results[
                        "recommendations"
                    ]
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Analizowane produkty",
                    executive_summary[
                        "ProductsAnalyzed"
                    ],
                )

                col2.metric(
                    "Wysoki priorytet",
                    executive_summary[
                        "HighPriorityRecommendations"
                    ],
                )

                col3.metric(
                    "Perspektywa sprzedaży",
                    executive_summary[
                        "ForecastDirection"
                    ],
                )

                forecast_change = executive_summary[
                    "ForecastChangePct"
                ]

                col4.metric(
                    "Zmiana prognozowanego poziomu",
                    (
                        f"{forecast_change:.2f}%"
                        if not pd.isna(
                            forecast_change
                        )
                        else "Brak danych"
                    ),
                )

                col5, col6 = st.columns(2)

                col5.metric(
                    "Najwyższe ryzyko",
                    executive_summary[
                        "TopRiskProduct"
                    ],
                )

                col6.metric(
                    "Największy potencjał",
                    executive_summary[
                        "TopOpportunityProduct"
                    ],
                )

                st.divider()

                st.subheader(
                    "Macierz sprzedaż–sentyment"
                )

                matrix_plot_data = product_matrix.copy()

                matrix_plot_data["ReviewsForSize"] = (
                    matrix_plot_data["Reviews"]
                    .clip(lower=1)
                )

                decision_matrix_figure = px.scatter(
                    matrix_plot_data,
                    x="SalesImportance",
                    y="SentimentBalance",
                    size="ReviewsForSize",
                    color="DecisionQuadrant",
                    hover_name="ProductName",
                    hover_data={
                        "Revenue": ":,.2f",
                        "RevenueTrendPct": ":.2f",
                        "AverageRating": ":.2f",
                        "PositiveShare": ":.2%",
                        "NegativeShare": ":.2%",
                        "ReviewsForSize": False,
                    },
                    title=(
                        "Pozycja produktów według znaczenia "
                        "sprzedażowego i bilansu opinii"
                    ),
                )

                decision_matrix_figure.add_vline(
                    x=0.50,
                    line_dash="dash",
                )

                decision_matrix_figure.add_hline(
                    y=0.20,
                    line_dash="dash",
                )

                st.plotly_chart(
                    decision_matrix_figure,
                    width="stretch",
                )

                st.caption(
                    """
                    Prawa część wykresu obejmuje produkty o wyższym znaczeniu
                    sprzedażowym. Górna część oznacza korzystniejszy bilans
                    opinii pozytywnych względem negatywnych.
                    """
                )

                st.divider()

                st.subheader(
                    "Ranking ryzyka i potencjału"
                )

                col1, col2 = st.columns(2)

                with col1:
                    risk_data = (
                        product_matrix
                        .nlargest(
                            10,
                            "RiskScore",
                        )
                        .sort_values(
                            by="RiskScore",
                            ascending=True,
                        )
                    )

                    risk_figure = px.bar(
                        risk_data,
                        x="RiskScore",
                        y="ProductName",
                        orientation="h",
                        title=(
                            "Produkty o najwyższym "
                            "wskaźniku ryzyka"
                        ),
                        range_x=[0, 100],
                    )

                    st.plotly_chart(
                        risk_figure,
                        width="stretch",
                    )

                with col2:
                    opportunity_data = (
                        product_matrix
                        .nlargest(
                            10,
                            "OpportunityScore",
                        )
                        .sort_values(
                            by="OpportunityScore",
                            ascending=True,
                        )
                    )

                    opportunity_figure = px.bar(
                        opportunity_data,
                        x="OpportunityScore",
                        y="ProductName",
                        orientation="h",
                        title=(
                            "Produkty o najwyższym "
                            "potencjale rozwojowym"
                        ),
                        range_x=[0, 100],
                    )

                    st.plotly_chart(
                        opportunity_figure,
                        width="stretch",
                    )

                st.divider()

                st.subheader(
                    "Macierz wskaźników produktowych"
                )

                matrix_columns = [
                    "ProductName",
                    "DecisionQuadrant",
                    "Revenue",
                    "RevenueShare",
                    "RevenueTrendPct",
                    "Reviews",
                    "AverageRating",
                    "PositiveShare",
                    "NegativeShare",
                    "DominantTopic",
                    "RiskScore",
                    "OpportunityScore",
                ]

                product_matrix_display = (
                    product_matrix[
                        matrix_columns
                    ]
                    .sort_values(
                        by="Revenue",
                        ascending=False,
                    )
                    .reset_index(drop=True)
                )

                selected_matrix_rows = st.selectbox(
                    "Liczba produktów wyświetlanych w tabeli",
                    options=[
                        100,
                        500,
                        1000,
                        5000,
                    ],
                    index=1,
                    help=(
                        "Tabela jest ograniczona wyłącznie w interfejsie. "
                        "Wszystkie produkty nadal są wykorzystywane "
                        "w obliczeniach centrum decyzji."
                    ),
                    key="decision_matrix_display_rows",
                )

                displayed_product_matrix = (
                    product_matrix_display
                    .head(selected_matrix_rows)
                )

                displayed_matrix_count = len(
                    displayed_product_matrix
                )

                total_matrix_count = len(
                    product_matrix_display
                )

                st.caption(
                    "Wyświetlono "
                    f"{displayed_matrix_count:,} z "
                    f"{total_matrix_count:,} produktów, "
                    "uporządkowanych malejąco według przychodu."
                    .replace(",", " ")
                )

                st.dataframe(
                    displayed_product_matrix.style.format(
                        {
                            "Revenue": "{:,.2f}",
                            "RevenueShare": "{:.2%}",
                            "RevenueTrendPct": "{:.2f}%",
                            "AverageRating": "{:.2f}",
                            "PositiveShare": "{:.2%}",
                            "NegativeShare": "{:.2%}",
                            "RiskScore": "{:.2f}",
                            "OpportunityScore": "{:.2f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.divider()

                st.subheader(
                    "Zaawansowane rekomendacje"
                )

                col1, col2 = st.columns(2)

                with col1:
                    selected_advanced_priority = (
                        st.selectbox(
                            "Priorytet",
                            options=[
                                "Wszystkie",
                                "Wysoki",
                                "Średni",
                                "Niski",
                            ],
                            key=(
                                "advanced_recommendation_priority"
                            ),
                        )
                    )

                with col2:
                    available_categories = sorted(
                        advanced_recommendations[
                            "Category"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    selected_advanced_category = (
                        st.selectbox(
                            "Kategoria",
                            options=[
                                "Wszystkie",
                                *available_categories,
                            ],
                            key=(
                                "advanced_recommendation_category"
                            ),
                        )
                    )

                displayed_recommendations = (
                    advanced_recommendations.copy()
                )

                if (
                    selected_advanced_priority
                    != "Wszystkie"
                ):
                    displayed_recommendations = (
                        displayed_recommendations[
                            displayed_recommendations[
                                "Priority"
                            ]
                            == selected_advanced_priority
                        ]
                    )

                if (
                    selected_advanced_category
                    != "Wszystkie"
                ):
                    displayed_recommendations = (
                        displayed_recommendations[
                            displayed_recommendations[
                                "Category"
                            ]
                            == selected_advanced_category
                        ]
                    )

                for _, recommendation_row in (
                    displayed_recommendations.iterrows()
                ):
                    with st.container(border=True):
                        st.subheader(
                            f"{recommendation_row['ProductName']} "
                            f"— {recommendation_row['Category']}"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.write(
                            "**Priorytet:** "
                            f"{recommendation_row['Priority']}"
                        )

                        col2.write(
                            "**Wskaźnik:** "
                            f"{recommendation_row['PriorityScore']:.2f}"
                        )

                        col3.write(
                            "**Macierz:** "
                            f"{recommendation_row['DecisionQuadrant']}"
                        )

                        st.write(
                            "**Wniosek:** "
                            f"{recommendation_row['Conclusion']}"
                        )

                        st.write(
                            "**Rekomendowane działanie:** "
                            f"{recommendation_row['Recommendation']}"
                        )

                st.dataframe(
                    advanced_recommendations.style.format(
                        {
                            "PriorityScore": "{:.2f}",
                            "Revenue": "{:,.2f}",
                            "RevenueTrendPct": "{:.2f}%",
                            "AverageRating": "{:.2f}",
                            "PositiveShare": "{:.2%}",
                            "NegativeShare": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.warning(
                    """
                    Wskaźniki ryzyka i potencjału są syntetycznymi miarami
                    wspierającymi analizę. Nie zastępują oceny menedżera
                    i powinny być interpretowane wraz z kontekstem biznesowym.
                    """
                )

            except Exception as error:
                st.error(
                    "Nie udało się przygotować centrum decyzji: "
                    f"{error}"
                )

if tab_monitoring.open:
    with tab_monitoring:
        st.header(
            "Monitoring stabilności danych i wykrywanie driftu"
        )

        st.write(
            """
            Moduł porównuje dwa kolejne okresy o tej samej długości.
            Pozwala wykryć zmiany struktury sprzedaży, zachowań klientów,
            ocen, sentymentu oraz słownictwa używanego w opiniach.
            """
        )

        st.info(
            """
            Wynik driftu mieści się w przedziale od 0 do 1.
            Wartość poniżej 0,10 oznacza niski poziom zmian,
            od 0,10 do 0,25 poziom średni, a od 0,25 poziom wysoki.
            """
        )

        if (
            filtered_sales_data is None
            or filtered_sales_data.empty
            or filtered_review_data is None
            or filtered_review_data.empty
        ):
            st.warning(
                "Monitoring wymaga jednocześnie danych "
                "sprzedażowych i opinii klientów."
            )
        else:
            selected_drift_window = st.slider(
                "Długość każdego porównywanego okresu w tygodniach",
                min_value=4,
                max_value=16,
                value=8,
                step=2,
            )

            try:
                with st.spinner(
                    "Trwa porównywanie okresów i obliczanie driftu..."
                ):
                    drift_results = (
                        get_drift_monitoring_results(
                            sales_data=filtered_sales_data,
                            review_data=filtered_review_data,
                            window_weeks=selected_drift_window,
                        )
                    )

                drift_summary = drift_results[
                    "summary"
                ]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Ogólny wynik driftu",
                    (
                        f"{drift_summary['OverallDriftScore']:.2%}"
                    ),
                )

                col2.metric(
                    "Poziom zmian",
                    drift_summary[
                        "OverallSeverity"
                    ],
                )

                col3.metric(
                    "Liczba alertów",
                    drift_summary[
                        "NumberOfAlerts"
                    ],
                )

                col4.metric(
                    "Alerty wysokiego poziomu",
                    drift_summary[
                        "HighSeverityAlerts"
                    ],
                )

                col5, col6, col7, col8 = st.columns(4)

                col5.metric(
                    "Sprzedaż — okres wcześniejszy",
                    drift_summary[
                        "ReferenceSalesRecords"
                    ],
                )

                col6.metric(
                    "Sprzedaż — okres bieżący",
                    drift_summary[
                        "CurrentSalesRecords"
                    ],
                )

                col7.metric(
                    "Opinie — okres wcześniejszy",
                    drift_summary[
                        "ReferenceReviews"
                    ],
                )

                col8.metric(
                    "Opinie — okres bieżący",
                    drift_summary[
                        "CurrentReviews"
                    ],
                )

                sales_windows = drift_results[
                    "sales_windows"
                ]

                review_windows = drift_results[
                    "review_windows"
                ]

                st.caption(
                    "Okres sprzedażowy referencyjny: "
                    f"{sales_windows['ReferenceStart'].date()} — "
                    f"{sales_windows['ReferenceEnd'].date()} | "
                    "okres bieżący: "
                    f"{sales_windows['CurrentStart'].date()} — "
                    f"{sales_windows['CurrentEnd'].date()}"
                )

                st.caption(
                    "Okres opinii referencyjny: "
                    f"{review_windows['ReferenceStart'].date()} — "
                    f"{review_windows['ReferenceEnd'].date()} | "
                    "okres bieżący: "
                    f"{review_windows['CurrentStart'].date()} — "
                    f"{review_windows['CurrentEnd'].date()}"
                )

                st.divider()

                st.subheader("Wyniki monitorowanych wskaźników")

                drift_components = drift_results[
                    "drift_components"
                ]

                st.dataframe(
                    drift_components.style.format(
                        {
                            "DriftScore": "{:.2%}",
                            "ChangePct": (
                                lambda value: (
                                    ""
                                    if pd.isna(value)
                                    else f"{value:.2f}%"
                                )
                            ),
                        }
                    ),
                    width="stretch",
                )

                drift_ranking_figure = px.bar(
                    drift_components.sort_values(
                        by="DriftScore",
                        ascending=True,
                    ),
                    x="DriftScore",
                    y="Metric",
                    color="Severity",
                    orientation="h",
                    title=(
                        "Ranking wykrytych zmian "
                        "— wyższa wartość oznacza większy drift"
                    ),
                    range_x=[0, 1],
                )

                st.plotly_chart(
                    drift_ranking_figure,
                    width="stretch",
                )

                st.divider()

                st.subheader("Alerty wymagające uwagi")

                if drift_results["alerts"].empty:
                    st.success(
                        "Nie wykryto zmian średniego "
                        "ani wysokiego poziomu."
                    )
                else:
                    st.dataframe(
                        drift_results[
                            "alerts"
                        ].style.format(
                            {
                                "DriftScore": "{:.2%}",
                                "ChangePct": (
                                    lambda value: (
                                        ""
                                        if pd.isna(value)
                                        else f"{value:.2f}%"
                                    )
                                ),
                            }
                        ),
                        width="stretch",
                    )

                st.divider()

                st.subheader(
                    "Zmiana struktury przychodów produktów"
                )

                product_distribution = drift_results[
                    "product_distribution"
                ]

                product_plot_data = (
                    product_distribution.melt(
                        id_vars=["Category"],
                        value_vars=[
                            "ReferenceShare",
                            "CurrentShare",
                        ],
                        var_name="Period",
                        value_name="Share",
                    )
                )

                product_period_names = {
                    "ReferenceShare": (
                        "Okres wcześniejszy"
                    ),
                    "CurrentShare": (
                        "Okres bieżący"
                    ),
                }

                product_plot_data["Period"] = (
                    product_plot_data[
                        "Period"
                    ].replace(
                        product_period_names
                    )
                )

                product_drift_figure = px.bar(
                    product_plot_data,
                    x="Category",
                    y="Share",
                    color="Period",
                    barmode="group",
                    title=(
                        "Udział produktów w przychodzie "
                        "w porównywanych okresach"
                    ),
                )

                product_drift_figure.update_xaxes(
                    tickangle=-30
                )

                st.plotly_chart(
                    product_drift_figure,
                    width="stretch",
                )

                st.dataframe(
                    product_distribution.style.format(
                        {
                            "ReferenceValue": "{:,.2f}",
                            "CurrentValue": "{:,.2f}",
                            "ReferenceShare": "{:.2%}",
                            "CurrentShare": "{:.2%}",
                            "ShareChange": "{:+.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Zmiana rozkładu sentymentu"
                )

                sentiment_distribution = (
                    drift_results[
                        "sentiment_distribution"
                    ]
                )

                sentiment_plot_data = (
                    sentiment_distribution.melt(
                        id_vars=["Category"],
                        value_vars=[
                            "ReferenceShare",
                            "CurrentShare",
                        ],
                        var_name="Period",
                        value_name="Share",
                    )
                )

                sentiment_plot_data["Period"] = (
                    sentiment_plot_data[
                        "Period"
                    ].replace(
                        product_period_names
                    )
                )

                sentiment_drift_figure = px.bar(
                    sentiment_plot_data,
                    x="Category",
                    y="Share",
                    color="Period",
                    barmode="group",
                    title=(
                        "Sentyment klientów "
                        "w porównywanych okresach"
                    ),
                )

                st.plotly_chart(
                    sentiment_drift_figure,
                    width="stretch",
                )

                st.dataframe(
                    sentiment_distribution.style.format(
                        {
                            "ReferenceShare": "{:.2%}",
                            "CurrentShare": "{:.2%}",
                            "ShareChange": "{:+.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.divider()

                st.subheader(
                    "Zmiany słownictwa opinii"
                )

                vocabulary_comparison = (
                    drift_results[
                        "vocabulary_comparison"
                    ]
                )

                vocabulary_plot_data = (
                    vocabulary_comparison
                    .assign(
                        AbsoluteChange=lambda frame: (
                            frame["ShareChange"].abs()
                        )
                    )
                    .nlargest(
                        20,
                        "AbsoluteChange",
                    )
                    .sort_values(
                        by="ShareChange",
                        ascending=True,
                    )
                )

                vocabulary_figure = px.bar(
                    vocabulary_plot_data,
                    x="ShareChange",
                    y="Term",
                    orientation="h",
                    title=(
                        "Terminy o największej zmianie "
                        "udziału w opiniach"
                    ),
                )

                st.plotly_chart(
                    vocabulary_figure,
                    width="stretch",
                )

                st.dataframe(
                    vocabulary_comparison.style.format(
                        {
                            "ReferenceShare": "{:.2%}",
                            "CurrentShare": "{:.2%}",
                            "ShareChange": "{:+.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.warning(
                    """
                    Drift nie oznacza automatycznie pogorszenia jakości.
                    Może wynikać z sezonowości, zmian oferty, kampanii,
                    wejścia na nowe rynki lub zmiany zachowań klientów.
                    Wykryty alert powinien zostać poddany interpretacji.
                    """
                )

            except Exception as error:
                st.error(
                    "Nie udało się przygotować monitoringu driftu: "
                    f"{error}"
                )

if tab_recommendations.open:
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

if tab_summary.open:
    with tab_summary:
        st.header("Podsumowanie badania")

        st.write(
            """
            Ta sekcja prezentuje syntetyczne podsumowanie danych, metod oraz wyników
            uzyskanych w ramach działania prototypu. Informacje z tej części mogą
            zostać wykorzystane przy opisie eksperymentu badawczego w pracy magisterskiej.
            """
        )

        if filtered_sales_data is None or filtered_sales_data.empty:
            st.warning("Brak danych sprzedażowych do podsumowania.")
        else:
            sales_kpis_summary = calculate_sales_kpis(filtered_sales_data)

            min_sales_date = filtered_sales_data["InvoiceDate"].min().date()
            max_sales_date = filtered_sales_data["InvoiceDate"].max().date()

            st.subheader("Zakres danych sprzedażowych")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Rekordy sprzedażowe", len(filtered_sales_data))
            col2.metric("Liczba klientów", sales_kpis_summary["total_customers"])
            col3.metric("Liczba produktów", sales_kpis_summary["total_products"])
            col4.metric("Liczba transakcji", sales_kpis_summary["total_transactions"])

            col5, col6, col7 = st.columns(3)

            col5.metric("Łączny przychód", f"{sales_kpis_summary['total_revenue']:,.2f}")
            col6.metric("Śr. wartość zamówienia", f"{sales_kpis_summary['average_order_value']:,.2f}")
            col7.metric("Zakres dat", f"{min_sales_date} — {max_sales_date}")

            st.subheader("Wyniki prognozowania sprzedaży")

            try:
                summary_forecast_results = (
                    get_sales_forecast_results(
                        sales_data=filtered_sales_data,
                        test_fraction=0.25,
                        forecast_horizon=8,
                    )
                )

                summary_forecast_metrics = (
                    summary_forecast_results[
                        "metrics"
                    ]
                )

                summary_best_forecast = (
                    summary_forecast_metrics.iloc[0]
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Najlepszy model",
                    summary_forecast_results[
                        "best_model_name"
                    ],
                )

                col2.metric(
                    "RMSE",
                    f"{summary_best_forecast['RMSE']:,.2f}",
                )

                col3.metric(
                    "MAE",
                    f"{summary_best_forecast['MAE']:,.2f}",
                )

                col4.metric(
                    "sMAPE",
                    f"{summary_best_forecast['sMAPE']:.2f}%",
                )

            except Exception as error:
                st.warning(
                    "Nie udało się przygotować podsumowania "
                    f"prognozowania: {error}"
                )

        st.divider()

        if filtered_review_data is None or filtered_review_data.empty:
            st.warning("Brak danych tekstowych do podsumowania.")
        else:
            review_kpis_summary = review_kpis(filtered_review_data)

            st.subheader("Zakres danych tekstowych")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Liczba opinii", review_kpis_summary["total_reviews"])
            col2.metric("Średnia ocena", f"{review_kpis_summary['average_rating']:.2f}")
            col3.metric("Śr. długość opinii", f"{review_kpis_summary['average_review_length']:.0f} znaków")
            col4.metric("Produkty z opiniami", review_kpis_summary["unique_products"])

        st.divider()

        st.subheader("Zastosowane metody analityczne")

        methods_data = [
            {
                "Obszar": "Dane sprzedażowe",
                "Metoda": "Czyszczenie danych, agregacja KPI, analiza sprzedaży w czasie, ranking produktów i krajów",
                "Cel": "Identyfikacja wyników sprzedażowych oraz struktury przychodów",
            },
            {
                "Obszar": "Klienci",
                "Metoda": "Segmentacja RFM",
                "Cel": "Podział klientów według aktualności zakupów, częstotliwości i wartości zakupów",
            },
            {
                "Obszar": "Dane tekstowe",
                "Metoda": "Czyszczenie tekstu, analiza ocen, rozkład sentymentu, najczęstsze słowa",
                "Cel": "Identyfikacja nastrojów klientów i cech opinii",
            },
            {
                "Obszar": "AI/ML",
                "Metoda": (
                    "TF-IDF oraz porównanie Logistic Regression, Linear SVM, "
                    "Multinomial Naive Bayes i Complement Naive Bayes "
                    "z walidacją krzyżową"
                ),
                "Cel": (
                    "Wybór najskuteczniejszego modelu klasyfikacji sentymentu "
                    "oraz ocena jego zdolności generalizacji"
                ),
            },
            {
                "Obszar": "Explainable AI",
                "Metoda": (
                    "Globalna analiza współczynników Logistic Regression "
                    "oraz lokalna analiza wpływu terminów TF-IDF"
                ),
                "Cel": (
                    "Wyjaśnienie, jakie słowa i frazy wpływają "
                    "na klasyfikację sentymentu"
                ),
            },
            {
                "Obszar": "Deep Learning / NLP",
                "Metoda": (
                    "Wielojęzyczny model transformerowy dostrojony "
                    "do pięcioklasowej analizy sentymentu"
                ),
                "Cel": (
                    "Porównanie klasycznych metod TF-IDF "
                    "z kontekstowym modelem językowym"
                ),
            },
            {
                "Obszar": "Analiza tematów",
                "Metoda": (
                    "TF-IDF oraz Non-negative Matrix Factorization (NMF)"
                ),
                "Cel": (
                    "Automatyczne wykrywanie ukrytych tematów i aspektów "
                    "w opiniach klientów oraz analiza ich sentymentu"
                ),
            },
            {
                "Obszar": "Prognozowanie sprzedaży",
                "Metoda": (
                    "Tygodniowa agregacja przychodów, walidacja krocząca, "
                    "prognoza naiwna, średnia ruchoma, trend liniowy "
                    "oraz metoda Holta"
                ),
                "Cel": (
                    "Porównanie skuteczności modeli i oszacowanie "
                    "przyszłego poziomu przychodów"
                ),
            },
            {
                "Obszar": "Zaawansowane wsparcie decyzji",
                "Metoda": (
                    "Integracja sprzedaży, trendów produktowych, sentymentu, "
                    "tematów opinii oraz prognozy przyszłej sprzedaży"
                ),
                "Cel": (
                    "Identyfikacja ryzyka, potencjału rozwojowego oraz "
                    "priorytetów działań dla poszczególnych produktów"
                ),
            },
            {
                "Obszar": "Wsparcie decyzji",
                "Metoda": "Regułowy moduł rekomendacyjny",
                "Cel": "Generowanie rekomendacji biznesowych na podstawie danych sprzedażowych i tekstowych",
            },
            {
                "Obszar": "Monitoring modeli i danych",
                "Metoda": (
                    "Porównanie kolejnych okien czasowych, "
                    "dywergencja Jensena-Shannona oraz analiza "
                    "zmian procentowych"
                ),
                "Cel": (
                    "Wykrywanie zmian struktury sprzedaży, "
                    "sentymentu, ocen i słownictwa opinii"
                ),
            },
            {
                "Obszar": "Jakość danych",
                "Metoda": (
                    "Automatyczne reguły kompletności, poprawności "
                    "i spójności oraz ważony wskaźnik jakości"
                ),
                "Cel": (
                    "Formalna kontrola danych przed analizą "
                    "i trenowaniem modeli"
                ),
            },
        ]

        # st.dataframe(methods_data, width="stretch")
        st.dataframe(pd.DataFrame(methods_data), width="stretch")

        st.divider()

        st.subheader("Wyniki modelu AI/ML")

        if filtered_review_data is None or filtered_review_data.empty:
            st.warning("Brak danych do oceny modeli AI/ML.")
        else:
            try:
                summary_benchmark_results = (
                    get_model_comparison_results(
                        review_data=filtered_review_data,
                        requested_folds=5,
                    )
                )

                summary_comparison_table = (
                    summary_benchmark_results[
                        "comparison_table"
                    ]
                )

                summary_best_row = (
                    summary_comparison_table.iloc[0]
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Najlepszy model",
                    summary_benchmark_results[
                        "best_model_name"
                    ],
                )

                col2.metric(
                    "Accuracy",
                    f"{summary_best_row['AccuracyMean']:.2%}",
                )

                col3.metric(
                    "Macro F1",
                    f"{summary_best_row['F1MacroMean']:.2%}",
                )

                col4.metric(
                    "Walidacja",
                    (
                        f"{summary_benchmark_results['number_of_folds']}"
                        "-fold"
                    ),
                )

                st.caption(
                    "Strategia walidacji: "
                    f"{summary_benchmark_results['validation_strategy']}"
                )

            except Exception as error:
                st.warning("Nie udało się obliczyć podsumowania benchmarku: "f"{error}")

        st.divider()

        st.subheader("Podsumowanie rekomendacji")

        if filtered_sales_data is None or filtered_sales_data.empty:
            st.warning("Brak danych sprzedażowych do wygenerowania podsumowania rekomendacji.")
        else:
            summary_kpis = calculate_sales_kpis(filtered_sales_data)
            summary_top_products = top_products(filtered_sales_data, limit=10)
            summary_country_data = sales_by_country(filtered_sales_data)
            summary_rfm_data = rfm_analysis(filtered_sales_data)

            summary_sales_recommendations = generate_sales_recommendations(
                kpis=summary_kpis,
                top_products_data=summary_top_products,
                country_data=summary_country_data,
                rfm_data=summary_rfm_data,
            )

            col1, col2 = st.columns(2)

            col1.metric("Rekomendacje sprzedażowe", len(summary_sales_recommendations))

            if filtered_review_data is not None and not filtered_review_data.empty:
                summary_product_reviews = product_review_summary(filtered_review_data)
                summary_combined_recommendations = generate_combined_product_recommendations(
                    top_products_data=summary_top_products,
                    product_reviews_data=summary_product_reviews,
                )

                col2.metric("Rekomendacje łączone", len(summary_combined_recommendations))
            else:
                col2.metric("Rekomendacje łączone", 0)

        st.divider()

        st.subheader("Interpretacja ogólna")

        st.write(
            """
            Uzyskane wyniki pokazują, że połączenie analizy danych sprzedażowych,
            analizy opinii klientów oraz metod uczenia maszynowego może wspierać
            procesy decyzyjne przedsiębiorstwa. Moduł sprzedażowy umożliwia ocenę
            wyników produktów, klientów i rynków, natomiast moduł tekstowy pozwala
            uwzględnić głos klienta. Warstwa rekomendacyjna integruje te informacje
            i wskazuje obszary wymagające działań biznesowych.
            """
        )

if tab_export.open:
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

            prepare_full_export_package = st.toggle(
                "Przygotuj pełny pakiet eksportów",
                value=False,
                key="prepare_full_export_package",
                help=(
                    "Włączenie tej opcji uruchamia "
                    "przygotowanie wszystkich wyników "
                    "i może potrwać dłużej."
                ),
            )

            if not prepare_full_export_package:
                st.info(
                    "Włącz przygotowanie pakietu, "
                    "aby wygenerować pliki CSV z wynikami "
                    "analiz. Pozostałe moduły nie są "
                    "teraz ponownie obliczane."
                )
            else:
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

                st.subheader("Porównanie modeli AI/ML")

                if filtered_review_data is None or filtered_review_data.empty:
                    st.warning(
                        "Brak danych tekstowych do eksportu benchmarku."
                    )
                else:
                    try:
                        export_benchmark_results = (
                            get_model_comparison_results(
                                review_data=filtered_review_data,
                                requested_folds=5,
                            )
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.download_button(
                                label="Pobierz ranking modeli",
                                data=convert_dataframe_to_csv(
                                    export_benchmark_results[
                                        "comparison_table"
                                    ]
                                ),
                                file_name="sentiment_model_comparison.csv",
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz raport najlepszego modelu",
                                data=convert_dataframe_to_csv(
                                    export_benchmark_results[
                                        "classification_report"
                                    ]
                                ),
                                file_name="best_model_classification_report.csv",
                                mime="text/csv",
                            )

                        with col3:
                            st.download_button(
                                label="Pobierz analizę błędów",
                                data=convert_dataframe_to_csv(
                                    export_benchmark_results[
                                        "errors"
                                    ]
                                ),
                                file_name="best_model_error_analysis.csv",
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu benchmarku: "
                            f"{error}"
                        )

                st.subheader("Interpretowalność modelu")

                if filtered_review_data is None or filtered_review_data.empty:
                    st.warning(
                        "Brak danych tekstowych do eksportu interpretacji."
                    )
                else:
                    try:
                        export_interpretability = (
                            get_interpretability_results(
                                review_data=filtered_review_data,
                                top_n=20,
                            )
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.download_button(
                                label="Pobierz terminy wspierające klasy",
                                data=convert_dataframe_to_csv(
                                    export_interpretability[
                                        "supporting_terms"
                                    ]
                                ),
                                file_name=(
                                    "sentiment_supporting_terms.csv"
                                ),
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz terminy działające przeciw klasom",
                                data=convert_dataframe_to_csv(
                                    export_interpretability[
                                        "opposing_terms"
                                    ]
                                ),
                                file_name=(
                                    "sentiment_opposing_terms.csv"
                                ),
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu "
                            f"interpretacji: {error}"
                        )

                st.subheader("Modelowanie tematów opinii")

                if filtered_review_data is None or filtered_review_data.empty:
                    st.warning(
                        "Brak danych tekstowych do eksportu tematów."
                    )
                else:
                    try:
                        export_topic_results = (
                            get_topic_analysis_results(
                                review_data=filtered_review_data,
                                number_of_topics=5,
                                top_terms_per_topic=10,
                            )
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.download_button(
                                label="Pobierz wykryte tematy",
                                data=convert_dataframe_to_csv(
                                    export_topic_results[
                                        "topic_terms"
                                    ]
                                ),
                                file_name="review_topic_terms.csv",
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz przypisania opinii",
                                data=convert_dataframe_to_csv(
                                    export_topic_results[
                                        "review_assignments"
                                    ]
                                ),
                                file_name="review_topic_assignments.csv",
                                mime="text/csv",
                            )

                        with col3:
                            st.download_button(
                                label="Pobierz sentyment tematów",
                                data=convert_dataframe_to_csv(
                                    export_topic_results[
                                        "topic_sentiment_summary"
                                    ]
                                ),
                                file_name="topic_sentiment_summary.csv",
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu "
                            f"modelowania tematów: {error}"
                        )

                st.subheader("Prognozowanie sprzedaży")

                if filtered_sales_data is None or filtered_sales_data.empty:
                    st.warning(
                        "Brak danych sprzedażowych do eksportu prognoz."
                    )
                else:
                    try:
                        export_forecast_results = (
                            get_sales_forecast_results(
                                sales_data=filtered_sales_data,
                                test_fraction=0.25,
                                forecast_horizon=8,
                            )
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.download_button(
                                label="Pobierz ranking modeli prognoz",
                                data=convert_dataframe_to_csv(
                                    export_forecast_results[
                                        "metrics"
                                    ]
                                ),
                                file_name=(
                                    "sales_forecast_model_ranking.csv"
                                ),
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz wyniki walidacji",
                                data=convert_dataframe_to_csv(
                                    export_forecast_results[
                                        "validation_predictions"
                                    ]
                                ),
                                file_name=(
                                    "sales_forecast_validation.csv"
                                ),
                                mime="text/csv",
                            )

                        with col3:
                            st.download_button(
                                label="Pobierz przyszłą prognozę",
                                data=convert_dataframe_to_csv(
                                    export_forecast_results[
                                        "future_forecast"
                                    ]
                                ),
                                file_name=(
                                    "future_sales_forecast.csv"
                                ),
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu prognoz: "
                            f"{error}"
                        )

                st.subheader("Zaawansowane centrum decyzji")

                if (
                    filtered_sales_data is None
                    or filtered_sales_data.empty
                    or filtered_review_data is None
                    or filtered_review_data.empty
                ):
                    st.warning(
                        "Brak danych wymaganych do eksportu "
                        "zaawansowanych rekomendacji."
                    )
                else:
                    try:
                        export_decision_topics = (
                            get_topic_analysis_results(
                                review_data=filtered_review_data,
                                number_of_topics=5,
                                top_terms_per_topic=10,
                            )
                        )

                        export_decision_forecast = (
                            get_sales_forecast_results(
                                sales_data=filtered_sales_data,
                                test_fraction=0.25,
                                forecast_horizon=8,
                            )
                        )

                        export_decision_results = (
                            get_advanced_decision_results(
                                sales_data=filtered_sales_data,
                                review_data=filtered_review_data,
                                topic_assignments=(
                                    export_decision_topics[
                                        "review_assignments"
                                    ]
                                ),
                                weekly_data=(
                                    export_decision_forecast[
                                        "weekly_data"
                                    ]
                                ),
                                future_forecast=(
                                    export_decision_forecast[
                                        "future_forecast"
                                    ]
                                ),
                            )
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.download_button(
                                label="Pobierz macierz produktów",
                                data=convert_dataframe_to_csv(
                                    export_decision_results[
                                        "product_matrix"
                                    ]
                                ),
                                file_name=(
                                    "advanced_product_decision_matrix.csv"
                                ),
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz rekomendacje 3.0",
                                data=convert_dataframe_to_csv(
                                    export_decision_results[
                                        "recommendations"
                                    ]
                                ),
                                file_name=(
                                    "advanced_business_recommendations.csv"
                                ),
                                mime="text/csv",
                            )

                        with col3:
                            st.download_button(
                                label="Pobierz perspektywę prognozy",
                                data=convert_dataframe_to_csv(
                                    export_decision_results[
                                        "forecast_outlook"
                                    ]
                                ),
                                file_name=(
                                    "forecast_business_outlook.csv"
                                ),
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu centrum decyzji: "
                            f"{error}"
                        )

                st.subheader("Monitoring danych i drift")

                if (
                    filtered_sales_data is None
                    or filtered_sales_data.empty
                    or filtered_review_data is None
                    or filtered_review_data.empty
                ):
                    st.warning(
                        "Brak danych wymaganych do eksportu monitoringu."
                    )
                else:
                    try:
                        export_drift_results = (
                            get_drift_monitoring_results(
                                sales_data=filtered_sales_data,
                                review_data=filtered_review_data,
                                window_weeks=8,
                            )
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.download_button(
                                label="Pobierz wyniki driftu",
                                data=convert_dataframe_to_csv(
                                    export_drift_results[
                                        "drift_components"
                                    ]
                                ),
                                file_name="data_drift_metrics.csv",
                                mime="text/csv",
                            )

                        with col2:
                            st.download_button(
                                label="Pobierz alerty driftu",
                                data=convert_dataframe_to_csv(
                                    export_drift_results[
                                        "alerts"
                                    ]
                                ),
                                file_name="data_drift_alerts.csv",
                                mime="text/csv",
                            )

                        with col3:
                            st.download_button(
                                label="Pobierz zmianę słownictwa",
                                data=convert_dataframe_to_csv(
                                    export_drift_results[
                                        "vocabulary_comparison"
                                    ]
                                ),
                                file_name="review_vocabulary_drift.csv",
                                mime="text/csv",
                            )

                    except Exception as error:
                        st.warning(
                            "Nie udało się przygotować eksportu driftu: "
                            f"{error}"
                        )

                st.subheader("Kontrola jakości danych")

                try:
                    export_quality_results = (
                        get_data_quality_results(
                            sales_data=filtered_sales_data,
                            review_data=filtered_review_data,
                            minimum_review_length=10,
                        )
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.download_button(
                            label="Pobierz raport jakości",
                            data=convert_dataframe_to_csv(
                                export_quality_results[
                                    "quality_checks"
                                ]
                            ),
                            file_name="data_quality_report.csv",
                            mime="text/csv",
                        )

                    with col2:
                        st.download_button(
                            label="Pobierz wykryte problemy",
                            data=convert_dataframe_to_csv(
                                export_quality_results[
                                    "failed_checks"
                                ]
                            ),
                            file_name="data_quality_issues.csv",
                            mime="text/csv",
                        )

                    with col3:
                        st.download_button(
                            label="Pobierz zalecenia naprawcze",
                            data=convert_dataframe_to_csv(
                                export_quality_results[
                                    "recommendations"
                                ]
                            ),
                            file_name="data_quality_recommendations.csv",
                            mime="text/csv",
                        )

                except Exception as error:
                    st.warning(
                        "Nie udało się przygotować eksportu "
                        f"kontroli jakości: {error}"
                    )

                st.subheader("Model transformerowy")

                prepare_transformer_export = st.checkbox(
                    "Przygotuj wyniki transformera do eksportu",
                    value=False,
                    key="prepare_transformer_export",
                )

                if prepare_transformer_export:
                    if (
                        filtered_review_data is None
                        or filtered_review_data.empty
                    ):
                        st.warning(
                            "Brak opinii do eksportu wyników "
                            "modelu transformerowego."
                        )
                    else:
                        try:
                            with st.spinner(
                                "Trwa przygotowywanie wyników "
                                "modelu transformerowego..."
                            ):
                                export_transformer_results = (
                                    get_transformer_evaluation_results(
                                        review_data=(
                                            filtered_review_data
                                        ),
                                        model_name=(
                                            DEFAULT_TRANSFORMER_MODEL
                                        ),
                                        batch_size=16,
                                    )
                                )

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.download_button(
                                    label="Pobierz metryki modelu transformerowego",
                                    data=convert_dataframe_to_csv(
                                        export_transformer_results[
                                            "metrics"
                                        ]
                                    ),
                                    file_name=(
                                        "transformer_sentiment_metrics.csv"
                                    ),
                                    mime="text/csv",
                                )

                            with col2:
                                st.download_button(
                                    label="Pobierz predykcje modelu transformerowego",
                                    data=convert_dataframe_to_csv(
                                        export_transformer_results[
                                            "predictions"
                                        ]
                                    ),
                                    file_name=(
                                        "transformer_sentiment_predictions.csv"
                                    ),
                                    mime="text/csv",
                                )

                            with col3:
                                st.download_button(
                                    label="Pobierz błędy modelu transformerowego",
                                    data=convert_dataframe_to_csv(
                                        export_transformer_results[
                                            "errors"
                                        ]
                                    ),
                                    file_name=(
                                        "transformer_sentiment_errors.csv"
                                    ),
                                    mime="text/csv",
                                )

                        except Exception as error:
                            st.warning(
                                "Nie udało się przygotować eksportu modelu transformerowego: "
                                f"{error}"
                            )

                st.divider()

                st.info(
                    """
                    Funkcja eksportu wyników zwiększa praktyczną użyteczność prototypu,
                    ponieważ pozwala przenieść rezultaty analizy do arkusza kalkulacyjnego
                    lub wykorzystać je jako załączniki do raportu biznesowego.
                    """
                )

render_footer(
    version=APP_VERSION,
    disclaimer=DATA_DISCLAIMER,
    )
