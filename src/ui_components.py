from __future__ import annotations

import html

import pandas as pd
import streamlit as st


def inject_global_styles() -> None:
    """
    Dodaje ograniczony zestaw stylów poprawiających
    czytelność interfejsu aplikacji.
    """
    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 1500px;
                padding-top: 1.6rem;
                padding-bottom: 3rem;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 12px;
                padding: 0.9rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            }

            div[data-testid="stMetricLabel"] {
                font-weight: 600;
            }

            div[data-testid="stAlert"] {
                border-radius: 10px;
            }

            .app-header {
                padding: 1.3rem 1.5rem;
                margin-bottom: 1.2rem;
                border-radius: 16px;
                background:
                    linear-gradient(
                        135deg,
                        rgba(37, 99, 235, 0.12),
                        rgba(14, 165, 233, 0.07)
                    );
                border: 1px solid rgba(37, 99, 235, 0.18);
            }

            .app-header h1 {
                margin: 0;
                font-size: 2rem;
                color: #0F172A;
            }

            .app-header p {
                margin: 0.55rem 0 0 0;
                color: #475569;
                line-height: 1.6;
            }

            .app-version {
                display: inline-block;
                margin-top: 0.8rem;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                background: #DBEAFE;
                color: #1D4ED8;
                font-size: 0.82rem;
                font-weight: 700;
            }

            .app-footer {
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(148, 163, 184, 0.35);
                color: #64748B;
                font-size: 0.82rem;
                text-align: center;
            }

            .stApp {
                background:
                    linear-gradient(
                        180deg,
                        #F8FAFC 0%,
                        #FFFFFF 24rem
                    );
            }

            section[data-testid="stSidebar"] {
                background:
                    linear-gradient(
                        180deg,
                        #F8FAFC 0%,
                        #EFF6FF 100%
                    );
                border-right:
                    1px solid rgba(
                        148,
                        163,
                        184,
                        0.28
                    );
            }

            section[data-testid="stSidebar"]
            div[data-baseweb="select"] > div {
                min-height: 3rem;
                border-radius: 12px;
                border:
                    1px solid rgba(
                        37,
                        99,
                        235,
                        0.25
                    );
                background: #FFFFFF;
                box-shadow:
                    0 4px 12px rgba(
                        15,
                        23,
                        42,
                        0.05
                    );
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 14px;
                border-color:
                    rgba(
                        148,
                        163,
                        184,
                        0.32
                    );
                background:
                    rgba(
                        255,
                        255,
                        255,
                        0.72
                    );
            }

            div[data-testid="stDataFrame"] {
                overflow: hidden;
                border-radius: 12px;
                border:
                    1px solid rgba(
                        148,
                        163,
                        184,
                        0.30
                    );
            }

            div.stButton > button,
            div.stDownloadButton > button {
                min-height: 2.65rem;
                border-radius: 10px;
                font-weight: 650;
                border:
                    1px solid rgba(
                        37,
                        99,
                        235,
                        0.24
                    );
                box-shadow:
                    0 3px 10px rgba(
                        15,
                        23,
                        42,
                        0.05
                    );
            }

            div.stButton > button:hover,
            div.stDownloadButton > button:hover {
                border-color: #2563EB;
                transform: translateY(-1px);
            }

            h1,
            h2,
            h3 {
                letter-spacing: -0.025em;
            }

            h2 {
                padding-bottom: 0.35rem;
                border-bottom:
                    1px solid rgba(
                        148,
                        163,
                        184,
                        0.22
                    );
            }

            div[data-testid="stExpander"] {
                border-radius: 12px;
                overflow: hidden;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(
    app_name: str,
    description: str,
    version: str,
) -> None:
    """
    Wyświetla główny nagłówek aplikacji.
    """
    safe_name = html.escape(app_name)
    safe_description = html.escape(description)
    safe_version = html.escape(version)

    st.markdown(
        f"""
        <div class="app-header">
            <h1>{safe_name}</h1>
            <p>{safe_description}</p>
            <span class="app-version">Wersja {safe_version}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(
    sales_data: pd.DataFrame | None,
    review_data: pd.DataFrame | None,
    version: str,
) -> None:
    """
    Wyświetla status aplikacji oraz aktualnie dostępnych danych.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Status systemu")

    sales_records = (
        len(sales_data)
        if sales_data is not None
        else 0
    )

    review_records = (
        len(review_data)
        if review_data is not None
        else 0
    )

    sales_status = (
        "Dostępne"
        if sales_records > 0
        else "Brak danych"
    )

    review_status = (
        "Dostępne"
        if review_records > 0
        else "Brak danych"
    )

    st.sidebar.write(
        f"**Wersja:** {version}"
    )

    st.sidebar.write(
        f"**Dane sprzedażowe:** {sales_status}"
    )

    st.sidebar.caption(
        f"Liczba aktywnych rekordów: {sales_records:,}"
    )

    st.sidebar.write(
        f"**Opinie klientów:** {review_status}"
    )

    st.sidebar.caption(
        f"Liczba aktywnych opinii: {review_records:,}"
    )

    if sales_records > 0 and review_records > 0:
        st.sidebar.success(
            "System gotowy do pełnej analizy."
        )
    elif sales_records > 0 or review_records > 0:
        st.sidebar.warning(
            "Dostępna jest tylko część wymaganych danych."
        )
    else:
        st.sidebar.error(
            "Brak danych do analizy."
        )


def render_footer(
    version: str,
    disclaimer: str,
) -> None:
    """
    Wyświetla stopkę aplikacji.
    """
    safe_version = html.escape(version)
    safe_disclaimer = html.escape(disclaimer)

    st.markdown(
        f"""
        <div class="app-footer">
            AI Business Insight System — wersja {safe_version}<br>
            {safe_disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )