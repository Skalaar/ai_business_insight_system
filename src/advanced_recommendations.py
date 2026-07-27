from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_product_sales(
    sales_data: pd.DataFrame,
    trend_window_weeks: int = 8,
) -> pd.DataFrame:
    """
    Przygotowuje produktowe wskaźniki sprzedaży i trendu.

    """
    required_columns = [
        "Description",
        "InvoiceDate",
        "TotalPrice",
        "Quantity",
        "InvoiceNo",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in sales_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn sprzedażowych wymaganych przez "
            f"centrum decyzji: {missing_columns}"
        )

    prepared = sales_data.copy()

    prepared["InvoiceDate"] = pd.to_datetime(
        prepared["InvoiceDate"],
        errors="coerce",
    )

    prepared["TotalPrice"] = pd.to_numeric(
        prepared["TotalPrice"],
        errors="coerce",
    )

    prepared["Quantity"] = pd.to_numeric(
        prepared["Quantity"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            "Description",
            "InvoiceDate",
            "TotalPrice",
        ]
    )

    if prepared.empty:
        raise ValueError(
            "Brak poprawnych danych sprzedażowych "
            "do utworzenia macierzy decyzyjnej."
        )

    aggregation = {
        "Revenue": (
            "TotalPrice",
            "sum",
        ),
        "Quantity": (
            "Quantity",
            "sum",
        ),
        "Transactions": (
            "InvoiceNo",
            "nunique",
        ),
    }

    if "CustomerID" in prepared.columns:
        aggregation["Customers"] = (
            "CustomerID",
            "nunique",
        )

    product_sales = (
        prepared
        .groupby(
            "Description",
            as_index=False,
        )
        .agg(**aggregation)
        .rename(
            columns={
                "Description": "ProductName",
            }
        )
    )

    if "Customers" not in product_sales.columns:
        product_sales["Customers"] = np.nan

    maximum_date = prepared[
        "InvoiceDate"
    ].max()

    recent_start = (
        maximum_date
        - pd.Timedelta(
            weeks=trend_window_weeks
        )
    )

    previous_start = (
        recent_start
        - pd.Timedelta(
            weeks=trend_window_weeks
        )
    )

    recent_data = prepared[
        prepared["InvoiceDate"] > recent_start
    ]

    previous_data = prepared[
        (
            prepared["InvoiceDate"]
            > previous_start
        )
        & (
            prepared["InvoiceDate"]
            <= recent_start
        )
    ]

    recent_revenue = (
        recent_data
        .groupby("Description")[
            "TotalPrice"
        ]
        .sum()
        .rename("RecentRevenue")
    )

    previous_revenue = (
        previous_data
        .groupby("Description")[
            "TotalPrice"
        ]
        .sum()
        .rename("PreviousRevenue")
    )

    trend_data = pd.concat(
        [
            recent_revenue,
            previous_revenue,
        ],
        axis=1,
    ).fillna(0.0)

    trend_data["RevenueTrendPct"] = np.where(
        trend_data["PreviousRevenue"] > 0,
        (
            (
                trend_data["RecentRevenue"]
                - trend_data["PreviousRevenue"]
            )
            / trend_data["PreviousRevenue"]
            * 100
        ),
        np.where(
            trend_data["RecentRevenue"] > 0,
            100.0,
            0.0,
        ),
    )

    trend_data["RevenueTrendPct"] = (
        trend_data["RevenueTrendPct"]
        .clip(
            lower=-100.0,
            upper=300.0,
        )
    )

    trend_data = (
        trend_data
        .reset_index()
        .rename(
            columns={
                "Description": "ProductName",
            }
        )
    )

    product_sales = product_sales.merge(
        trend_data,
        on="ProductName",
        how="left",
    )

    trend_columns = [
        "RecentRevenue",
        "PreviousRevenue",
        "RevenueTrendPct",
    ]

    product_sales[trend_columns] = (
        product_sales[trend_columns]
        .fillna(0.0)
    )

    total_revenue = product_sales[
        "Revenue"
    ].sum()

    product_sales["RevenueShare"] = np.where(
        total_revenue > 0,
        product_sales["Revenue"]
        / total_revenue,
        0.0,
    )

    product_sales["SalesImportance"] = (
        product_sales["Revenue"]
        .rank(
            method="average",
            pct=True,
        )
    )

    return product_sales


def _prepare_product_reviews(
    review_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje produktowe wskaźniki ocen i sentymentu.
    """
    required_columns = [
        "ProductName",
        "Rating",
        "RatingSentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in review_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn tekstowych wymaganych przez "
            f"centrum decyzji: {missing_columns}"
        )

    prepared = review_data.dropna(
        subset=required_columns
    ).copy()

    prepared["Rating"] = pd.to_numeric(
        prepared["Rating"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=["Rating"]
    )

    if prepared.empty:
        raise ValueError(
            "Brak poprawnych opinii do utworzenia "
            "macierzy decyzyjnej."
        )

    review_summary = (
        prepared
        .groupby(
            "ProductName",
            as_index=False,
        )
        .agg(
            Reviews=(
                "RatingSentiment",
                "size",
            ),
            AverageRating=(
                "Rating",
                "mean",
            ),
        )
    )

    sentiment_counts = pd.crosstab(
        prepared["ProductName"],
        prepared["RatingSentiment"],
    )

    expected_sentiments = [
        "Pozytywny",
        "Neutralny",
        "Negatywny",
    ]

    for sentiment in expected_sentiments:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    sentiment_counts = (
        sentiment_counts[
            expected_sentiments
        ]
        .reset_index()
        .rename(
            columns={
                "Pozytywny": "PositiveReviews",
                "Neutralny": "NeutralReviews",
                "Negatywny": "NegativeReviews",
            }
        )
    )

    review_summary = review_summary.merge(
        sentiment_counts,
        on="ProductName",
        how="left",
    )

    review_summary["PositiveShare"] = (
        review_summary["PositiveReviews"]
        / review_summary["Reviews"]
    )

    review_summary["NeutralShare"] = (
        review_summary["NeutralReviews"]
        / review_summary["Reviews"]
    )

    review_summary["NegativeShare"] = (
        review_summary["NegativeReviews"]
        / review_summary["Reviews"]
    )

    review_summary["SentimentBalance"] = (
        review_summary["PositiveShare"]
        - review_summary["NegativeShare"]
    )

    return review_summary


def _prepare_product_topics(
    topic_assignments: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Wskazuje dominujący temat opinii dla każdego produktu.
    """
    if (
        topic_assignments is None
        or topic_assignments.empty
        or "ProductName"
        not in topic_assignments.columns
        or "TopicLabel"
        not in topic_assignments.columns
    ):
        return pd.DataFrame(
            columns=[
                "ProductName",
                "DominantTopic",
                "TopicReviews",
            ]
        )

    topic_counts = (
        topic_assignments
        .groupby(
            [
                "ProductName",
                "TopicLabel",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "TopicReviews",
            }
        )
        .sort_values(
            by=[
                "ProductName",
                "TopicReviews",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    dominant_topics = (
        topic_counts
        .drop_duplicates(
            subset=["ProductName"],
            keep="first",
        )
        .rename(
            columns={
                "TopicLabel": "DominantTopic",
            }
        )
    )

    return dominant_topics[
        [
            "ProductName",
            "DominantTopic",
            "TopicReviews",
        ]
    ]


def _calculate_forecast_outlook(
    weekly_data: pd.DataFrame | None,
    future_forecast: pd.DataFrame | None,
) -> dict:
    """
    Porównuje prognozowany przychód z ostatnimi tygodniami historii.
    """
    default_result = {
        "RecentAverageRevenue": np.nan,
        "FutureAverageRevenue": np.nan,
        "ForecastChangePct": np.nan,
        "ForecastDirection": "Brak prognozy",
    }

    if (
        weekly_data is None
        or future_forecast is None
        or weekly_data.empty
        or future_forecast.empty
    ):
        return default_result

    if (
        "Revenue" not in weekly_data.columns
        or "ForecastRevenue"
        not in future_forecast.columns
    ):
        return default_result

    recent_average = float(
        weekly_data["Revenue"]
        .tail(8)
        .mean()
    )

    future_average = float(
        future_forecast[
            "ForecastRevenue"
        ].mean()
    )

    if recent_average > 0:
        change_percentage = (
            (
                future_average
                - recent_average
            )
            / recent_average
            * 100
        )
    else:
        change_percentage = np.nan

    if pd.isna(change_percentage):
        direction = "Brak prognozy"
    elif change_percentage >= 5:
        direction = "Prognozowany wzrost"
    elif change_percentage <= -5:
        direction = "Prognozowany spadek"
    else:
        direction = "Stabilizacja"

    return {
        "RecentAverageRevenue": recent_average,
        "FutureAverageRevenue": future_average,
        "ForecastChangePct": float(
            change_percentage
        ),
        "ForecastDirection": direction,
    }


def _assign_decision_quadrant(
    row: pd.Series,
) -> str:
    """
    Przypisuje produkt do macierzy sprzedaż–sentyment.
    """
    high_sales = (
        row["SalesImportance"] >= 0.50
    )

    positive_sentiment = (
        row["SentimentBalance"] >= 0.20
    )

    if high_sales and positive_sentiment:
        return "Produkt strategiczny"

    if high_sales and not positive_sentiment:
        return "Ryzyko jakościowe"

    if not high_sales and positive_sentiment:
        return "Szansa rozwojowa"

    return "Obszar diagnostyczny"


def _calculate_decision_scores(
    product_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Oblicza syntetyczne wskaźniki ryzyka i potencjału.
    """
    matrix = product_matrix.copy()

    matrix["ReviewEvidence"] = (
        matrix["Reviews"]
        .fillna(0)
        .div(20)
        .clip(
            lower=0,
            upper=1,
        )
    )

    matrix["RatingNormalized"] = (
        matrix["AverageRating"]
        .fillna(3.0)
        .div(5)
        .clip(
            lower=0,
            upper=1,
        )
    )

    matrix["DeclineSeverity"] = (
        -matrix["RevenueTrendPct"]
        .div(100)
    ).clip(
        lower=0,
        upper=1,
    )

    matrix["GrowthStrength"] = (
        matrix["RevenueTrendPct"]
        .div(100)
    ).clip(
        lower=0,
        upper=1,
    )

    quality_risk = (
        matrix["NegativeShare"]
        * (
            0.5
            + 0.5
            * matrix["SalesImportance"]
        )
    )

    decline_risk = (
        matrix["DeclineSeverity"]
        * (
            0.5
            + 0.5
            * matrix["SalesImportance"]
        )
    )

    matrix["RiskScore"] = (
        100
        * (
            0.50 * quality_risk
            + 0.30 * decline_risk
            + 0.15
            * (
                1
                - matrix["RatingNormalized"]
            )
            * matrix["ReviewEvidence"]
            + 0.05
            * (
                1
                - matrix["ReviewEvidence"]
            )
        )
    ).clip(
        lower=0,
        upper=100,
    )

    matrix["OpportunityScore"] = (
        100
        * (
            0.40
            * matrix["PositiveShare"]
            * (
                1
                - matrix["SalesImportance"]
            )
            + 0.25
            * matrix["GrowthStrength"]
            + 0.20
            * matrix["ReviewEvidence"]
            + 0.15
            * matrix["RatingNormalized"]
        )
    ).clip(
        lower=0,
        upper=100,
    )

    matrix["DecisionQuadrant"] = matrix.apply(
        _assign_decision_quadrant,
        axis=1,
    )

    return matrix


def _generate_product_recommendations(
    product_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generuje rekomendacje dla poszczególnych produktów.
    """
    recommendation_rows = []

    for _, row in product_matrix.iterrows():
        product_name = row["ProductName"]
        topic = row.get(
            "DominantTopic",
            "Brak dominującego tematu",
        )

        if pd.isna(topic):
            topic = "Brak dominującego tematu"

        if row["Reviews"] < 5:
            category = "Luka informacyjna"
            priority = "Niski"
            conclusion = (
                f"Produkt posiada jedynie {int(row['Reviews'])} opinii, "
                "co ogranicza wiarygodność analizy sentymentu."
            )
            recommendation = (
                "Zwiększyć liczbę pozyskiwanych opinii, np. poprzez "
                "automatyczne prośby o ocenę po zakupie."
            )

        elif (
            row["NegativeShare"] >= 0.30
            and row["SalesImportance"] >= 0.50
        ):
            category = "Ryzyko jakościowe"
            priority = "Wysoki"
            conclusion = (
                f"Produkt ma istotny udział w sprzedaży, natomiast "
                f"{row['NegativeShare']:.1%} opinii jest negatywnych. "
                f"Dominujący temat: {topic}."
            )
            recommendation = (
                "Przeprowadzić pilną analizę jakości produktu, opisu "
                "oferty i najczęściej zgłaszanych problemów."
            )

        elif (
            row["RevenueTrendPct"] >= 20
            and row["NegativeShare"] >= 0.25
        ):
            category = "Wzrost obarczony ryzykiem"
            priority = "Wysoki"
            conclusion = (
                f"Przychód produktu wzrósł o "
                f"{row['RevenueTrendPct']:.1f}%, ale udział "
                f"negatywnych opinii wynosi "
                f"{row['NegativeShare']:.1%}."
            )
            recommendation = (
                "Zabezpieczyć jakość i obsługę rosnącej sprzedaży, "
                "zanim problemy klientów ograniczą dalszy wzrost."
            )

        elif (
            row["RevenueTrendPct"] <= -20
            and row["PositiveShare"] >= 0.60
        ):
            category = "Spadek mimo dobrych opinii"
            priority = "Średni"
            conclusion = (
                f"Przychód produktu spadł o "
                f"{abs(row['RevenueTrendPct']):.1f}%, mimo że "
                f"{row['PositiveShare']:.1%} opinii jest pozytywnych."
            )
            recommendation = (
                "Sprawdzić widoczność produktu, dostępność, cenę "
                "oraz działania konkurencji."
            )

        elif (
            row["SalesImportance"] < 0.50
            and row["PositiveShare"] >= 0.70
        ):
            category = "Szansa promocyjna"
            priority = "Średni"
            conclusion = (
                f"Produkt ma relatywnie niski udział w sprzedaży, "
                f"ale {row['PositiveShare']:.1%} opinii jest pozytywnych."
            )
            recommendation = (
                "Rozważyć kampanię promocyjną, lepszą ekspozycję "
                "lub wykorzystanie produktu w rekomendacjach."
            )

        elif row["DecisionQuadrant"] == "Produkt strategiczny":
            category = "Utrzymanie przewagi"
            priority = "Średni"
            conclusion = (
                "Produkt łączy wysokie znaczenie sprzedażowe "
                "z korzystnym bilansem opinii klientów."
            )
            recommendation = (
                "Utrzymać dostępność produktu, monitorować jakość "
                "i wykorzystać go jako punkt odniesienia dla oferty."
            )

        else:
            category = "Monitoring"
            priority = "Niski"
            conclusion = (
                "Nie wykryto obecnie sygnału wymagającego "
                "natychmiastowej interwencji."
            )
            recommendation = (
                "Kontynuować monitoring sprzedaży, trendu "
                "i opinii klientów."
            )

        risk_categories = {
            "Ryzyko jakościowe",
            "Wzrost obarczony ryzykiem",
            "Spadek mimo dobrych opinii",
        }

        opportunity_categories = {
            "Szansa promocyjna",
            "Utrzymanie przewagi",
        }

        if category in risk_categories:
            priority_score = float(row["RiskScore"])
        elif category in opportunity_categories:
            priority_score = float(row["OpportunityScore"])
        else:
            priority_score = max(
                float(row["RiskScore"]),
                float(row["OpportunityScore"]),
            )

        recommendation_rows.append(
            {
                "ProductName": product_name,
                "Category": category,
                "Priority": priority,
                "PriorityScore": priority_score,
                "DecisionQuadrant": row[
                    "DecisionQuadrant"
                ],
                "Revenue": row["Revenue"],
                "RevenueTrendPct": row[
                    "RevenueTrendPct"
                ],
                "AverageRating": row[
                    "AverageRating"
                ],
                "PositiveShare": row[
                    "PositiveShare"
                ],
                "NegativeShare": row[
                    "NegativeShare"
                ],
                "DominantTopic": topic,
                "Conclusion": conclusion,
                "Recommendation": recommendation,
            }
        )

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    priority_order = pd.CategoricalDtype(
        categories=[
            "Wysoki",
            "Średni",
            "Niski",
        ],
        ordered=True,
    )

    recommendations["Priority"] = (
        recommendations["Priority"]
        .astype(priority_order)
    )

    recommendations = (
        recommendations
        .sort_values(
            by=[
                "Priority",
                "PriorityScore",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    recommendations["Priority"] = (
        recommendations["Priority"]
        .astype(str)
    )

    return recommendations


def build_advanced_decision_support(
    sales_data: pd.DataFrame,
    review_data: pd.DataFrame,
    topic_assignments: pd.DataFrame | None = None,
    weekly_data: pd.DataFrame | None = None,
    future_forecast: pd.DataFrame | None = None,
    trend_window_weeks: int = 8,
) -> dict:
    """
    Buduje zintegrowane centrum wspomagania decyzji.
    """
    product_sales = _prepare_product_sales(
        sales_data=sales_data,
        trend_window_weeks=trend_window_weeks,
    )

    product_reviews = _prepare_product_reviews(
        review_data=review_data,
    )

    product_topics = _prepare_product_topics(
        topic_assignments=topic_assignments,
    )

    product_matrix = product_sales.merge(
        product_reviews,
        on="ProductName",
        how="left",
    )

    product_matrix = product_matrix.merge(
        product_topics,
        on="ProductName",
        how="left",
    )

    numeric_review_columns = [
        "Reviews",
        "AverageRating",
        "PositiveReviews",
        "NeutralReviews",
        "NegativeReviews",
        "PositiveShare",
        "NeutralShare",
        "NegativeShare",
        "SentimentBalance",
        "TopicReviews",
    ]

    for column in numeric_review_columns:
        if column not in product_matrix.columns:
            product_matrix[column] = 0.0

    product_matrix[numeric_review_columns] = (
        product_matrix[numeric_review_columns]
        .fillna(0.0)
    )

    product_matrix = _calculate_decision_scores(
        product_matrix
    )

    recommendations = (
        _generate_product_recommendations(
            product_matrix
        )
    )

    forecast_outlook = (
        _calculate_forecast_outlook(
            weekly_data=weekly_data,
            future_forecast=future_forecast,
        )
    )

    top_risk_product = (
        product_matrix
        .sort_values(
            by="RiskScore",
            ascending=False,
        )
        .iloc[0]["ProductName"]
    )

    top_opportunity_product = (
        product_matrix
        .sort_values(
            by="OpportunityScore",
            ascending=False,
        )
        .iloc[0]["ProductName"]
    )

    executive_summary = {
        "ProductsAnalyzed": len(
            product_matrix
        ),
        "HighPriorityRecommendations": int(
            (
                recommendations["Priority"]
                == "Wysoki"
            ).sum()
        ),
        "MediumPriorityRecommendations": int(
            (
                recommendations["Priority"]
                == "Średni"
            ).sum()
        ),
        "TopRiskProduct": top_risk_product,
        "TopOpportunityProduct": (
            top_opportunity_product
        ),
        **forecast_outlook,
    }

    return {
        "product_matrix": product_matrix,
        "recommendations": recommendations,
        "forecast_outlook": pd.DataFrame(
            [forecast_outlook]
        ),
        "executive_summary": executive_summary,
    }