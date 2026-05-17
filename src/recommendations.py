import pandas as pd


def generate_sales_recommendations(
    kpis: dict,
    top_products_data: pd.DataFrame,
    country_data: pd.DataFrame,
    rfm_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generuje rekomendacje biznesowe na podstawie wyników analizy sprzedaży i RFM.

    Parametry:
        kpis: słownik z podstawowymi wskaźnikami sprzedażowymi,
        top_products_data: ranking produktów według sprzedaży,
        country_data: sprzedaż według kraju,
        rfm_data: wyniki segmentacji RFM.

    Zwraca:
        DataFrame z rekomendacjami biznesowymi.
    """
    recommendations = []

    total_revenue = kpis.get("total_revenue", 0)
    total_transactions = kpis.get("total_transactions", 0)
    total_customers = kpis.get("total_customers", 0)
    average_order_value = kpis.get("average_order_value", 0)

    if total_revenue > 0:
        recommendations.append(
            {
                "Obszar": "Sprzedaż",
                "Priorytet": "Wysoki",
                "Wniosek": f"Łączna wartość sprzedaży w analizowanym zbiorze wynosi {total_revenue:,.2f}.",
                "Rekomendacja": "Należy monitorować dynamikę sprzedaży w czasie oraz identyfikować okresy zwiększonego popytu.",
            }
        )

    if total_transactions > 0 and average_order_value > 0:
        recommendations.append(
            {
                "Obszar": "Koszyk zakupowy",
                "Priorytet": "Średni",
                "Wniosek": f"Średnia wartość zamówienia wynosi {average_order_value:,.2f}.",
                "Rekomendacja": "Warto rozważyć działania zwiększające wartość koszyka, np. sprzedaż pakietową, cross-selling lub progi darmowej dostawy.",
            }
        )

    if total_customers > 0 and total_transactions > 0:
        transactions_per_customer = total_transactions / total_customers

        if transactions_per_customer < 2:
            priority = "Wysoki"
            recommendation = "Niska liczba transakcji na klienta wskazuje na potrzebę działań retencyjnych i aktywizujących ponowne zakupy."
        else:
            priority = "Średni"
            recommendation = "Warto rozwijać działania zwiększające częstotliwość zakupów oraz wzmacniające lojalność klientów."

        recommendations.append(
            {
                "Obszar": "Aktywność klientów",
                "Priorytet": priority,
                "Wniosek": f"Średnia liczba transakcji na klienta wynosi {transactions_per_customer:.2f}.",
                "Rekomendacja": recommendation,
            }
        )

    if not top_products_data.empty:
        best_product = top_products_data.iloc[0]
        product_name = best_product["Description"]
        product_revenue = best_product["Revenue"]

        recommendations.append(
            {
                "Obszar": "Produkty",
                "Priorytet": "Wysoki",
                "Wniosek": f"Najwyższą wartość sprzedaży osiąga produkt: {product_name} ({product_revenue:,.2f}).",
                "Rekomendacja": "Należy utrzymać dostępność tego produktu oraz rozważyć jego wykorzystanie w kampaniach promocyjnych lub sprzedaży produktów powiązanych.",
            }
        )

        if len(top_products_data) >= 3:
            top_3_revenue = top_products_data.head(3)["Revenue"].sum()
            top_products_share = top_3_revenue / total_revenue if total_revenue > 0 else 0

            if top_products_share > 0.5:
                recommendations.append(
                    {
                        "Obszar": "Koncentracja sprzedaży",
                        "Priorytet": "Średni",
                        "Wniosek": f"Trzy najlepiej sprzedające się produkty generują {top_products_share:.1%} sprzedaży.",
                        "Rekomendacja": "Wysoka koncentracja sprzedaży na kilku produktach może zwiększać ryzyko biznesowe. Warto analizować możliwość dywersyfikacji oferty.",
                    }
                )

    if not country_data.empty:
        best_country = country_data.iloc[0]
        country_name = best_country["Country"]
        country_revenue = best_country["Revenue"]

        recommendations.append(
            {
                "Obszar": "Rynek geograficzny",
                "Priorytet": "Średni",
                "Wniosek": f"Największą wartość sprzedaży wygenerował rynek: {country_name} ({country_revenue:,.2f}).",
                "Rekomendacja": "Warto przeanalizować potencjał dalszego rozwoju tego rynku oraz porównać go z krajami o niższej sprzedaży.",
            }
        )

        if total_revenue > 0:
            country_share = country_revenue / total_revenue

            if country_share > 0.7:
                recommendations.append(
                    {
                        "Obszar": "Ryzyko rynkowe",
                        "Priorytet": "Wysoki",
                        "Wniosek": f"Rynek {country_name} odpowiada za {country_share:.1%} całkowitej sprzedaży.",
                        "Rekomendacja": "Silna zależność od jednego rynku może oznaczać ryzyko. Warto rozważyć działania zwiększające udział innych rynków.",
                    }
                )

    if not rfm_data.empty:
        segment_summary = (
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

        if not segment_summary.empty:
            most_valuable_segment = segment_summary.iloc[0]

            recommendations.append(
                {
                    "Obszar": "Segmentacja klientów",
                    "Priorytet": "Wysoki",
                    "Wniosek": (
                        f"Największą wartość zakupów generuje segment: "
                        f"{most_valuable_segment['Segment']} "
                        f"({most_valuable_segment['TotalValue']:,.2f})."
                    ),
                    "Rekomendacja": "Dla tego segmentu warto przygotować dedykowane działania utrzymaniowe lub program lojalnościowy.",
                }
            )

        risky_segments = rfm_data[
            rfm_data["Segment"].isin(
                [
                    "Klienci zagrożeni odejściem",
                    "Wartościowi nieaktywni klienci",
                ]
            )
        ]

        if not risky_segments.empty:
            risky_value = risky_segments["Monetary"].sum()
            risky_customers = risky_segments["CustomerID"].nunique()

            recommendations.append(
                {
                    "Obszar": "Retencja klientów",
                    "Priorytet": "Wysoki",
                    "Wniosek": (
                        f"W segmentach wymagających działań retencyjnych znajduje się "
                        f"{risky_customers} klientów o łącznej wartości zakupów {risky_value:,.2f}."
                    ),
                    "Rekomendacja": "Rekomendowane jest przygotowanie kampanii reaktywacyjnej, np. indywidualnego rabatu, przypomnienia mailowego lub oferty specjalnej.",
                }
            )

        best_customers = rfm_data[rfm_data["Segment"] == "Najlepsi klienci"]

        if not best_customers.empty:
            best_customers_count = best_customers["CustomerID"].nunique()
            best_customers_value = best_customers["Monetary"].sum()

            recommendations.append(
                {
                    "Obszar": "Najlepsi klienci",
                    "Priorytet": "Wysoki",
                    "Wniosek": (
                        f"Segment najlepszych klientów obejmuje {best_customers_count} klientów "
                        f"o łącznej wartości zakupów {best_customers_value:,.2f}."
                    ),
                    "Rekomendacja": "Warto chronić tę grupę klientów poprzez działania VIP, wcześniejszy dostęp do ofert oraz komunikację personalizowaną.",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "Obszar": "Brak danych",
                "Priorytet": "Niski",
                "Wniosek": "Nie wygenerowano rekomendacji, ponieważ zakres danych jest niewystarczający.",
                "Rekomendacja": "Należy wczytać pełniejszy zbiór danych sprzedażowych zawierający informacje o transakcjach, klientach i produktach.",
            }
        )

    return pd.DataFrame(recommendations)

def generate_combined_product_recommendations(
    top_products_data: pd.DataFrame,
    product_reviews_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generuje rekomendacje na podstawie połączenia wyników analizy sprzedaży
    oraz analizy opinii klientów.

    Funkcja porównuje:
    - wartość sprzedaży produktów,
    - liczbę opinii,
    - średnią ocenę,
    - udział opinii pozytywnych i negatywnych.

    Zwraca:
        DataFrame z rekomendacjami produktowymi.
    """
    recommendations = []

    if top_products_data.empty or product_reviews_data.empty:
        return pd.DataFrame(
            [
                {
                    "Produkt": "Brak danych",
                    "Obszar": "Dane",
                    "Priorytet": "Niski",
                    "Wniosek": "Brakuje danych sprzedażowych lub tekstowych potrzebnych do wygenerowania rekomendacji produktowych.",
                    "Rekomendacja": "Należy wczytać dane sprzedażowe oraz dane tekstowe zawierające opinie klientów dla produktów.",
                }
            ]
        )

    sales_products = top_products_data.copy()
    reviews_products = product_reviews_data.copy()

    merged_data = sales_products.merge(
        reviews_products,
        left_on="Description",
        right_on="ProductName",
        how="inner",
    )

    if merged_data.empty:
        return pd.DataFrame(
            [
                {
                    "Produkt": "Brak dopasowania",
                    "Obszar": "Integracja danych",
                    "Priorytet": "Średni",
                    "Wniosek": "Nie znaleziono wspólnych produktów między danymi sprzedażowymi i opiniami klientów.",
                    "Rekomendacja": "Należy zweryfikować spójność nazw produktów lub zastosować wspólny identyfikator produktu, np. ProductID/StockCode.",
                }
            ]
        )

    revenue_median = merged_data["Revenue"].median()
    rating_median = merged_data["AverageRating"].median()

    for _, row in merged_data.iterrows():
        product_name = row["Description"]
        revenue = row["Revenue"]
        average_rating = row["AverageRating"]
        reviews = row["Reviews"]
        negative_share = row["NegativeShare"]
        positive_share = row["PositiveShare"]

        if revenue >= revenue_median and negative_share >= 0.30:
            recommendations.append(
                {
                    "Produkt": product_name,
                    "Obszar": "Jakość produktu",
                    "Priorytet": "Wysoki",
                    "Wniosek": (
                        f"Produkt osiąga relatywnie wysoką sprzedaż ({revenue:,.2f}), "
                        f"ale udział opinii negatywnych wynosi {negative_share:.1%}."
                    ),
                    "Rekomendacja": "Rekomendowana jest kontrola jakości produktu oraz analiza treści negatywnych opinii klientów.",
                }
            )

        if revenue < revenue_median and average_rating >= rating_median and positive_share >= 0.70:
            recommendations.append(
                {
                    "Produkt": product_name,
                    "Obszar": "Potencjał sprzedażowy",
                    "Priorytet": "Średni",
                    "Wniosek": (
                        f"Produkt ma relatywnie niższą sprzedaż ({revenue:,.2f}), "
                        f"ale wysoką średnią ocenę ({average_rating:.2f}) i udział opinii pozytywnych {positive_share:.1%}."
                    ),
                    "Rekomendacja": "Warto rozważyć zwiększenie widoczności produktu, kampanię promocyjną lub rekomendowanie go klientom w sklepie internetowym.",
                }
            )

        if reviews >= 2 and average_rating <= 2.5:
            recommendations.append(
                {
                    "Produkt": product_name,
                    "Obszar": "Ryzyko niezadowolenia klientów",
                    "Priorytet": "Wysoki",
                    "Wniosek": (
                        f"Produkt posiada niską średnią ocenę ({average_rating:.2f}) "
                        f"na podstawie {reviews} opinii."
                    ),
                    "Rekomendacja": "Należy przeanalizować przyczyny niskiej oceny oraz rozważyć poprawę produktu, opisu oferty lub procesu obsługi klienta.",
                }
            )

        if revenue >= revenue_median and positive_share >= 0.80:
            recommendations.append(
                {
                    "Produkt": product_name,
                    "Obszar": "Produkt strategiczny",
                    "Priorytet": "Średni",
                    "Wniosek": (
                        f"Produkt łączy wysoką sprzedaż ({revenue:,.2f}) "
                        f"z wysokim udziałem opinii pozytywnych ({positive_share:.1%})."
                    ),
                    "Rekomendacja": "Produkt może zostać wykorzystany jako element kampanii sprzedażowej lub jako punkt odniesienia dla rozwoju podobnych produktów.",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "Produkt": "Ogół produktów",
                "Obszar": "Analiza łączona",
                "Priorytet": "Niski",
                "Wniosek": "Nie wykryto istotnych zależności wymagających natychmiastowej reakcji decyzyjnej.",
                "Rekomendacja": "Należy kontynuować monitorowanie sprzedaży i opinii klientów oraz przeprowadzić analizę na większym zbiorze danych.",
            }
        )

    return pd.DataFrame(recommendations)