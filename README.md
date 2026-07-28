# AI Business Insight System

Prototyp zaawansowanej aplikacji analitycznej przygotowanej na potrzeby pracy magisterskiej pt.:

**„Wykorzystanie sztucznej inteligencji do analizy dużych zbiorów danych tekstowych i sprzedażowych jako wsparcie procesów decyzyjnych przedsiębiorstwa”**

Celem projektu jest integracja analizy danych sprzedażowych, opinii klientów, metod uczenia maszynowego oraz modeli przetwarzania języka naturalnego w celu wspierania procesów decyzyjnych przedsiębiorstwa.

## Główne funkcjonalności

Aplikacja obejmuje:

- wczytywanie i walidację danych sprzedażowych oraz tekstowych,
- czyszczenie i przygotowanie danych,
- analizę kluczowych wskaźników sprzedażowych,
- analizę sprzedaży według czasu, produktów i krajów,
- segmentację klientów metodą RFM,
- analizę ocen i sentymentu opinii,
- klasyfikację sentymentu za pomocą TF-IDF i modeli uczenia maszynowego,
- porównanie modeli z zastosowaniem walidacji krzyżowej,
- analizę interpretowalności modelu,
- automatyczne wykrywanie tematów opinii metodą NMF,
- transformerową analizę sentymentu przy użyciu modelu BERT,
- prognozowanie sprzedaży z walidacją kroczącą,
- porównanie modeli prognostycznych,
- wykrywanie zmian i driftu danych,
- generowanie rekomendacji biznesowych,
- zaawansowane centrum wspomagania decyzji,
- interaktywne filtrowanie wyników,
- eksport tabel i wyników analiz do plików CSV.
- automatyczną kontrolę kompletności, poprawności i spójności danych,

## Technologie

Projekt wykorzystuje między innymi:

- Python 3.11,
- Streamlit,
- Pandas,
- NumPy,
- Plotly,
- Scikit-learn,
- Statsmodels,
- PyTorch,
- Hugging Face Transformers,
- Pytest,
- Pytest-cov.

## Struktura projektu

```text
ai_business_insight_system/
├── app.py
├── requirements.txt
├── README.md
├── pytest.ini
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── outputs/
│   ├── charts/
│   ├── tables/
│   └── screenshots/
├── scripts/
│   └── generate_sample_data.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sales_analysis.py
│   ├── sales_forecasting.py
│   ├── text_analysis.py
│   ├── sentiment_models.py
│   ├── model_comparison.py
│   ├── model_interpretability.py
│   ├── transformer_sentiment.py
│   ├── topic_modeling.py
│   ├── advanced_recommendations.py
│   ├── drift_monitoring.py
│   ├── recommendations.py
│   ├── filters.py
│   └── visualizations.py
│   ├── data_quality.py
└── tests/
    ├── conftest.py
    ├── test_data_quality.py
    ├── test_text_analytics.py
    ├── test_sales_forecasting.py
    ├── test_decision_support.py
    └── test_drift_monitoring.py
```

## Instalacja

W głównym katalogu projektu utwórz środowisko wirtualne:

```powershell
python -m venv .venv
```

Aktywuj środowisko:

```powershell
.venv\Scripts\Activate.ps1
```

Zainstaluj wymagane biblioteki:

```powershell
python -m pip install -r requirements.txt
```

Sprawdź poprawność zależności:

```powershell
python -m pip check
```

## Uruchomienie aplikacji

Po aktywowaniu środowiska wirtualnego uruchom:

```powershell
streamlit run app.py
```

Aplikacja zostanie udostępniona lokalnie, domyślnie pod adresem:

```text
http://localhost:8501
```

## Dane przykładowe

Projekt zawiera syntetyczne dane testowe:

```text
data/sample/generated_sales.csv
data/sample/generated_reviews.csv
```

Można je wygenerować ponownie za pomocą polecenia:

```powershell
python scripts/generate_sample_data.py
```

Dane syntetyczne służą do testowania działania aplikacji. Wyniki uzyskane na ich podstawie nie powinny być traktowane jako rzeczywiste wyniki biznesowe.

## Model transformerowy

Moduł transformerowy wykorzystuje model:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

Przy pierwszym uruchomieniu model zostanie pobrany z Hugging Face. Wymagane jest wtedy połączenie z Internetem oraz wolne miejsce na dysku.

Kolejne uruchomienia korzystają z lokalnie zapisanych plików modelu.

## Testy automatyczne

Projekt zawiera testy jednostkowe i integracyjne najważniejszych modułów analitycznych.

Uruchomienie wszystkich testów:

```powershell
python -m pytest
```

Uruchomienie testów z raportem pokrycia kodu:

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

Testy obejmują:

- porównanie modeli klasyfikacji sentymentu,
- interpretowalność modelu,
- modelowanie tematów NMF,
- prognozowanie sprzedaży,
- zaawansowane centrum wspomagania decyzji,
- monitoring stabilności i driftu danych.
- kontrolę jakości danych i wykrywanie nieprawidłowych rekordów,

Model transformerowy jest testowany osobno w aplikacji, ponieważ jego uruchomienie wymaga pobrania zewnętrznych zasobów i większej mocy obliczeniowej.

## Wersjonowanie

Projekt jest rozwijany z wykorzystaniem systemu Git.

Stabilna wersja analityczna została oznaczona tagiem:

```text
v2.0
```

Rozwój zaawansowanej wersji odbywa się na gałęzi:

```text
v3-development
```

## Charakter projektu

Aplikacja stanowi prototyp systemu wspomagania decyzji. Generowane rekomendacje i wskaźniki mają charakter pomocniczy i nie zastępują oceny analityka lub menedżera.

Ostateczna interpretacja wyników powinna uwzględniać:

- jakość i kompletność danych,
- specyfikę przedsiębiorstwa,
- kontekst rynkowy,
- ograniczenia modeli analitycznych,
- możliwość występowania błędów i driftu danych.