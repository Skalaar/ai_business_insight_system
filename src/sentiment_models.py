import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_tfidf_logistic_regression_model(data: pd.DataFrame) -> dict:
    """
    Trenuje model klasyfikacji sentymentu opinii klientów.

    Model wykorzystuje:
    - TF-IDF do reprezentacji tekstu,
    - Logistic Regression do klasyfikacji sentymentu.

    Etykietą uczącą jest RatingSentiment, czyli sentyment wyznaczony
    na podstawie oceny gwiazdkowej.

    Zwraca słownik zawierający:
    - wytrenowany model,
    - dokładność,
    - raport klasyfikacji,
    - macierz pomyłek,
    - dane testowe z predykcjami.
    """
    required_columns = ["CleanReviewText", "RatingSentiment"]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn do trenowania modelu: {missing_columns}")

    model_data = data.dropna(subset=["CleanReviewText", "RatingSentiment"]).copy()
    model_data = model_data[model_data["CleanReviewText"].str.len() > 0]

    if len(model_data) < 10:
        raise ValueError(
            "Zbyt mało opinii do trenowania modelu. Wymagane jest co najmniej 10 rekordów."
        )

    class_counts = model_data["RatingSentiment"].value_counts()
    available_classes = class_counts[class_counts >= 2].index.tolist()

    model_data = model_data[model_data["RatingSentiment"].isin(available_classes)]

    if model_data["RatingSentiment"].nunique() < 2:
        raise ValueError(
            "Do trenowania modelu wymagane są co najmniej dwie klasy sentymentu."
        )

    X = model_data["CleanReviewText"]
    y = model_data["RatingSentiment"]

    stratify_values = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify_values,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    labels = sorted(model_data["RatingSentiment"].unique().tolist())

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"Rzeczywiste: {label}" for label in labels],
        columns=[f"Predykcja: {label}" for label in labels],
    )

    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "Class"})

    predictions_df = pd.DataFrame(
        {
            "ReviewText": X_test.values,
            "ActualSentiment": y_test.values,
            "PredictedSentiment": y_pred,
        }
    )

    predictions_df["CorrectPrediction"] = (
        predictions_df["ActualSentiment"] == predictions_df["PredictedSentiment"]
    )

    return {
        "model": pipeline,
        "accuracy": accuracy,
        "classification_report": report_df,
        "confusion_matrix": confusion_df,
        "predictions": predictions_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": labels,
    }


def predict_sentiment_for_reviews(data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Dodaje do danych opinii kolumnę z sentymentem przewidzianym przez model ML.
    """
    predicted_data = data.copy()

    predicted_data["MLSentiment"] = model.predict(predicted_data["CleanReviewText"])

    predicted_data["SentimentAgreement"] = (
        predicted_data["RatingSentiment"] == predicted_data["MLSentiment"]
    )

    return predicted_data


def ml_sentiment_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca rozkład sentymentu przewidzianego przez model ML.
    """
    result = (
        data.groupby("MLSentiment", as_index=False)
        .agg(Reviews=("ReviewID", "count"))
        .sort_values("Reviews", ascending=False)
    )

    return result


def sentiment_comparison_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Porównuje sentyment wynikający z oceny gwiazdkowej z sentymentem modelu ML.
    """
    result = (
        data.groupby(["RatingSentiment", "MLSentiment"], as_index=False)
        .agg(Reviews=("ReviewID", "count"))
        .sort_values("Reviews", ascending=False)
    )

    return result