from __future__ import annotations

import pandas as pd

from src.sentiment_models import (
    train_tfidf_logistic_regression_model,
)
from src.text_analysis import (
    normalize_text,
    top_review_words,
)
from src.text_resources import (
    get_stopwords,
    normalize_language_code,
)


def test_portuguese_stopwords() -> None:
    stopwords = set(
        get_stopwords(
            language="pt-br",
            include_domain=True,
        )
    )

    assert "que" in stopwords
    assert "não" in stopwords
    assert "com" in stopwords
    assert "produto" in stopwords

    model_stopwords = set(
        get_stopwords(
            language="pt-br",
            include_domain=True,
            preserve_negations=True,
        )
    )

    assert "não" not in model_stopwords
    assert "nem" not in model_stopwords
    assert "sem" not in model_stopwords

    assert "que" in model_stopwords
    assert "com" in model_stopwords
    assert "produto" in model_stopwords


def test_language_code_normalization() -> None:
    assert (
        normalize_language_code("en")
        == "english"
    )

    assert (
        normalize_language_code("pt_BR")
        == "portuguese"
    )

    assert (
        normalize_language_code(None)
        == "multilingual"
    )


def test_normalize_text_preserves_accents() -> None:
    result = normalize_text(
        "Ótimo produto, não chegou em 2 dias!"
    )

    assert result == (
        "ótimo produto não chegou em dias"
    )


def test_top_words_remove_stopwords() -> None:
    data = pd.DataFrame(
        {
            "CleanReviewText": [
                (
                    "produto muito bom "
                    "entrega rápida"
                ),
                (
                    "que produto excelente"
                ),
            ]
        }
    )

    result = top_review_words(
        data=data,
        limit=10,
    )

    words = set(
        result["Word"].tolist()
    )

    assert "produto" not in words
    assert "muito" not in words
    assert "que" not in words

    assert "bom" in words
    assert "entrega" in words
    assert "rápida" in words
    assert "excelente" in words

def test_baseline_model_uses_portuguese_stopwords() -> None:
    data = pd.DataFrame(
        {
            "CleanReviewText": [
                "produto muito bom entrega rápida",
                "excelente qualidade recomendo",
                "chegou antes prazo ótimo",
                "bom atendimento compra perfeita",
                "entrega rápida ótima qualidade",
                "recomendo loja excelente",
                "produto regular entrega normal",
                "qualidade aceitável",
                "compra comum sem destaque",
                "entrega dentro prazo esperado",
                "experiência razoável",
                "avaliação neutra compra",
                "produto ruim não chegou",
                "entrega atrasada péssima",
                "qualidade muito baixa",
                "não recomendo loja",
                "produto quebrado atraso",
                "experiência horrível compra",
            ],
            "RatingSentiment": [
                "Pozytywny",
                "Pozytywny",
                "Pozytywny",
                "Pozytywny",
                "Pozytywny",
                "Pozytywny",
                "Neutralny",
                "Neutralny",
                "Neutralny",
                "Neutralny",
                "Neutralny",
                "Neutralny",
                "Negatywny",
                "Negatywny",
                "Negatywny",
                "Negatywny",
                "Negatywny",
                "Negatywny",
            ],
        }
    )

    results = (
        train_tfidf_logistic_regression_model(
            data=data,
            text_language="pt-br",
            random_state=42,
        )
    )

    vectorizer = results[
        "model"
    ].named_steps[
        "tfidf"
    ]

    feature_names = set(
        vectorizer.get_feature_names_out()
    )

    assert results["text_language"] == (
        "portuguese"
    )

    assert "produto" not in feature_names
    assert "muito" not in feature_names
    assert "não" in feature_names

    assert "entrega" in feature_names
    assert "qualidade" in feature_names
    assert results["number_of_features"] > 0