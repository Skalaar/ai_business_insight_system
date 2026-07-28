from __future__ import annotations


ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "very",
        "after",
        "again",
        "than",
        "too",
    }
)


PORTUGUESE_STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "aquela",
        "aquelas",
        "aquele",
        "aqueles",
        "aquilo",
        "as",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "delas",
        "dele",
        "deles",
        "depois",
        "do",
        "dos",
        "e",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "eram",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estas",
        "este",
        "estes",
        "eu",
        "foi",
        "foram",
        "há",
        "isso",
        "isto",
        "já",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesma",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "muito",
        "muita",
        "muitos",
        "muitas",
        "na",
        "nas",
        "não",
        "nem",
        "no",
        "nos",
        "nós",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "porque",
        "qual",
        "quando",
        "que",
        "quem",
        "se",
        "sem",
        "ser",
        "seu",
        "seus",
        "sua",
        "suas",
        "são",
        "também",
        "tem",
        "tinha",
        "um",
        "uma",
        "umas",
        "uns",
        "você",
        "vocês",
    }
)


DOMAIN_STOPWORDS = frozenset(
    {
        "product",
        "products",
        "produto",
        "produtos",
    }
)

NEGATION_WORDS = frozenset(
    {
        "no",
        "nor",
        "not",
        "never",
        "não",
        "nao",
        "nem",
        "sem",
    }
)

LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "pt": "portuguese",
    "pt-br": "portuguese",
    "pt_br": "portuguese",
    "por": "portuguese",
    "portuguese": "portuguese",
    "multilingual": "multilingual",
    "multi": "multilingual",
    "all": "multilingual",
}


def normalize_language_code(
    language: str | None,
) -> str:
    """
    Ujednolica oznaczenie języka wykorzystywane
    przez moduły analizy tekstowej.
    """
    if language is None:
        return "multilingual"

    normalized = (
        str(language)
        .strip()
        .lower()
        .replace("_", "-")
    )

    if normalized not in LANGUAGE_ALIASES:
        supported = ", ".join(
            sorted(
                {
                    "english",
                    "portuguese",
                    "multilingual",
                }
            )
        )

        raise ValueError(
            "Nieobsługiwany język tekstu: "
            f"{language}. Obsługiwane wartości: "
            f"{supported}."
        )

    return LANGUAGE_ALIASES[normalized]


def get_stopwords(
    language: str | None = "multilingual",
    include_domain: bool = True,
    preserve_negations: bool = False,
) -> list[str]:
    """
    Zwraca uporządkowaną listę stopwords odpowiednią
    dla wskazanego języka.
    """
    normalized_language = normalize_language_code(
        language
    )

    if normalized_language == "english":
        stopwords = set(
            ENGLISH_STOPWORDS
        )

    elif normalized_language == "portuguese":
        stopwords = set(
            PORTUGUESE_STOPWORDS
        )

    else:
        stopwords = (
            set(ENGLISH_STOPWORDS)
            | set(PORTUGUESE_STOPWORDS)
        )

    if include_domain:
        stopwords.update(
            DOMAIN_STOPWORDS
        )

    if preserve_negations:
        stopwords.difference_update(
            NEGATION_WORDS
        )

    return sorted(stopwords)