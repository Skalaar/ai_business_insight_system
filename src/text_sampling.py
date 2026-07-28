from __future__ import annotations

import numpy as np
import pandas as pd


def _calculate_stratified_allocations(
    class_counts: pd.Series,
    sample_size: int,
) -> dict[str, int]:
    """
    Wyznacza liczbę rekordów pobieranych z każdej klasy.

    Alokacja:
    - zachowuje przybliżone proporcje klas,
    - gwarantuje co najmniej jeden rekord z każdej klasy,
    - nie przekracza liczebności dostępnych klas,
    - daje dokładnie oczekiwaną wielkość próbki.
    """
    number_of_classes = len(class_counts)

    if sample_size < number_of_classes:
        raise ValueError(
            "Wielkość próbki musi być co najmniej równa "
            "liczbie klas."
        )

    proportions = (
        class_counts
        / class_counts.sum()
    )

    exact_allocations = (
        proportions
        * sample_size
    )

    allocations = (
        np.floor(
            exact_allocations
        )
        .astype(int)
    )

    allocations = allocations.clip(
        lower=1
    )

    allocations = pd.concat(
        [
            allocations.rename(
                "Allocation"
            ),
            class_counts.rename(
                "Available"
            ),
            (
                exact_allocations
                - np.floor(
                    exact_allocations
                )
            ).rename(
                "Fraction"
            ),
        ],
        axis=1,
    )

    allocations["Allocation"] = np.minimum(
        allocations["Allocation"],
        allocations["Available"],
    )

    current_total = int(
        allocations["Allocation"].sum()
    )

    while current_total < sample_size:
        candidates = allocations[
            allocations["Allocation"]
            < allocations["Available"]
        ].copy()

        if candidates.empty:
            break

        selected_class = (
            candidates
            .sort_values(
                by=[
                    "Fraction",
                    "Available",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .index[0]
        )

        allocations.loc[
            selected_class,
            "Allocation",
        ] += 1

        allocations.loc[
            selected_class,
            "Fraction",
        ] = 0.0

        current_total += 1

    while current_total > sample_size:
        candidates = allocations[
            allocations["Allocation"] > 1
        ].copy()

        if candidates.empty:
            break

        selected_class = (
            candidates
            .sort_values(
                by=[
                    "Fraction",
                    "Allocation",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .index[0]
        )

        allocations.loc[
            selected_class,
            "Allocation",
        ] -= 1

        current_total -= 1

    if current_total != sample_size:
        raise ValueError(
            "Nie udało się utworzyć próbki "
            "o wymaganej liczbie rekordów."
        )

    return {
        str(class_name): int(allocation)
        for class_name, allocation
        in allocations["Allocation"].items()
    }


def create_stratified_text_sample(
    data: pd.DataFrame,
    max_samples: int | None,
    target_column: str = "RatingSentiment",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Tworzy deterministyczną próbkę danych tekstowych.

    Jeżeli liczba rekordów nie przekracza max_samples,
    zwracany jest cały zbiór.

    Gdy kolumna target_column jest dostępna, próbka
    zachowuje przybliżone proporcje klas. W przeciwnym
    razie wykonywane jest zwykłe losowanie.

    Parametry:
        data:
            Dane wejściowe.
        max_samples:
            Maksymalna liczba rekordów w próbce.
            Wartość None oznacza wykorzystanie
            całego zbioru.
        target_column:
            Kolumna wykorzystywana do stratyfikacji.
        random_state:
            Ziarno zapewniające powtarzalność.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Parametr data musi być obiektem DataFrame."
        )

    if data.empty:
        return data.copy().reset_index(
            drop=True
        )

    if max_samples is None:
        return data.copy().reset_index(
            drop=True
        )

    if max_samples < 2:
        raise ValueError(
            "Maksymalna wielkość próbki musi "
            "wynosić co najmniej 2."
        )

    if len(data) <= max_samples:
        return data.copy().reset_index(
            drop=True
        )

    if target_column not in data.columns:
        return (
            data
            .sample(
                n=max_samples,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    working_data = data.dropna(
        subset=[target_column]
    ).copy()

    if working_data.empty:
        raise ValueError(
            "Brak rekordów z poprawną wartością "
            f"w kolumnie {target_column}."
        )

    if len(working_data) <= max_samples:
        return working_data.reset_index(
            drop=True
        )

    class_counts = (
        working_data[
            target_column
        ]
        .astype(str)
        .value_counts()
    )

    allocations = (
        _calculate_stratified_allocations(
            class_counts=class_counts,
            sample_size=max_samples,
        )
    )

    sampled_parts = []

    for class_offset, (
        class_name,
        allocation,
    ) in enumerate(
        allocations.items()
    ):
        class_data = working_data[
            working_data[
                target_column
            ].astype(str)
            == class_name
        ]

        sampled_part = class_data.sample(
            n=allocation,
            random_state=(
                random_state
                + class_offset
            ),
        )

        sampled_parts.append(
            sampled_part
        )

    sampled_data = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    sampled_data = sampled_data.sample(
        frac=1,
        random_state=random_state,
    )

    return sampled_data.reset_index(
        drop=True
    )