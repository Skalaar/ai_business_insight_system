from pathlib import Path
import random

import numpy as np
import pandas as pd


RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PRODUCTS = [
    {
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "UnitPrice": 2.55,
        "Quality": "high",
    },
    {
        "StockCode": "71053",
        "Description": "WHITE METAL LANTERN",
        "UnitPrice": 3.39,
        "Quality": "high",
    },
    {
        "StockCode": "84406B",
        "Description": "CREAM CUPID HEARTS COAT HANGER",
        "UnitPrice": 2.75,
        "Quality": "medium",
    },
    {
        "StockCode": "84029G",
        "Description": "KNITTED UNION FLAG HOT WATER BOTTLE",
        "UnitPrice": 3.39,
        "Quality": "low",
    },
    {
        "StockCode": "84029E",
        "Description": "RED WOOLLY HOTTIE WHITE HEART",
        "UnitPrice": 3.39,
        "Quality": "low",
    },
    {
        "StockCode": "22752",
        "Description": "SET 7 BABUSHKA NESTING BOXES",
        "UnitPrice": 7.65,
        "Quality": "high",
    },
    {
        "StockCode": "21730",
        "Description": "GLASS STAR FROSTED T-LIGHT HOLDER",
        "UnitPrice": 4.25,
        "Quality": "medium",
    },
    {
        "StockCode": "22633",
        "Description": "HAND WARMER UNION JACK",
        "UnitPrice": 1.85,
        "Quality": "high",
    },
    {
        "StockCode": "22632",
        "Description": "HAND WARMER RED POLKA DOT",
        "UnitPrice": 1.85,
        "Quality": "medium",
    },
    {
        "StockCode": "22086",
        "Description": "PAPER CHAIN KIT 50'S CHRISTMAS",
        "UnitPrice": 2.55,
        "Quality": "high",
    },
    {
        "StockCode": "21258",
        "Description": "VICTORIAN SEWING BOX LARGE",
        "UnitPrice": 10.95,
        "Quality": "medium",
    },
    {
        "StockCode": "22114",
        "Description": "HOT WATER BOTTLE TEA AND SYMPATHY",
        "UnitPrice": 3.45,
        "Quality": "low",
    },
    {
        "StockCode": "22386",
        "Description": "JUMBO BAG PINK POLKADOT",
        "UnitPrice": 1.95,
        "Quality": "high",
    },
    {
        "StockCode": "85099C",
        "Description": "JUMBO BAG BAROQUE BLACK WHITE",
        "UnitPrice": 1.95,
        "Quality": "medium",
    },
    {
        "StockCode": "22961",
        "Description": "JAM MAKING SET PRINTED",
        "UnitPrice": 1.45,
        "Quality": "high",
    },
]


COUNTRIES = [
    "United Kingdom",
    "France",
    "Germany",
    "Netherlands",
    "Spain",
    "Belgium",
    "Norway",
    "Portugal",
    "Italy",
]


POSITIVE_REVIEWS = [
    "Excellent product, very good quality and beautiful design.",
    "I am satisfied with this purchase, although delivery could be faster.",
    "Great value for money and the product works well.",
    "Very useful product, solid and nicely made.",
    "Beautiful item, exactly as described. I would recommend it.",
    "The product quality is good, but the packaging was average.",
    "I would buy this product again because it is reliable and practical.",
    "Nice design and good quality for the price.",
    "The item looks elegant and performs as expected.",
    "Good product overall, even if it is not perfect.",
    "I like the product, but the material could feel a little stronger.",
    "The product met my expectations and arrived in acceptable condition.",
    "Useful and practical item. It works better than I expected.",
    "Good quality, but the price is slightly higher than I would like.",
    "Overall positive experience with this product.",
]


NEUTRAL_REVIEWS = [
    "Average product, acceptable but nothing special.",
    "The product is okay, but I expected a little more.",
    "It works as described, but the quality could be better.",
    "Neutral experience. The item is usable but not impressive.",
    "The product is fine for the price.",
    "Delivery was normal and the product is acceptable.",
    "Some parts are good, but other details feel average.",
    "The product is not bad, but I would not call it excellent.",
    "It is a standard item with no major advantages or disadvantages.",
    "The product works, although the design is not very impressive.",
    "I have mixed feelings about this product.",
    "The item is usable, but I am not fully convinced.",
    "Quality is acceptable, but there are better alternatives.",
    "It does the job, but nothing more.",
    "The purchase was neither very good nor very bad.",
]


NEGATIVE_REVIEWS = [
    "Poor quality. The product broke after first use.",
    "The item looks nice, but the quality is disappointing.",
    "The product is not worth the money.",
    "The material feels cheap and unreliable.",
    "I am disappointed with this purchase.",
    "The product did not meet my expectations.",
    "Bad quality and poor durability.",
    "The product arrived in poor condition.",
    "Delivery was fast, but the product quality was poor.",
    "The design is nice, but the item does not work well.",
    "I expected better quality for this price.",
    "The product has some useful features, but overall I am not satisfied.",
    "It looked good at first, but after using it I noticed several problems.",
    "The item is usable, but I would not buy it again.",
    "The product feels fragile and not very reliable.",
]


def choose_country() -> str:
    return random.choices(
        population=COUNTRIES,
        weights=[70, 8, 7, 4, 3, 2, 2, 2, 2],
        k=1,
    )[0]


def choose_product() -> dict:
    return random.choices(
        population=PRODUCTS,
        weights=[13, 10, 6, 8, 8, 6, 5, 9, 9, 8, 4, 8, 7, 5, 4],
        k=1,
    )[0]


def generate_sales_data(number_of_rows: int = 2500) -> pd.DataFrame:
    rows = []

    start_date = pd.Timestamp("2010-12-01")
    customer_ids = list(range(12000, 12550))

    for row_number in range(number_of_rows):
        product = choose_product()

        invoice_number = 536000 + row_number // random.randint(1, 4)
        invoice_date = start_date + pd.Timedelta(
            days=random.randint(0, 365),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59),
        )

        quantity = random.choices(
            population=[1, 2, 3, 4, 5, 6, 8, 10, 12, 24, 36, 48],
            weights=[4, 6, 5, 5, 5, 9, 5, 4, 4, 2, 1, 1],
            k=1,
        )[0]

        unit_price = round(product["UnitPrice"] * random.uniform(0.95, 1.08), 2)

        rows.append(
            {
                "InvoiceNo": str(invoice_number),
                "StockCode": product["StockCode"],
                "Description": product["Description"],
                "Quantity": quantity,
                "InvoiceDate": invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
                "UnitPrice": unit_price,
                "CustomerID": random.choice(customer_ids),
                "Country": choose_country(),
            }
        )

    return pd.DataFrame(rows)


def choose_rating(product_quality: str) -> int:
    if product_quality == "high":
        return random.choices(
            population=[1, 2, 3, 4, 5],
            weights=[2, 4, 8, 36, 50],
            k=1,
        )[0]

    if product_quality == "medium":
        return random.choices(
            population=[1, 2, 3, 4, 5],
            weights=[6, 12, 28, 34, 20],
            k=1,
        )[0]

    return random.choices(
        population=[1, 2, 3, 4, 5],
        weights=[24, 30, 22, 16, 8],
        k=1,
    )[0]


def choose_review_text(rating: int) -> str:
    """
    Wybiera tekst opinii na podstawie oceny, ale celowo wprowadza pewien poziom
    niejednoznaczności. Dzięki temu dane są bardziej realistyczne, a model ML
    nie osiąga sztucznie idealnych wyników.
    """
    if rating >= 4:
        return random.choices(
            population=POSITIVE_REVIEWS + NEUTRAL_REVIEWS + NEGATIVE_REVIEWS,
            weights=[8] * len(POSITIVE_REVIEWS)
            + [2] * len(NEUTRAL_REVIEWS)
            + [1] * len(NEGATIVE_REVIEWS),
            k=1,
        )[0]

    if rating == 3:
        return random.choices(
            population=POSITIVE_REVIEWS + NEUTRAL_REVIEWS + NEGATIVE_REVIEWS,
            weights=[2] * len(POSITIVE_REVIEWS)
            + [8] * len(NEUTRAL_REVIEWS)
            + [2] * len(NEGATIVE_REVIEWS),
            k=1,
        )[0]

    return random.choices(
        population=POSITIVE_REVIEWS + NEUTRAL_REVIEWS + NEGATIVE_REVIEWS,
        weights=[1] * len(POSITIVE_REVIEWS)
        + [2] * len(NEUTRAL_REVIEWS)
        + [8] * len(NEGATIVE_REVIEWS),
        k=1,
    )[0]


def add_review_noise(review_text: str) -> str:
    """
    Dodaje krótkie uzupełnienia do części opinii, aby zwiększyć różnorodność tekstów.
    """
    additions = [
        "I used it for a few days.",
        "The first impression was different than expected.",
        "For this price, the result is acceptable.",
        "I checked similar products before buying.",
        "The product has both advantages and disadvantages.",
        "This may depend on customer expectations.",
        "The description was mostly accurate.",
        "I noticed this after using the product.",
        "It is hard to compare with other items.",
        "My opinion may change after longer use.",
    ]

    if random.random() < 0.45:
        return f"{review_text} {random.choice(additions)}"

    return review_text


def generate_review_data(number_of_reviews: int = 600) -> pd.DataFrame:
    rows = []

    start_date = pd.Timestamp("2010-12-01")

    for review_id in range(1, number_of_reviews + 1):
        product = choose_product()
        rating = choose_rating(product["Quality"])
        review_text = add_review_noise(choose_review_text(rating))

        review_date = start_date + pd.Timedelta(days=random.randint(0, 365))

        rows.append(
            {
                "ReviewID": review_id,
                "ProductID": product["StockCode"],
                "ProductName": product["Description"],
                "Rating": rating,
                "ReviewDate": review_date.strftime("%Y-%m-%d"),
                "ReviewText": review_text,
            }
        )

    return pd.DataFrame(rows)


def main():
    sales_data = generate_sales_data(number_of_rows=2500)
    review_data = generate_review_data(number_of_reviews=600)

    sales_output_path = OUTPUT_DIR / "generated_sales.csv"
    reviews_output_path = OUTPUT_DIR / "generated_reviews.csv"

    sales_data.to_csv(sales_output_path, index=False, encoding="utf-8-sig")
    review_data.to_csv(reviews_output_path, index=False, encoding="utf-8-sig")

    print(f"Generated sales data: {sales_output_path}")
    print(f"Generated review data: {reviews_output_path}")
    print(f"Sales rows: {len(sales_data)}")
    print(f"Review rows: {len(review_data)}")


if __name__ == "__main__":
    main()