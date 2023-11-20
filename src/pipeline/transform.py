# import pandas as pd from forex_python.converter import CurrencyRates from sklearn.preprocessing import MinMaxScaler


def fix_encoding(problem_string):
    """
    Função para corrigir a codificação de uma string
    """
    if isinstance(
        problem_string, str
    ):  # Checar se é uma string, foi necessário por ter valores nulos
        return problem_string.encode("Windows-1252", errors="ignore").decode(
            "utf-8", errors="ignore"
        )
    else:
        return problem_string


city_to_country = {
    "Paris": "France",
    "New York": "United States",
    "Sydney": "Australia",
    "Rome": "Italy",
    "Rio de Janeiro": "Brazil",
    "Istanbul": "Turkey",
    "Mexico City": "Mexico",
    "Bangkok": "Thailand",
    "Cape Town": "South Africa",
    "Hong Kong": "Hong Kong",
}

country_to_currency = {
    "France": "EUR",
    "United States": "USD",
    "Australia": "AUD",
    "Italy": "EUR",
    "Brazil": "BRL",
    "Turkey": "TRY",
    "Mexico": "MXN",
    "Thailand": "THB",
    "South Africa": "ZAR",
    "Hong Kong": "HKD",
}


def get_country(row):
    return city_to_country.get(row["city"], "Unknown")


def get_exchange_rate(row, currency_rates):
    currency = country_to_currency.get(row["country"])
    if currency == "EUR":
        return 1 / currency_rates.get("USD/EUR")
    return currency_rates.get(f"USD/{currency}")


def fill_na_with_zero(df, columns_to_fill):
    df[columns_to_fill] = df[columns_to_fill].fillna(0)
    return df


def merge_dataframes(left_df, right_df, on_column):
    merged_df = left_df.merge(right_df, on=on_column, how="left")
    return merged_df


def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df


def check_for_duplicates(df, column_to_check):
    duplicates = df[column_to_check].duplicated()
    return duplicates.sum()


def normalize_review_scores(df, columns_to_normalize):
    df[columns_to_normalize] = (df[columns_to_normalize] / 20).round(2)
    return df


def transform_data(df, df_rev, currency_rates):
    rev_columns_to_fill = [
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]

    df = fill_na_with_zero(df, rev_columns_to_fill)

    qty_reviews = df_rev["listing_id"].value_counts().reset_index()
    qty_reviews.columns = ["listing_id", "qty_reviews"]

    merge_df = merge_dataframes(df, qty_reviews, "listing_id")
    merge_df["qty_reviews"] = merge_df["qty_reviews"].fillna(0)

    columns_to_drop = [
        "host_since",
        "host_location",
        "host_response_time",
        "host_response_rate",
        "host_acceptance_rate",
        "host_is_superhost",
        "host_total_listings_count",
        "host_has_profile_pic",
        "host_identity_verified",
        "district",
    ]

    merge_df = drop_columns(merge_df, columns_to_drop)

    duplicate_count = check_for_duplicates(merge_df, "listing_id")
    if duplicate_count > 0:
        print(f"Há {duplicate_count} dados duplicados na coluna 'listing_id'.")
    else:
        print("Não existem dados duplicados na coluna 'listing_id'.")

    duplicate_count = check_for_duplicates(df_rev, "review_id")
    if duplicate_count > 0:
        print(f"Há {duplicate_count} dados duplicados na coluna 'review_id'.")
    else:
        print("Não existem dados duplicados na coluna 'review_id'.")

    review_columns_to_normalize = [
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]

    merge_df = normalize_review_scores(merge_df, review_columns_to_normalize)

    merge_df["country"] = merge_df.apply(get_country, axis=1)

    return merge_df
