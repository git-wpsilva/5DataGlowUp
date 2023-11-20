import pandas as pd

from pipeline.extract import load_airbnb_data, transform_data
from pipeline.transform import get_exchange_rate


def save_dataframe_to_csv(df, file_path, encoding="utf-8"):
    df.to_csv(file_path, encoding=encoding, index=False)


def load_exchange_rates():
    exchange_rates = pd.DataFrame(
        {
            "country": [
                "France",
                "United States",
                "Australia",
                "Italy",
                "Brazil",
                "Turkey",
                "Mexico",
                "Thailand",
                "South Africa",
                "Hong Kong",
            ],
        }
    )

    exchange_rates["exchange_rate"] = exchange_rates.apply(get_exchange_rate, axis=1)

    return exchange_rates


def load_and_transform_data(data_file, rev_file, exchange_rates):
    df = load_airbnb_data(data_file)
    df_rev = load_airbnb_data(rev_file)

    transformed_df = transform_data(df, df_rev, exchange_rates)

    return transformed_df
