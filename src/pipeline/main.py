from pipeline.extract import extract_from_csv


def main():
    """
    This function extracts data from a CSV file located in the 'data/input' directory and prints the resulting DataFrame.
    """
    lista_df = extract_from_csv("data/input")
    print(lista_df)


if __name__ == "__main__":
    main()
