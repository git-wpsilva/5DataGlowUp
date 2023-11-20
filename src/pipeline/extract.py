"""Arquivo de extração de dados."""

import os  # biblioteca para manipular arquivos e pastas
import glob  # Biblioteca para listar arquivos de um diretorio

import pandas as pd
from typing import List


path = "data/input"


def extract_from_csv(path: str) -> List[pd.DataFrame]:
    """
    Função para extrair os dados de vários arquivos csv
    Extraindo dentro de uma pasta, data/input e retornar uma lista de dataframes

    args: input_path: caminho para a pasta com os arquivos csv

    return: lista de dataframes
    """

    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, sep=";")
        df_list.append(df)
    return df_list


if __name__ == "__main__":
    df_list = extract_from_csv(path)
    print(df_list)
