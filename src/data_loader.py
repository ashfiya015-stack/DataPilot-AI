import pandas as pd


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_excel(file_path):
    return pd.read_excel(file_path)