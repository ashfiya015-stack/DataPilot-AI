def get_basic_info(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum())
    }


def get_column_types(df):
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns
    }


def get_statistics(df):
    return df.describe(include="all")