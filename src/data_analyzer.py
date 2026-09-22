def analyze_numeric_data(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return "No numerical columns found."

    return numeric_df.describe()


def analyze_categorical_data(df):
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    results = {}

    for column in categorical_columns:
        results[column] = df[column].value_counts()

    return results


def calculate_correlation(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    return numeric_df.corr()