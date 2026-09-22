def remove_duplicates(df):
    before = len(df)

    cleaned_df = df.drop_duplicates()

    after = len(cleaned_df)

    removed = before - after

    return cleaned_df, removed


def get_missing_values(df):
    return df.isnull().sum()