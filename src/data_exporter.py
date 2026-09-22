def get_missing_values(df):
    return df.isnull().sum()