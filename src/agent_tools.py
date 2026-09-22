import pandas as pd


# ============================================================
# DATA CLEANING TOOLS
# ============================================================

def remove_duplicates_tool(df):
    """Remove exact duplicate rows."""
    before = len(df)
    cleaned_df = df.drop_duplicates().copy()
    removed = before - len(cleaned_df)
    return cleaned_df, removed


def get_missing_values_tool(df):
    """Return missing-value counts for every column."""
    missing = df.isna().sum()
    return {str(c): int(v) for c, v in missing.items()}


def fill_missing_with_median_tool(df, column):
    """Fill missing values in a numerical column using its median."""
    if column not in df.columns:
        return df, {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, {"error": f"Column '{column}' must be numerical."}

    missing_before = int(df[column].isna().sum())
    if missing_before == 0:
        return df, {
            "success": True, "column": column, "filled_values": 0,
            "median": None, "remaining_missing": 0,
            "message": f"No missing values found in '{column}'."
        }

    median_value = df[column].median()
    if pd.isna(median_value):
        return df, {"error": f"Cannot calculate a median for '{column}'."}

    updated_df = df.copy()
    updated_df[column] = updated_df[column].fillna(median_value)

    return updated_df, {
        "success": True,
        "column": column,
        "filled_values": missing_before,
        "median": float(median_value),
        "remaining_missing": int(updated_df[column].isna().sum()),
    }


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info_tool(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "numerical_columns": [str(c) for c in df.select_dtypes(include="number").columns],
        "categorical_columns": [
            str(c) for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        ],
    }


def get_column_statistics_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    series = df[column]
    result = {
        "column": str(column),
        "data_type": str(series.dtype),
        "count": int(series.count()),
        "missing_values": int(series.isna().sum()),
        "unique_values": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        result.update({
            "mean": None if pd.isna(series.mean()) else float(series.mean()),
            "median": None if pd.isna(series.median()) else float(series.median()),
            "minimum": None if pd.isna(series.min()) else float(series.min()),
            "maximum": None if pd.isna(series.max()) else float(series.max()),
        })
    else:
        result["top_values"] = series.value_counts(dropna=False).head(10).to_dict()

    return result


# ============================================================
# NUMERICAL TOOLS
# ============================================================

def calculate_average_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numerical."}
    value = df[column].mean()
    if pd.isna(value):
        return {"error": f"No valid numerical values found in '{column}'."}
    return {"column": column, "average": float(value)}


def calculate_median_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numerical."}
    value = df[column].median()
    if pd.isna(value):
        return {"error": f"No valid numerical values found in '{column}'."}
    return {"column": column, "median": float(value)}


def calculate_maximum_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numerical."}
    value = df[column].max()
    if pd.isna(value):
        return {"error": f"No valid numerical values found in '{column}'."}
    return {"column": column, "maximum": float(value)}


def calculate_minimum_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numerical."}
    value = df[column].min()
    if pd.isna(value):
        return {"error": f"No valid numerical values found in '{column}'."}
    return {"column": column, "minimum": float(value)}


# ============================================================
# GROUPED ANALYSIS
# ============================================================

def calculate_grouped_average_tool(df, group_column, value_column):
    if group_column not in df.columns:
        return {"error": f"Column '{group_column}' does not exist."}
    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"error": f"Column '{value_column}' must be numerical."}

    result = (
        df.groupby(group_column, dropna=True)[value_column]
        .mean()
        .sort_values(ascending=False)
    )
    return {str(group): float(value) for group, value in result.items()}


# ============================================================
# VISUALIZATION TOOLS
# ============================================================

def create_histogram_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numerical for a histogram."}
    return {"success": True, "chart_type": "histogram", "column": column}


def create_bar_chart_tool(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}
    if not (
        pd.api.types.is_object_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[column])
    ):
        return {"error": f"Column '{column}' should be categorical for a bar chart."}
    return {"success": True, "chart_type": "bar", "column": column}


def create_scatter_plot_tool(df, x_column, y_column):
    if x_column not in df.columns:
        return {"error": f"Column '{x_column}' does not exist."}
    if y_column not in df.columns:
        return {"error": f"Column '{y_column}' does not exist."}
    if not pd.api.types.is_numeric_dtype(df[x_column]):
        return {"error": f"Column '{x_column}' must be numerical."}
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return {"error": f"Column '{y_column}' must be numerical."}
    return {
        "success": True,
        "chart_type": "scatter",
        "x_column": x_column,
        "y_column": y_column,
    }


def create_correlation_heatmap_tool(df):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) < 2:
        return {"error": "At least two numerical columns are required for a correlation heatmap."}
    return {
        "success": True,
        "chart_type": "correlation_heatmap",
        "numerical_columns": [str(c) for c in numeric_columns],
    }


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlation_tool(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"error": "At least two numerical columns are required for correlation analysis."}

    correlation = numeric_df.corr()
    return {
        "success": True,
        "correlation": correlation.round(4).to_dict(),
        "numerical_columns": [str(c) for c in numeric_df.columns],
    }
