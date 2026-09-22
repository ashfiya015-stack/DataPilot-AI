import plotly.express as px


def create_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    return fig


def create_bar_chart(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]

    fig = px.bar(
        counts,
        x=column,
        y="count",
        title=f"{column} Distribution"
    )

    return fig


def create_scatter_plot(df, x_column, y_column):
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{x_column} vs {y_column}"
    )

    return fig


def create_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        title="Correlation Heatmap"
    )

    return fig