from src.data_loader import load_csv
from src.data_profiler import (
    get_basic_info,
    get_column_types,
    get_statistics
)
from src.data_cleaner import (
    remove_duplicates,
    get_missing_values
)


# Load dataset
df = load_csv("data/sample.csv")


# Basic information
print("\n========== DATASET INFO ==========")
print(get_basic_info(df))


# Missing values
print("\n========== MISSING VALUES ==========")
print(get_missing_values(df))


# Column types
print("\n========== COLUMN TYPES ==========")

column_types = get_column_types(df)

print("\nNumerical columns:")
print(column_types["numerical"])

print("\nCategorical columns:")
print(column_types["categorical"])


# Statistics
print("\n========== STATISTICS ==========")
print(get_statistics(df))


# Remove duplicates
df, removed = remove_duplicates(df)

print("\n========== CLEANING ==========")
print("Duplicates removed:", removed)


# Final dataset
print("\n========== CLEANED DATA ==========")
print(df)

from src.data_analyzer import (
    analyze_numeric_data,
    analyze_categorical_data,
    calculate_correlation
)


print("\n========== NUMERICAL ANALYSIS ==========")
print(analyze_numeric_data(df))


print("\n========== CATEGORICAL ANALYSIS ==========")

categorical_results = analyze_categorical_data(df)

for column, result in categorical_results.items():
    print(f"\n{column}:")
    print(result)


print("\n========== CORRELATION ==========")
print(calculate_correlation(df))

from src.visualizer import (
    create_histogram,
    create_bar_chart,
    create_scatter_plot,
    create_correlation_heatmap
)
print("\n========== VISUALIZATION ==========")

fig1 = create_histogram(df, "Age")
fig1.show()

fig2 = create_bar_chart(df, "Department")
fig2.show()

fig3 = create_scatter_plot(df, "Age", "Salary")
fig3.show()

fig4 = create_correlation_heatmap(df)
fig4.show()