import streamlit as st
import pandas as pd
from io import StringIO

from src.data_loader import load_csv, load_excel
from src.data_profiler import (
    get_basic_info,
    get_column_types,
    get_statistics,
)
from src.data_cleaner import (
    remove_duplicates,
    get_missing_values,
)
from src.data_analyzer import (
    analyze_numeric_data,
    analyze_categorical_data,
    calculate_correlation,
)
from src.visualizer import (
    create_histogram,
    create_bar_chart,
    create_scatter_plot,
    create_correlation_heatmap,
)
from src.ai_agent import ask_ai, run_autonomous_analysis


# ============================================================
# HELPER: NORMALIZE MISSING VALUES
# ============================================================

def _normalize_missing_values(result):
    """
    Normalize get_missing_values() output to pandas Series.
    
    Handles:
    - pandas.Series (older format)
    - dict with column names as keys (current format)
    - dict with "missing_values" key containing inner dict
    - Unexpected types (strings, None, etc.)
    - Empty results
    
    Returns: pandas.Series with column names and missing counts
    """
    
    try:
        # If already a pandas Series, return it as-is
        if isinstance(result, pd.Series):
            return result.copy() if hasattr(result, 'copy') else result
        
        # If it's a dictionary, convert to Series
        if isinstance(result, dict):
            # Case 1: Direct dict of {column: count}
            if result and all(isinstance(v, (int, float)) for v in result.values()):
                return pd.Series({
                    str(column): int(count)
                    for column, count in result.items()
                }, dtype="int64")
            
            # Case 2: Nested dict with "missing_values" key
            if "missing_values" in result:
                values = result["missing_values"]
                if isinstance(values, dict):
                    return pd.Series({
                        str(column): int(count)
                        for column, count in values.items()
                    }, dtype="int64")
        
        # If it's a string or other unexpected type, log and return empty
        if isinstance(result, str):
            # Silently ignore string returns (e.g., error messages)
            return pd.Series(dtype="int64")
        
        # For anything else (None, unexpected types), return empty Series
        return pd.Series(dtype="int64")
    
    except Exception as e:
        # If anything goes wrong, return empty Series gracefully
        return pd.Series(dtype="int64")


# ============================================================
# AUTONOMOUS ANALYSIS ENGINE
# ============================================================

def build_complete_analysis(df):
    """
    Build a complete, deterministic analysis from the current
    dataframe. The AI is used afterward to turn these verified
    results into a natural-language report.
    """
    report = {}

    # ---------- basic profile ----------
    info = get_basic_info(df)
    column_types = get_column_types(df)

    report["overview"] = {
        "rows": info["rows"],
        "columns": info["columns"],
        "duplicates": info["duplicates"],
        "missing_values": info["missing_values"],
        "numerical_columns": column_types["numerical"],
        "categorical_columns": column_types["categorical"],
    }

    # ---------- data quality ----------
    try:
        missing = _normalize_missing_values(get_missing_values(df))
    except Exception as e:
        # If get_missing_values fails, calculate manually
        missing = pd.Series({col: int(df[col].isna().sum()) for col in df.columns}, dtype="int64")
    
    missing_items = []
    for col, count in missing.items():
        if int(count) > 0:
            pct = round((int(count) / max(len(df), 1)) * 100, 1)
            missing_items.append({
                "column": str(col),
                "count": int(count),
                "percentage": pct,
            })

    report["data_quality"] = {
        "missing": missing_items,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # ---------- numerical analysis ----------
    numerical = {}
    for col in column_types["numerical"]:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        numerical[col] = {
            "count": int(series.count()),
            "missing": int(df[col].isna().sum()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "minimum": round(float(series.min()), 4),
            "maximum": round(float(series.max()), 4),
            "range": round(float(series.max() - series.min()), 4),
        }

    report["numerical_analysis"] = numerical

    # ---------- categorical analysis ----------
    categorical = {}
    for col in column_types["categorical"]:
        counts = df[col].value_counts(dropna=False)
        values = []
        for value, count in counts.head(10).items():
            values.append({
                "value": "Missing" if pd.isna(value) else str(value),
                "count": int(count),
            })
        categorical[col] = values

    report["categorical_analysis"] = categorical

    # ---------- grouped insights ----------
    grouped = {}
    for group_col in column_types["categorical"]:
        if df[group_col].nunique(dropna=True) > 1:
            for value_col in column_types["numerical"]:
                if df[value_col].notna().sum() > 0:
                    grouped[f"{group_col} → {value_col}"] = (
                        df.groupby(group_col, dropna=True)[value_col]
                        .mean()
                        .sort_values(ascending=False)
                        .round(4)
                        .to_dict()
                    )

    report["grouped_analysis"] = grouped

    # ---------- correlations ----------
    correlation = calculate_correlation(df)
    if correlation is not None:
        report["correlation"] = correlation.round(4).to_dict()
    else:
        report["correlation"] = None

    # ---------- automatic chart plan ----------
    charts = []

    if column_types["numerical"]:
        # Show distributions for up to 3 numerical columns.
        for col in column_types["numerical"][:3]:
            charts.append({
                "type": "histogram",
                "column": col,
            })

    if column_types["categorical"]:
        # Show the first categorical distribution.
        charts.append({
            "type": "bar",
            "column": column_types["categorical"][0],
        })

    if len(column_types["numerical"]) >= 2:
        charts.append({
            "type": "scatter",
            "x_column": column_types["numerical"][0],
            "y_column": column_types["numerical"][1],
        })
        charts.append({
            "type": "correlation_heatmap",
        })

    report["recommended_charts"] = charts

    return report


def make_ai_analysis_prompt(report):
    """Create a compact verified context for the AI narrative."""
    return f"""
Perform a complete professional analysis of the uploaded dataset.

IMPORTANT:
- The numbers below were calculated directly from the dataframe.
- Use these verified results; do not invent or alter statistics.
- Explain important patterns, data-quality issues, grouped differences,
  and correlations when present.
- Clearly distinguish observations from conclusions.
- If the dataset is very small, explicitly warn that findings are
  directional and should not be generalized.
- Do not claim causation from correlation.
- Organize the answer with these headings:
  1. Executive Summary
  2. Data Quality
  3. Numerical Insights
  4. Categorical / Group Insights
  5. Relationship Insights
  6. Most Important Takeaways

VERIFIED DATA:
{report}
"""


def render_complete_analysis(df, report):
    """Render the verified report and automatically selected charts."""

    st.markdown("## 🔎 1. Executive Overview")

    overview = report["overview"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", overview["rows"])
    with c2:
        st.metric("Columns", overview["columns"])
    with c3:
        st.metric("Missing Values", overview["missing_values"])
    with c4:
        st.metric("Duplicate Rows", overview["duplicates"])

    st.markdown("## 🧹 2. Data Quality")

    missing_items = report["data_quality"]["missing"]
    if missing_items:
        st.warning("Missing values were detected.")
        st.dataframe(
            pd.DataFrame(missing_items),
            width="stretch",
        )
    else:
        st.success("No missing values found.")

    if report["data_quality"]["duplicate_rows"] > 0:
        st.warning(
            f'{report["data_quality"]["duplicate_rows"]} duplicate row(s) detected.'
        )
    else:
        st.success("No duplicate rows found.")

    st.markdown("## 📈 3. Numerical Insights")

    numerical = report["numerical_analysis"]
    if numerical:
        rows = []
        for col, values in numerical.items():
            rows.append({
                "Column": col,
                "Mean": values["mean"],
                "Median": values["median"],
                "Minimum": values["minimum"],
                "Maximum": values["maximum"],
                "Range": values["range"],
                "Missing": values["missing"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No numerical columns are available.")

    st.markdown("## 📊 4. Categorical / Group Insights")

    grouped = report["grouped_analysis"]
    if grouped:
        for title, values in grouped.items():
            st.write(f"**{title}**")
            group_rows = [
                {"Group": str(k), "Average": round(float(v), 4)}
                for k, v in values.items()
            ]
            st.dataframe(
                pd.DataFrame(group_rows),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("No useful grouped analysis was available.")

    st.markdown("## 🔗 5. Relationship Insights")

    correlation = report["correlation"]
    if correlation is not None:
        st.dataframe(
            pd.DataFrame(correlation),
            width="stretch",
        )
    else:
        st.info(
            "At least two numerical columns are required for correlation analysis."
        )

    st.markdown("## 📊 6. Automatically Selected Visualizations")

    for chart in report["recommended_charts"]:
        chart_type = chart["type"]

        if chart_type == "histogram":
            col = chart["column"]
            st.subheader(f"Distribution of {col}")
            st.plotly_chart(
                create_histogram(df, col),
                width="stretch",
                key=f"render_histogram_{col}",
            )

        elif chart_type == "bar":
            col = chart["column"]
            st.subheader(f"Distribution of {col}")
            st.plotly_chart(
                create_bar_chart(df, col),
                width="stretch",
                key=f"render_bar_{col}",
            )

        elif chart_type == "scatter":
            x_col = chart["x_column"]
            y_col = chart["y_column"]
            st.subheader(f"{x_col} vs {y_col}")
            st.plotly_chart(
                create_scatter_plot(df, x_col, y_col),
                width="stretch",
                key=f"render_scatter_{x_col}_{y_col}",
            )

        elif chart_type == "correlation_heatmap":
            st.subheader("Correlation Heatmap")
            st.plotly_chart(
                create_correlation_heatmap(df),
                width="stretch",
                key="render_correlation_heatmap",
            )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DataPilot AI")
st.subheader("Intelligent Data Analysis Dashboard")
st.write(
    "Upload your CSV or Excel dataset and explore, clean, "
    "analyze, and visualize your data."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 DataPilot AI")
st.sidebar.info(
    """
DataPilot AI helps you:

• Upload datasets
• Understand your data
• Find missing values
• Detect duplicates
• Clean datasets
• Analyze numerical data
• Analyze categorical data
• Create visualizations
• Explore correlations
• Ask questions using natural language
"""
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:
        # Detect a new upload and reset the dataframe.
        upload_id = f"{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"

        if st.session_state.get("upload_id") != upload_id:
            if uploaded_file.name.lower().endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            st.session_state.upload_id = upload_id

        df = st.session_state.df

        st.success(f"✅ Successfully loaded: {uploaded_file.name}")


        # ====================================================
        # DATA OVERVIEW
        # ====================================================

        st.header("📊 Dataset Overview")

        info = get_basic_info(df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", info["rows"])

        with col2:
            st.metric("Columns", info["columns"])

        with col3:
            st.metric("Missing Values", info["missing_values"])

        with col4:
            st.metric("Duplicate Rows", info["duplicates"])


        # ====================================================
        # DATA PREVIEW
        # ====================================================

        st.header("👀 Data Preview")
        st.dataframe(df.head(100), width="stretch")


        # ====================================================
        # COLUMN INFORMATION
        # ====================================================

        st.header("📋 Column Information")

        column_types = get_column_types(df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔢 Numerical Columns")

            if column_types["numerical"]:
                for column in column_types["numerical"]:
                    st.write(f"• {column}")
            else:
                st.write("No numerical columns found.")

        with col2:
            st.subheader("🔤 Categorical Columns")

            if column_types["categorical"]:
                for column in column_types["categorical"]:
                    st.write(f"• {column}")
            else:
                st.write("No categorical columns found.")


        # ====================================================
        # MISSING VALUES
        # ====================================================

        st.header("📊 Missing Values")

        try:
            missing_values = _normalize_missing_values(get_missing_values(df))
        except Exception as e:
            # Fallback: calculate missing values manually
            missing_values = pd.Series({col: int(df[col].isna().sum()) for col in df.columns}, dtype="int64")

        missing_table = missing_values[missing_values > 0].sort_values(ascending=False)

        if missing_table.empty:
            st.success("✅ No missing values found.")
        else:
            missing_df = missing_table.reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            st.dataframe(missing_df, width="stretch")


        # ====================================================
        # DATA CLEANING
        # ====================================================

        st.header("🧹 Data Cleaning")

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count > 0:
            st.warning(
                f"Found {duplicate_count} duplicate row(s)."
            )

            if st.button("🗑️ Remove Duplicate Rows", key="manual_remove_duplicates"):
                cleaned_df, removed = remove_duplicates(df)
                st.session_state.df = cleaned_df

                st.success(
                    f"✅ Removed {removed} duplicate row(s)."
                )

                st.rerun()
        else:
            st.success("✅ No duplicate rows found.")


        # ====================================================
        # STATISTICAL SUMMARY
        # ====================================================

        st.header("📊 Statistical Summary")

        stats = get_statistics(df)

        st.dataframe(stats, width="stretch")


        # ====================================================
        # DETAILED ANALYSIS
        # ====================================================

        st.header("🔍 Detailed Analysis")

        tab1, tab2, tab3 = st.tabs(
            ["Numerical Analysis", "Categorical Analysis", "Correlations"]
        )

        with tab1:
            st.subheader("📈 Numerical Analysis")

            try:
                numerical_results = analyze_numeric_data(df)
                st.dataframe(numerical_results, width="stretch")
            except Exception as e:
                st.info(
                    "No numerical columns available for analysis."
                )

        with tab2:
            st.subheader("📊 Categorical Analysis")

            try:
                categorical_results = analyze_categorical_data(df)

                for column, result in categorical_results.items():
                    st.write(f"**{column}**")
                    st.dataframe(result, width="stretch")
            except Exception as e:
                st.info(
                    "No categorical columns available for analysis."
                )

        with tab3:
            st.subheader("🔗 Correlation Analysis")

            try:
                correlation = calculate_correlation(df)

                if correlation is not None:
                    st.dataframe(correlation, width="stretch")
                else:
                    st.info(
                        "At least two numerical columns are required "
                        "for correlation analysis."
                    )
            except Exception as e:
                st.info("Correlation analysis not available.")


        # ====================================================
        # VISUALIZATIONS
        # ====================================================

        st.header("📊 Visualizations")

        column_types = get_column_types(df)

        # -------- Histograms --------

        if column_types["numerical"]:
            st.subheader("📊 Histograms")

            num_columns = column_types["numerical"]

            hist_col = st.selectbox(
                "Select a numerical column for histogram:",
                num_columns,
                key="histogram_column_select",
            )

            if hist_col:
                try:
                    fig = create_histogram(df, hist_col)
                    st.plotly_chart(fig, width="stretch", key=f"manual_histogram_{hist_col}")
                except Exception as e:
                    st.error(f"Error creating histogram: {e}")

        # -------- Bar Charts --------

        if column_types["categorical"]:
            st.subheader("📊 Bar Charts")

            cat_columns = column_types["categorical"]

            bar_col = st.selectbox(
                "Select a categorical column for bar chart:",
                cat_columns,
                key="bar_column_select",
            )

            if bar_col:
                try:
                    fig = create_bar_chart(df, bar_col)
                    st.plotly_chart(fig, width="stretch", key=f"manual_bar_{bar_col}")
                except Exception as e:
                    st.error(f"Error creating bar chart: {e}")

        # -------- Scatter Plots --------

        if len(column_types["numerical"]) >= 2:
            st.subheader("📈 Scatter Plot")

            num_columns = column_types["numerical"]

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox(
                    "Select X axis:",
                    num_columns,
                    key="scatter_x_select",
                )

            with col2:
                y_col = st.selectbox(
                    "Select Y axis:",
                    num_columns,
                    index=1 if len(num_columns) > 1 else 0,
                    key="scatter_y_select",
                )

            if x_col and y_col:
                try:
                    fig = create_scatter_plot(df, x_col, y_col)
                    st.plotly_chart(fig, width="stretch", key=f"manual_scatter_{x_col}_{y_col}")
                except Exception as e:
                    st.error(f"Error creating scatter plot: {e}")

        # -------- Correlation Heatmap --------

        if len(column_types["numerical"]) >= 2:
            st.subheader("🔗 Correlation Heatmap")

            try:
                fig = create_correlation_heatmap(df)
                st.plotly_chart(fig, width="stretch", key="manual_correlation_heatmap")
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {e}")


        # ====================================================
        # AUTONOMOUS ANALYSIS
        # ====================================================

        st.divider()

        st.header("🚀 Autonomous Analysis")

        st.write(
            "Click below to run a complete automated analysis "
            "of your dataset. DataPilot AI will:"
        )

        st.write(
            """
- Profile your data
- Detect data quality issues
- Analyze numerical and categorical patterns
- Identify relationships
- Generate visualizations
- Produce an AI-generated report
"""
        )

        if st.button("🚀 Run Autonomous Analysis", key="autonomous_button"):

            with st.spinner("🤖 DataPilot AI is analyzing your dataset..."):

                try:

                    # Call run_autonomous_analysis with the dataframe
                    # It returns: (answer, updated_df, charts, actions)
                    (
                        autonomous_answer,
                        updated_df,
                        autonomous_charts,
                        autonomous_actions,
                    ) = run_autonomous_analysis(st.session_state.df)

                    # Update the session state dataframe
                    st.session_state.df = updated_df

                    # Store the analysis results
                    st.session_state.autonomous_answer = autonomous_answer
                    st.session_state.autonomous_actions = autonomous_actions

                    # Build the report for display
                    report = build_complete_analysis(updated_df)
                    st.session_state.autonomous_report = report

                    st.success("✅ Autonomous analysis complete!")

                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error (OpenAI 429)
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        st.warning(
                            "⚠️ OpenAI Rate Limit Reached\n\n"
                            "Your API quota is temporarily exhausted. "
                            "Showing analysis from calculated data instead..."
                        )
                        
                        # Generate fallback analysis from the data
                        try:
                            df = st.session_state.df
                            report = build_complete_analysis(df)
                            
                            # Create a text summary from the report
                            summary_parts = []
                            summary_parts.append("📊 DATASET ANALYSIS SUMMARY (Local Mode)")
                            summary_parts.append("=" * 60)
                            
                            overview = report["overview"]
                            summary_parts.append(f"\n📋 OVERVIEW")
                            summary_parts.append(f"  • Rows: {overview['rows']}")
                            summary_parts.append(f"  • Columns: {overview['columns']}")
                            summary_parts.append(f"  • Duplicate Rows: {overview['duplicates']}")
                            summary_parts.append(f"  • Total Missing Values: {overview['missing_values']}")
                            
                            summary_parts.append(f"\n📊 COLUMN TYPES")
                            summary_parts.append(f"  • Numerical: {', '.join(overview['numerical_columns']) if overview['numerical_columns'] else 'None'}")
                            summary_parts.append(f"  • Categorical: {', '.join(overview['categorical_columns']) if overview['categorical_columns'] else 'None'}")
                            
                            data_quality = report["data_quality"]
                            summary_parts.append(f"\n🧹 DATA QUALITY")
                            if data_quality["missing"]:
                                summary_parts.append(f"  Missing Values Detected:")
                                for item in data_quality["missing"]:
                                    summary_parts.append(f"    - {item['column']}: {item['count']} ({item['percentage']}%)")
                            else:
                                summary_parts.append(f"  ✅ No missing values found")
                            
                            if data_quality["duplicate_rows"] > 0:
                                summary_parts.append(f"  ⚠️ Duplicate Rows: {data_quality['duplicate_rows']}")
                            else:
                                summary_parts.append(f"  ✅ No duplicate rows found")
                            
                            numerical = report["numerical_analysis"]
                            if numerical:
                                summary_parts.append(f"\n📈 NUMERICAL INSIGHTS")
                                for col, stats in numerical.items():
                                    summary_parts.append(f"  {col}:")
                                    summary_parts.append(f"    - Mean: {stats['mean']}")
                                    summary_parts.append(f"    - Median: {stats['median']}")
                                    summary_parts.append(f"    - Min: {stats['minimum']}, Max: {stats['maximum']}")
                                    summary_parts.append(f"    - Missing: {stats['missing']}")
                            
                            grouped = report["grouped_analysis"]
                            if grouped:
                                summary_parts.append(f"\n👥 GROUPED ANALYSIS")
                                for title, values in grouped.items():
                                    summary_parts.append(f"  {title}:")
                                    for group, avg in values.items():
                                        summary_parts.append(f"    - {group}: {avg}")
                            
                            if report["correlation"] is not None:
                                summary_parts.append(f"\n🔗 CORRELATION ANALYSIS")
                                summary_parts.append(f"  Correlation matrix calculated for numerical columns")
                            
                            summary_parts.append(f"\n✨ Analysis generated from your dataset (no AI required)")
                            
                            autonomous_answer = "\n".join(summary_parts)
                            
                            # Store the results
                            st.session_state.autonomous_answer = autonomous_answer
                            st.session_state.autonomous_report = report
                            st.session_state.autonomous_actions = ["Ran autonomous analysis (local mode - no API)"]
                            
                            st.success("✅ Analysis complete (using local data)")
                        
                        except Exception as fallback_error:
                            st.error(f"❌ Could not generate fallback analysis: {fallback_error}")
                    else:
                        # Other errors
                        st.error(
                            f"❌ Error during autonomous analysis: {e}"
                        )

        # ---------- Display autonomous results ----------

        if st.session_state.get("autonomous_answer"):

            st.markdown("---")

            st.subheader("🤖 AI Analysis Report")

            st.write(st.session_state.get("autonomous_answer"))

            st.markdown("---")

            # Render the full analysis dashboard
            report = st.session_state.get("autonomous_report")

            if report:
                render_complete_analysis(df, report)

            # ---------- Professional Report Export ----------

            st.markdown("---")

            st.subheader("📄 Professional Report Export")

            if st.session_state.get("autonomous_answer"):

                report_parts = []

                report_parts.append("DATAPILOT AI - ANALYSIS REPORT")
                report_parts.append("=" * 50)

                report_parts.append("\nOVERVIEW")
                overview = st.session_state.get("autonomous_report", {}).get("overview", {})
                report_parts.append(f"Rows: {overview.get('rows', 'N/A')}")
                report_parts.append(f"Columns: {overview.get('columns', 'N/A')}")
                report_parts.append(f"Missing Values: {overview.get('missing_values', 'N/A')}")
                report_parts.append(f"Duplicate Rows: {overview.get('duplicates', 'N/A')}")

                # Calculate final missing values
                final_df = st.session_state.df
                report_parts.append("\nDATASET STATUS")
                report_parts.append(
                    f"Missing values remaining: {int(final_df.isna().sum().sum())}. "
                    f"Duplicate rows remaining: {int(final_df.duplicated().sum())}."
                )

                report_parts.append("\nAGENT ACTIVITY")
                for i, action in enumerate(st.session_state.get("autonomous_actions", []), 1):
                    report_parts.append(f"{i}. {action}")

                report_parts.append("\nAI ANALYSIS")
                report_parts.append(st.session_state.get("autonomous_answer", "No AI report generated."))

                report_text = "\n".join(report_parts)
                st.text_area(
                    "📄 Analysis Report",
                    report_text,
                    height=420,
                    key="professional_report_preview",
                )

                export_cols = st.columns(2)

                with export_cols[0]:
                    csv_buffer = StringIO()
                    final_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download Cleaned Dataset",
                        data=csv_buffer.getvalue(),
                        file_name="datapilot_cleaned_dataset.csv",
                        mime="text/csv",
                        key="download_cleaned_dataset",
                        width="stretch",
                    )

                with export_cols[1]:
                    st.download_button(
                        "📄 Download Analysis Report",
                        data=report_text,
                        file_name="datapilot_analysis_report.txt",
                        mime="text/plain",
                        key="download_analysis_report",
                        width="stretch",
                    )

        # ----------------------------------------------------
        # FINAL DATASET STATUS
        # ----------------------------------------------------

        if st.session_state.get("autonomous_answer"):
            st.markdown("---")
            st.subheader("📌 Dataset After Autonomous Processing")

            final_df = st.session_state.df
            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric("Rows", len(final_df))

            with metric_cols[1]:
                st.metric("Columns", len(final_df.columns))

            with metric_cols[2]:
                st.metric("Missing Values", int(final_df.isna().sum().sum()))

            with metric_cols[3]:
                st.metric("Duplicate Rows", int(final_df.duplicated().sum()))


# ====================================================
        # AI ASSISTANT
        # ====================================================

        st.divider()

        st.header("🤖 Ask DataPilot AI")

        st.write(
            "Ask questions about your uploaded dataset "
            "using natural language."
        )

        user_question = st.text_input(
            "💬 Ask something about your dataset:",
            placeholder=(
                "Example: Create a correlation heatmap."
            ),
            key="ai_question",
        )

        if st.button(
            "🚀 Ask DataPilot AI",
            key="ask_ai_button",
        ):

            if not user_question.strip():

                st.info(
                    "Please enter a question before asking DataPilot AI."
                )

            else:

                try:

                    with st.spinner(
                        "🤖 DataPilot AI is analyzing your dataset..."
                    ):

                        answer, updated_df, chart_info = ask_ai(
                            user_question,
                            st.session_state.df,
                        )

                    st.session_state.df = updated_df

                    st.success("🤖 DataPilot AI")
                    st.write(answer)


                    # ----------------------------------------
                    # AI-GENERATED CHART
                    # ----------------------------------------

                    if chart_info:

                        chart_type = chart_info.get(
                            "chart_type"
                        )

                        if chart_type == "histogram":

                            column = chart_info["column"]

                            if column in updated_df.columns:

                                fig = create_histogram(
                                    updated_df,
                                    column,
                                )

                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key=f"ai_histogram_{column}",
                                )


                        elif chart_type == "bar":

                            column = chart_info["column"]

                            if column in updated_df.columns:

                                fig = create_bar_chart(
                                    updated_df,
                                    column,
                                )

                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key=f"ai_bar_{column}",
                                )


                        elif chart_type == "scatter":

                            x_column = chart_info["x_column"]
                            y_column = chart_info["y_column"]

                            if (
                                x_column in updated_df.columns
                                and y_column in updated_df.columns
                            ):

                                fig = create_scatter_plot(
                                    updated_df,
                                    x_column,
                                    y_column,
                                )

                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key=f"ai_scatter_{x_column}_{y_column}",
                                )


                        elif chart_type == "correlation_heatmap":

                            numerical_columns = (
                                updated_df.select_dtypes(
                                    include="number"
                                ).columns
                            )

                            if len(numerical_columns) >= 2:

                                fig = create_correlation_heatmap(
                                    updated_df
                                )

                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key="ai_correlation_heatmap",
                                )


                except Exception as e:

                    st.error(
                        f"❌ Error while answering your "
                        f"dataset question: {e}"
                    )


    except Exception as e:

        st.error(
            f"❌ Error while processing the dataset: {e}"
        )


else:

    st.info(
        "📁 Upload a CSV or Excel file to start analyzing your data."
    )


# ============================================================
# FOOTER / CAPABILITIES
# ============================================================

st.divider()

st.markdown(
    """
### 🚀 What DataPilot AI can do

**1. Upload Data**  
CSV and Excel datasets.

**2. Understand Data**  
Rows, columns, data types, preview and dataset information.

**3. Clean Data**  
Detect/remove duplicates and handle missing numerical values.

**4. Analyze Data**  
Average, median, minimum, maximum, grouped averages,
comparisons, recommendations and correlation analysis.

**5. Visualize Data**  
Histograms, bar charts, scatter plots and correlation heatmaps.

**6. AI Agent**  
Ask questions about your dataset using natural language.
DataPilot AI selects the appropriate analysis tool automatically.
"""
)

st.caption(
    "📊 DataPilot AI | Intelligent Data Analysis Platform"
)
