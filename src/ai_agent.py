import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from src.agent_tools import (
    remove_duplicates_tool,
    get_missing_values_tool,
    get_dataset_info_tool,
    get_column_statistics_tool,
    calculate_average_tool,
    calculate_median_tool,
    calculate_maximum_tool,
    calculate_minimum_tool,
    fill_missing_with_median_tool,
    calculate_grouped_average_tool,
    create_histogram_tool,
    create_bar_chart_tool,
    create_scatter_plot_tool,
    calculate_correlation_tool,
    create_correlation_heatmap_tool,
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

client = OpenAI(api_key=api_key)


def _schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
        "strict": True,
    }


tools = [
    _schema("get_dataset_info", "Inspect dataset size, columns, types, missing values and duplicates."),
    _schema("get_missing_values", "Inspect missing/null values in every column."),
    _schema(
        "get_column_statistics",
        "Get statistics for one exact column.",
        {"column": {"type": "string", "description": "Exact dataset column name."}},
        ["column"],
    ),
    _schema(
        "calculate_average",
        "Calculate the average of a numerical column.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "calculate_median",
        "Calculate the median of a numerical column.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "calculate_maximum",
        "Find the maximum of a numerical column.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "calculate_minimum",
        "Find the minimum of a numerical column.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "remove_duplicates",
        "Remove exact duplicate rows from the current dataframe.",
    ),
    _schema(
        "fill_missing_with_median",
        "Fill missing values in one numerical column with that column's median.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "calculate_grouped_average",
        "Calculate the mean of a numerical column for each group in a categorical column.",
        {
            "group_column": {"type": "string", "description": "Exact grouping column."},
            "value_column": {"type": "string", "description": "Exact numerical value column."},
        },
        ["group_column", "value_column"],
    ),
    _schema(
        "create_histogram",
        "Create a histogram for a numerical column.",
        {"column": {"type": "string", "description": "Exact numerical column name."}},
        ["column"],
    ),
    _schema(
        "create_bar_chart",
        "Create a category-count bar chart.",
        {"column": {"type": "string", "description": "Exact categorical column name."}},
        ["column"],
    ),
    _schema(
        "create_scatter_plot",
        "Create a scatter plot between two numerical columns.",
        {
            "x_column": {"type": "string", "description": "X-axis numerical column."},
            "y_column": {"type": "string", "description": "Y-axis numerical column."},
        },
        ["x_column", "y_column"],
    ),
    _schema("calculate_correlation", "Calculate the correlation matrix for numerical columns."),
    _schema("create_correlation_heatmap", "Create a correlation heatmap for numerical columns."),
]


def _dataset_context(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "numerical_columns": [str(c) for c in df.select_dtypes(include="number").columns],
        "categorical_columns": [
            str(c) for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        ],
    }


def _execute_tool(name, args, current_df, chart_list, action_log):
    if name == "get_dataset_info":
        result = get_dataset_info_tool(current_df)
    elif name == "get_missing_values":
        result = get_missing_values_tool(current_df)
    elif name == "get_column_statistics":
        result = get_column_statistics_tool(current_df, args["column"])
    elif name == "calculate_average":
        result = calculate_average_tool(current_df, args["column"])
    elif name == "calculate_median":
        result = calculate_median_tool(current_df, args["column"])
    elif name == "calculate_maximum":
        result = calculate_maximum_tool(current_df, args["column"])
    elif name == "calculate_minimum":
        result = calculate_minimum_tool(current_df, args["column"])
    elif name == "remove_duplicates":
        before = len(current_df)
        current_df, removed = remove_duplicates_tool(current_df)
        result = {
            "success": True,
            "action": "remove_duplicates",
            "removed_rows": int(removed),
            "remaining_rows": int(len(current_df)),
        }
        if removed:
            action_log.append(f"Removed {removed} duplicate row(s).")
    elif name == "fill_missing_with_median":
        column = args["column"]
        current_df, result = fill_missing_with_median_tool(current_df, column)
        if isinstance(result, dict) and result.get("success") and result.get("filled_values", 0):
            action_log.append(
                f"Filled {result['filled_values']} missing value(s) in '{column}' using median {result['median']}."
            )
    elif name == "calculate_grouped_average":
        result = calculate_grouped_average_tool(
            current_df, args["group_column"], args["value_column"]
        )
    elif name == "create_histogram":
        result = create_histogram_tool(current_df, args["column"])
        if result.get("success"):
            chart_list.append(result)
    elif name == "create_bar_chart":
        result = create_bar_chart_tool(current_df, args["column"])
        if result.get("success"):
            chart_list.append(result)
    elif name == "create_scatter_plot":
        result = create_scatter_plot_tool(
            current_df, args["x_column"], args["y_column"]
        )
        if result.get("success"):
            chart_list.append(result)
    elif name == "calculate_correlation":
        result = calculate_correlation_tool(current_df)
    elif name == "create_correlation_heatmap":
        result = create_correlation_heatmap_tool(current_df)
        if result.get("success"):
            chart_list.append(result)
    else:
        result = {"error": f"Unknown tool: {name}"}

    return current_df, result


def _run_agent(question, df, autonomous=False):
    current_df = df.copy()
    chart_list = []
    action_log = []

    context = _dataset_context(current_df)

    if autonomous:
        instructions = """
You are DataPilot AI, an autonomous data-analysis agent.

Your job is to inspect the ACTUAL dataframe, decide which actions and analyses
are useful, execute them through tools, and then produce a professional report.

AUTONOMOUS WORKFLOW:
1. Start by inspecting the dataset.
2. Check data quality.
3. Decide whether exact duplicate rows should be removed. If duplicates exist,
   remove them because they add repeated records without adding information.
4. Re-check missing values after cleaning.
5. For missing values in numerical columns, you may fill them with the median
   when that is a reasonable, low-risk cleaning action. Do NOT invent values.
6. Analyze useful numerical columns using statistics.
7. Analyze meaningful categorical/group relationships when possible.
8. If at least two numerical columns exist, calculate correlations.
9. Select visualizations that are relevant to the actual columns. Avoid useless
   charts. Use histograms for important numerical distributions, bar charts for
   useful categorical distributions, scatter plots for numerical relationships,
   and a heatmap when there are at least two numerical columns.
10. After tool results, continue deciding what is still needed. Do not stop after
    the first tool call.
11. Use exact column names.
12. Never invent statistics.
13. Correlation does not prove causation.
14. If the dataset is tiny, explicitly say findings are directional.

You are allowed to make multiple tool calls across multiple turns.
Do not merely describe actions; execute them with tools.
"""
    else:
        instructions = """
You are DataPilot AI, an intelligent data-analysis assistant.
Use tools whenever the user's question requires actual dataset information.
Never invent statistics or column names. Use exact column names.
For comparisons, calculate from actual tool results. For charts, call the
corresponding chart tool. For data modifications, clearly explain what changed.
"""

    initial_input = f"""
Current dataset:
Rows: {context['rows']}
Columns: {context['columns']}
Column names: {context['column_names']}
Numerical columns: {context['numerical_columns']}
Categorical columns: {context['categorical_columns']}

User request:
{question}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=instructions,
        input=initial_input,
        tools=tools,
        tool_choice="auto",
    )

    all_tool_results = []
    max_rounds = 8

    for _ in range(max_rounds):
        calls = [item for item in response.output if item.type == "function_call"]

        if not calls:
            return response.output_text, current_df, chart_list, action_log

        outputs = []

        for item in calls:
            args = json.loads(item.arguments or "{}")
            current_df, result = _execute_tool(
                item.name, args, current_df, chart_list, action_log
            )

            all_tool_results.append({
                "tool": item.name,
                "arguments": args,
                "result": result,
            })

            outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, default=str),
            })

        # Continue the same agent trajectory so it can inspect the returned
        # tool results and decide what to do next.
        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=outputs,
            previous_response_id=response.id,
            tools=tools,
            tool_choice="auto",
        )

    # Safety fallback if the model keeps requesting tools.
    final_response = client.responses.create(
        model="gpt-5.6-luna",
        instructions="""
Write the final DataPilot report using only the tool results already gathered.
Do not invent facts. Mention cleaning actions performed and important findings.
Use concise headings and include a small-data warning when appropriate.
""",
        input=[
            {
                "role": "user",
                "content": json.dumps({
                    "request": question,
                    "tool_results": all_tool_results,
                    "actions": action_log,
                }, default=str),
            }
        ],
    )

    return final_response.output_text, current_df, chart_list, action_log


def ask_ai(question, df):
    """Normal natural-language DataPilot assistant."""
    answer, updated_df, charts, actions = _run_agent(
        question, df, autonomous=False
    )
    return answer, updated_df, (charts[-1] if charts else None)


def run_autonomous_analysis(df):
    """
    Run a multi-step agentic analysis. The model can inspect the dataframe,
    decide cleaning/analysis actions, execute tools, continue reasoning, and
    return a final report plus the modified dataframe and selected charts.
    """
    return _run_agent(
        "Perform a complete autonomous analysis of this dataset. Inspect, clean "
        "when justified, analyze important patterns, compare useful groups, "
        "check numerical relationships, select useful visualizations, and "
        "produce a professional final report.",
        df,
        autonomous=True,
    )
