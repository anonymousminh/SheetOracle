"""
LangGraph analytical agent for SheetOracle.

Pipeline:
  1. Resolve chart intent (keyword match, no LLM)
  2. Generate and execute SQL against the session SQLite DB
  3. Optionally render a matplotlib chart
  4. Summarize results in natural language
"""

import ast
import os
import re
import sqlite3
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

import matplotlib
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    user_question: str
    db_path: str
    generated_sql: str
    query_result: Union[str, List[tuple]]
    error_message: str
    final_response: str
    retry_count: int
    chart_requested: bool
    chart_intent_resolved: bool
    chart_file_path: str
    chart_error: str
    chart_warning: str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MAX_CHART_ROWS = 50
MAX_PIE_SLICES = 10
MAX_PROMPT_ROWS = 20
MAX_SQL_RETRIES = 3
CHART_OUTPUT_DIR = "./tmp_charts"

CHART_KEYWORDS = (
    "plot",
    "chart",
    "graph",
    "visualize",
    "visualise",
    "visual",
    "bar chart",
    "line chart",
    "pie chart",
    "scatter plot",
    "scatter",
    "histogram",
    "trend",
    "over time",
    "timeline",
    "draw",
)

GRAPH_NODE_PROGRESS = {
    "generate_sql": "Querying data...",
    "execute_sql": "Running query...",
    "generate_chart": "Building chart...",
    "formulate_response": "Summarizing results...",
}

CHART_COLORS = ("#36C5F0", "#ECB22E", "#2EB67D", "#E01E5A")


# ---------------------------------------------------------------------------
# Public helpers (used by app.py and tests)
# ---------------------------------------------------------------------------


def is_chart_requested(question: str) -> bool:
    """Return True when the question asks for a visual/chart output."""
    normalized = question.lower()
    return any(keyword in normalized for keyword in CHART_KEYWORDS)


def build_initial_state(user_question: str, db_path: str) -> Dict[str, Any]:
    """Create the initial graph state with chart routing already resolved."""
    return {
        "user_question": user_question,
        "db_path": db_path,
        "generated_sql": "",
        "query_result": "",
        "error_message": "",
        "final_response": "",
        "retry_count": 0,
        "chart_requested": is_chart_requested(user_question),
        "chart_intent_resolved": True,
        "chart_file_path": "",
        "chart_error": "",
        "chart_warning": "",
    }


# ---------------------------------------------------------------------------
# Query-result helpers
# ---------------------------------------------------------------------------


def _parse_query_result(query_result: Union[str, List[tuple], Any]) -> List[tuple]:
    """Normalize SQL output into a list of row tuples."""
    if isinstance(query_result, list):
        return query_result
    if isinstance(query_result, str):
        if not query_result:
            return []
        return ast.literal_eval(query_result)
    raise ValueError(f"Unexpected query_result type: {type(query_result)}")


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _coerce_numeric(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} contains null values that cannot be plotted.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}.") from exc


def _summarize_query_result_for_prompt(query_result: Union[str, List[tuple], Any]) -> str:
    """Build a compact dataset summary for the response LLM prompt."""
    try:
        rows = _parse_query_result(query_result)
    except (ValueError, SyntaxError):
        text = str(query_result)
        return text[:2000] + ("..." if len(text) > 2000 else "")

    if not rows:
        return "No rows returned."
    if len(rows) <= MAX_PROMPT_ROWS:
        return str(rows)

    total = len(rows)
    col_count = len(rows[0]) if isinstance(rows[0], (tuple, list)) else 1
    lines = [
        f"Total rows: {total} (truncated for summarization).",
        f"First rows: {rows[:5]}",
    ]
    if total > 8:
        lines.append(f"Last rows: {rows[-3:]}")

    value_index = 1 if col_count >= 2 else 0
    numeric_rows = [
        row for row in rows
        if len(row) > value_index and _is_numeric(row[value_index])
    ]
    if numeric_rows:
        values = [float(row[value_index]) for row in numeric_rows]
        lines.append(
            f"Numeric stats (column {value_index + 1}): "
            f"min={min(values):.4g}, max={max(values):.4g}, "
            f"avg={sum(values) / len(values):.4g}, sum={sum(values):.4g}"
        )
        top_rows = sorted(
            numeric_rows,
            key=lambda row: float(row[value_index]),
            reverse=True,
        )[:5]
        lines.append(f"Top 5 rows by value: {top_rows}")

    return "\n".join(lines)


def _load_table_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(uploaded_data)")
    columns = cursor.fetchall()
    conn.close()
    return "\n".join(f"- {name}: {col_type}" for _, name, col_type, *_ in columns)


# ---------------------------------------------------------------------------
# Chart planning and rendering
# ---------------------------------------------------------------------------


def _looks_like_time_series(x_values: List[str]) -> bool:
    patterns = (
        r"^\d{4}-\d{2}",
        r"^\d{4}$",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    )
    sample = x_values[: min(len(x_values), 5)]
    return any(
        any(re.search(pattern, value.lower()) for pattern in patterns)
        for value in sample
    )


def _infer_chart_spec(question: str, data_points: List[tuple]) -> Dict[str, str]:
    """Pick chart type and labels using simple rules (no LLM)."""
    question_lower = question.lower()
    x_values = [str(row[0]) for row in data_points]

    pie_keywords = ("pie", "donut", "share", "percentage", "percent", "proportion", "composition")
    line_keywords = ("over time", "trend", "timeline", "time series", "monthly", "weekly", "daily")
    scatter_keywords = ("scatter", "correlation", " vs ", " versus ")

    if any(keyword in question_lower for keyword in pie_keywords) and len(data_points) <= MAX_PIE_SLICES:
        chart_type = "pie"
    elif (
        any(keyword in question_lower for keyword in scatter_keywords)
        and data_points
        and _is_numeric(data_points[0][0])
    ):
        chart_type = "scatter"
    elif any(keyword in question_lower for keyword in line_keywords) or _looks_like_time_series(x_values):
        chart_type = "line"
    elif "line chart" in question_lower:
        chart_type = "line"
    else:
        chart_type = "bar"

    title = question.strip().rstrip("?")
    if not title or len(title) > 80:
        title = "Data Visualization"
    else:
        title = title[0].upper() + title[1:]

    if chart_type == "line" and _looks_like_time_series(x_values):
        xlabel = "Time Period"
    elif chart_type == "pie":
        xlabel = ""
    else:
        xlabel = "Category"

    ylabel = "Value"
    if any(word in question_lower for word in ("expense", "cost", "spend", "amount", "revenue", "sales")):
        ylabel = "Amount (USD)" if "usd" in question_lower or "expense" in question_lower else "Amount"

    return {
        "chart_type": chart_type,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
    }


def _validate_and_prepare_chart_data(
    data_points: List[tuple],
    chart_type: str,
) -> Tuple[List[tuple], str]:
    """Ensure rows are plottable and cap how many points are rendered."""
    if not data_points:
        raise ValueError("Query returned no rows to plot.")

    normalized: List[tuple] = []
    for index, row in enumerate(data_points):
        if not isinstance(row, (tuple, list)):
            raise ValueError(f"Row {index + 1} is not a plottable record.")
        if len(row) < 2:
            raise ValueError("Each row must have at least two columns (label and value).")
        normalized.append(tuple(row))

    for index, row in enumerate(normalized):
        _coerce_numeric(row[1], f"Value in row {index + 1}")

    if chart_type == "scatter":
        for index, row in enumerate(normalized):
            if not _is_numeric(row[0]):
                raise ValueError(
                    f"Scatter plots need numeric X values; got {row[0]!r} in row {index + 1}."
                )

    warnings: List[str] = []
    total_rows = len(normalized)

    if chart_type == "pie" and total_rows > MAX_PIE_SLICES:
        normalized = normalized[:MAX_PIE_SLICES]
        warnings.append(f"Pie chart limited to {MAX_PIE_SLICES} slices.")

    if len(normalized) > MAX_CHART_ROWS:
        normalized = normalized[:MAX_CHART_ROWS]
        warnings.append(f"Chart shows the first {MAX_CHART_ROWS} of {total_rows} rows.")

    return normalized, " ".join(warnings)


def _session_id_from_db_path(db_path: str) -> str:
    return db_path.split("/")[-1].replace(".db", "")


def _draw_chart(
    ax: plt.Axes,
    chart_type: str,
    x_data: List[str],
    y_data: List[float],
    data_points: List[tuple],
) -> None:
    if chart_type == "line":
        ax.plot(x_data, y_data, color=CHART_COLORS[0], marker="o", linewidth=2.5, markersize=8)
    elif chart_type == "bar":
        ax.bar(x_data, y_data, color=CHART_COLORS[1], edgecolor="none", alpha=0.9, width=0.6)
    elif chart_type == "scatter":
        x_numeric = [_coerce_numeric(row[0], "X value") for row in data_points]
        ax.scatter(x_numeric, y_data, color=CHART_COLORS[2], s=100, alpha=0.8)
    elif chart_type == "pie":
        ax.pie(
            y_data,
            labels=x_data,
            autopct="%1.1f%%",
            startangle=140,
            colors=CHART_COLORS,
        )
    else:
        ax.bar(x_data, y_data, color=CHART_COLORS[1], edgecolor="none", alpha=0.9, width=0.6)


def _apply_chart_style(
    fig: plt.Figure,
    ax: plt.Axes,
    chart_type: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if chart_type == "pie":
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        return

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _save_chart(fig: plt.Figure, session_id: str) -> str:
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    chart_path = f"{CHART_OUTPUT_DIR}/{session_id}.png"
    fig.savefig(chart_path, dpi=150)
    return chart_path


def _chart_node_result(
    *,
    chart_file_path: str = "",
    chart_error: str = "",
    chart_warning: str = "",
) -> Dict[str, str]:
    return {
        "chart_file_path": chart_file_path,
        "chart_error": chart_error,
        "chart_warning": chart_warning,
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Fallback routing when chart intent was not set in build_initial_state()."""
    chart_requested = is_chart_requested(state["user_question"])
    print(f"[Router] chart_requested={chart_requested}")
    return {"chart_requested": chart_requested}


def route_from_start(state: AgentState) -> Literal["generate_sql", "classify_intent"]:
    if state.get("chart_intent_resolved"):
        return "generate_sql"
    return "classify_intent"


def generate_sql_node(state: AgentState) -> Dict[str, Any]:
    question = state["user_question"]
    error = state.get("error_message", "")
    schema_context = _load_table_schema(state["db_path"])

    system_prompt = (
        "You are an elite SQL expert. Write a raw SQL query targeting a table named "
        "'uploaded_data' to answer the user's question.\n"
        f"Available columns:\n{schema_context}\n\n"
        "CRITICAL: Return ONLY the executable SQL query string. "
        "Do not use markdown backticks or explanations."
    )
    if error:
        system_prompt += f"\n\nPrevious query failed with: {error}. Fix the query and try again."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])
    return {"generated_sql": response.content.strip(), "error_message": ""}


def execute_sql_node(state: AgentState) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(state["db_path"])
        cursor = conn.cursor()
        cursor.execute(state["generated_sql"])
        rows = cursor.fetchall()
        conn.close()
        return {"query_result": rows, "error_message": ""}
    except Exception as exc:
        return {
            "error_message": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
        }


def generate_chart_node(state: AgentState) -> Dict[str, str]:
    question = state["user_question"]
    session_id = _session_id_from_db_path(state["db_path"])

    try:
        raw_rows = _parse_query_result(state["query_result"])
        chart_spec = _infer_chart_spec(question, raw_rows)
        chart_type = chart_spec["chart_type"]
        rows, chart_warning = _validate_and_prepare_chart_data(raw_rows, chart_type)

        x_data = [str(row[0]) for row in rows]
        y_data = [_coerce_numeric(row[1], "Value") for row in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        _draw_chart(ax, chart_type, x_data, y_data, rows)
        _apply_chart_style(
            fig,
            ax,
            chart_type,
            chart_spec["title"],
            chart_spec["xlabel"],
            chart_spec["ylabel"],
        )
        fig.tight_layout()

        chart_path = _save_chart(fig, session_id)
        plt.close(fig)

        print(f"[Chart] saved {chart_type} chart to {chart_path}")
        if chart_warning:
            print(f"[Chart] {chart_warning}")

        return _chart_node_result(chart_file_path=chart_path, chart_warning=chart_warning)

    except ValueError as exc:
        print(f"[Chart] failed: {exc}")
        return _chart_node_result(chart_error=str(exc))
    except Exception as exc:
        print(f"[Chart] unexpected error: {exc}")
        return _chart_node_result(chart_error=f"Unexpected chart rendering error: {exc}")


def _build_summary_prompt(
    question: str,
    data_summary: str,
    chart_file_path: str,
    chart_error: str,
    chart_requested: bool,
) -> str:
    prompt = (
        "You are an elite corporate financial data analyst summarizing findings for executives.\n"
        f"Analyze this database dataset summary: {data_summary}\n"
        f"To directly answer the user's inquiry: '{question}'\n\n"
        "Provide a concise narrative highlighting the peaks, valleys, or key trends."
    )

    if chart_file_path:
        prompt += (
            "\n\nThe system already attached a chart to the user's message. "
            "Do NOT provide code or instructions for building a chart. "
            "Focus on business insights from the data."
        )
    elif chart_requested and chart_error:
        prompt += (
            f"\n\nChart generation failed ({chart_error}). "
            "Do not claim a chart was shown. Summarize the data in text instead."
        )

    return prompt


def formulate_response_node(state: AgentState) -> Dict[str, Any]:
    prompt = _build_summary_prompt(
        question=state["user_question"],
        data_summary=_summarize_query_result_for_prompt(state["query_result"]),
        chart_file_path=state.get("chart_file_path", ""),
        chart_error=state.get("chart_error", ""),
        chart_requested=state.get("chart_requested", False),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": str(response.content).strip()}


def route_after_sql(state: AgentState) -> Literal["generate_sql", "formulate_response", "generate_chart"]:
    if state.get("error_message") and state.get("retry_count", 0) < MAX_SQL_RETRIES:
        print(f"[SQL] retrying after error: {state['error_message']}")
        return "generate_sql"
    if state.get("chart_requested"):
        return "generate_chart"
    return "formulate_response"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_chart", generate_chart_node)
workflow.add_node("formulate_response", formulate_response_node)

workflow.add_conditional_edges(
    START,
    route_from_start,
    {"generate_sql": "generate_sql", "classify_intent": "classify_intent"},
)
workflow.add_edge("classify_intent", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges(
    "execute_sql",
    route_after_sql,
    {
        "generate_sql": "generate_sql",
        "generate_chart": "generate_chart",
        "formulate_response": "formulate_response",
    },
)
workflow.add_edge("generate_chart", "formulate_response")
workflow.add_edge("formulate_response", END)

analytical_agent = workflow.compile()
