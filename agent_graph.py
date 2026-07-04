import sqlite3
from typing import Dict, TypedDict, Any, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Define the state schema
class AgentState(TypedDict):
    """Tracks the continuous state of analytical conversation engine."""
    user_question: str
    db_path: str
    generated_sql: str
    query_result: str
    error_message: str
    final_response: str
    retry_count: int

# Initialize the reasoning engine
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the nodes
def generate_sql_node(state: AgentState) -> AgentState:
    """Inspects the database context and generate a SQL statement"""
    db_path = state["db_path"]
    question = state["user_question"]
    error = state.get("error_message", "")

    # Inspect the database context
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(uploaded_data)")
    columns = cursor.fetchall()
    conn.close()

    schema_context = "\n".join([f"- {col[1]}: {col[2]}" for col in columns])

    # System prompt for SQL generation
    system_prompt = (
        f"You are an elite SQL expert. Write a raw SQL query targeting a table named 'uploaded_data' to answer the user's question."
        f"Available columns: \n{schema_context}\n\n"
        f"CRITICAL: Return ONLY the executable SQL executable SQL query string. Do not use markdown backticks (```sql) or explainations."
    )

    # Add error notes if a previous attempt failed
    if error:
        system_prompt += f"\n\n Your previous query failed with this error: {error}. Please fix the error and try again."

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages)
    sql_query = response.content.strip()

    return {"generated_sql": sql_query, "error_message": ""}

def execute_sql_node(state: AgentState) -> Dict[str, Any]:
    """Executes the generated SQL query and returns the result"""
    sql_query = state["generated_sql"]
    db_path = state["db_path"]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        return {"query_result": rows, "error_message": ""}
    except Exception as e:
        return {"error_message": str(e), "retry_count": state.get("retry_count", 0) + 1}

def formulate_response_node(state: AgentState) -> Dict[str, Any]:
    """Formats the query result into a natural language response"""
    question = state["user_question"]
    data = state["query_result"]

    prompt = f"Synthesize a user-friendly answer to this question: '{question}' based on the following data: {data}"
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"final_response": response.content}

def check_execution_success(state: AgentState) -> Literal["generate_sql", "formulate_response"]:
    """Evaluates if the database layer threw an error and handles fallback paths"""
    if state.get("error_message") and state.get("retry_count", 0) < 3:
        print(f"SQL error detected; {state['error_message']}. Retrying query execution...")
        return "generate_sql"
    return "formulate_response"


# Build and compile the workflow graph

workflow = StateGraph(AgentState)

workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("formulate_response", formulate_response_node)

workflow.add_edge(START, "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")

# Inject the self-correcting conditional loop check
workflow.add_conditional_edges(
    "execute_sql",
    check_execution_success,
    {
        "generate_sql": "generate_sql",
        "formulate_response": "formulate_response",
    },
)

workflow.add_edge("formulate_response", END)

# Compile the graph
analytical_agent = workflow.compile()
    
    
    