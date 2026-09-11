"""Local smoke test for the LangGraph analytical agent."""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from agent_graph import GRAPH_NODE_PROGRESS, analytical_agent, build_initial_state


def _print_node_output(node_name: str, node_output: dict) -> None:
    if progress := GRAPH_NODE_PROGRESS.get(node_name):
        print(f"[Progress] {progress}")
    print(f"\n[Node: {node_name}]")

    if sql := node_output.get("generated_sql"):
        print(f"-> Generated SQL: {sql}")
    if rows := node_output.get("query_result"):
        print(f"-> Query Results: {rows}")
    if chart_path := node_output.get("chart_file_path"):
        print(f"-> Chart saved to: {chart_path}")
    if warning := node_output.get("chart_warning"):
        print(f"-> Chart warning: {warning}")
    if error := node_output.get("chart_error"):
        print(f"-> Chart error: {error}")
    if response := node_output.get("final_response"):
        print(f"-> Final Response: {response}")


async def run_test() -> None:
    initial_state = build_initial_state(
        user_question="Plot the expenses over time",
        db_path="tmp_databases/1783294038.516919.db",
    )

    print("--- Executing LangGraph Text-to-SQL Workflow ---")
    async for event in analytical_agent.astream(initial_state):
        for node_name, node_output in event.items():
            _print_node_output(node_name, node_output)


if __name__ == "__main__":
    asyncio.run(run_test())
