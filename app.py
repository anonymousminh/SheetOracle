"""
SheetOracle Slack bot.

Handles CSV uploads into per-thread SQLite sessions and routes @mentions
to the LangGraph analytical agent (SQL + optional charts + summary).
"""

import os
import threading
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from slack_bolt import App, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

load_dotenv()

from agent_graph import GRAPH_NODE_PROGRESS, analytical_agent, build_initial_state
from data_profiler import process_and_seed_csv
from ui_builder import build_schema_summary_card

app = App(token=os.getenv("SLACK_BOT_TOKEN"))


# ---------------------------------------------------------------------------
# Agent → Slack helpers
# ---------------------------------------------------------------------------


def _update_progress(client: WebClient, channel_id: str, message_ts: str, text: str) -> None:
    try:
        client.chat_update(channel=channel_id, ts=message_ts, text=text)
    except Exception as exc:
        print(f"[Progress] update failed: {exc}")


def _clear_progress(client: WebClient, channel_id: str, message_ts: str) -> None:
    try:
        client.chat_delete(channel=channel_id, ts=message_ts)
    except Exception as exc:
        print(f"[Progress] delete failed: {exc}")


def _format_slack_response(
    bot_response: str,
    *,
    chart_requested: bool,
    chart_file_path: str,
    chart_error: str,
    chart_warning: str,
) -> str:
    parts = [bot_response]
    if chart_warning:
        parts.append(f"_{chart_warning}_")
    if chart_requested and chart_error and not chart_file_path:
        parts.append(f"⚠️ *Chart could not be generated:* {chart_error}")
    return "\n\n".join(parts)


def _post_agent_result(
    client: WebClient,
    channel_id: str,
    thread_ts: str,
    accumulated: Dict[str, Any],
) -> None:
    bot_response = accumulated.get(
        "final_response",
        "I encountered an error while processing your question.",
    )
    chart_file_path = accumulated.get("chart_file_path", "")
    slack_message = _format_slack_response(
        bot_response,
        chart_requested=accumulated.get("chart_requested", False),
        chart_file_path=chart_file_path,
        chart_error=accumulated.get("chart_error", ""),
        chart_warning=accumulated.get("chart_warning", ""),
    )

    if chart_file_path and os.path.exists(chart_file_path):
        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=chart_file_path,
            title="Data Visualization",
            initial_comment=slack_message,
        )
    else:
        client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=slack_message)


def _run_agent_and_reply(
    client: WebClient,
    channel_id: str,
    thread_ts: str,
    user_question: str,
    db_path: str,
    status_ts: str | None,
) -> None:
    """Run the LangGraph agent in a background thread and post the result."""
    initial_state = build_initial_state(user_question, db_path)
    accumulated = dict(initial_state)

    try:
        for event in analytical_agent.stream(initial_state):
            for node_name, node_output in event.items():
                accumulated.update(node_output)
                if status_ts and (progress := GRAPH_NODE_PROGRESS.get(node_name)):
                    _update_progress(client, channel_id, status_ts, progress)

        if status_ts:
            _clear_progress(client, channel_id, status_ts)
        _post_agent_result(client, channel_id, thread_ts, accumulated)
        print(f"[Success] answered thread {thread_ts}")

    except Exception as exc:
        print(f"[Error] agent failed for thread {thread_ts}: {exc}")
        error_text = f"An error occurred while processing your question: {exc}"
        if status_ts:
            _update_progress(client, channel_id, status_ts, error_text)
        else:
            try:
                client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=error_text)
            except Exception as post_exc:
                print(f"[Error] failed to notify user in thread {thread_ts}: {post_exc}")


# ---------------------------------------------------------------------------
# Slack event handlers
# ---------------------------------------------------------------------------


@app.event("message")
def handle_incoming_file_uploads(
    event: Dict[str, Any],
    say: Say,
    client: WebClient,
    context: Dict[str, Any],
) -> None:
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    bot_user_id = context.get("bot_user_id")
    text = event.get("text", "")
    if bot_user_id and f"<@{bot_user_id}>" in text:
        handle_conversational_questions(event, say, client, context)
        return

    if "files" not in event:
        return

    for file_info in event["files"]:
        file_name = file_info.get("name", "")
        download_url = file_info.get("url_private_download")
        if not file_name.endswith(".csv") or not download_url:
            continue

        thread_ts = event.get("thread_ts", event["ts"])
        say(
            text=f"📥 **File detected:** `{file_name}`. Initializing secure data stream...",
            thread_ts=thread_ts,
        )

        try:
            headers = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
            response = requests.get(download_url, headers=headers, stream=True)
            if response.status_code != 200:
                say(
                    text=f"❌ Failed to download file. Status code: {response.status_code}",
                    thread_ts=thread_ts,
                )
                continue

            os.makedirs("./tmp_spreadsheets", exist_ok=True)
            csv_path = f"./tmp_spreadsheets/{thread_ts}.csv"
            with open(csv_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    file_handle.write(chunk)

            total_rows, columns_metadata = process_and_seed_csv(csv_path, thread_ts)
            client.chat_postMessage(
                channel=event["channel"],
                thread_ts=thread_ts,
                blocks=build_schema_summary_card(
                    file_name=file_name,
                    total_rows=total_rows,
                    columns_metadata=columns_metadata,
                ),
                text=f"Profile for {file_name} generated successfully",
            )
            print(f"[Upload] session ready for thread {thread_ts}")

        except Exception as exc:
            say(text=f"⚠️ An error occurred during file ingestion: {exc}", thread_ts=thread_ts)


@app.event("app_mention")
def handle_conversational_questions(
    event: Dict[str, Any],
    say: Say,
    client: WebClient,
    context: Dict[str, Any],
) -> None:
    raw_text = event.get("text", "")
    bot_user_id = context.get("bot_user_id")
    if bot_user_id:
        raw_text = raw_text.replace(f"<@{bot_user_id}>", "")

    user_question = raw_text.strip()
    channel_id = event.get("channel", "")
    thread_ts = event.get("thread_ts")

    if not thread_ts:
        say(
            text=(
                "SheetOracle runs inside file threads. Upload a CSV first, "
                "then @mention me inside that thread to ask questions."
            ),
            thread_ts=event.get("ts", ""),
        )
        return

    db_path = f"./tmp_databases/{thread_ts}.db"
    if not os.path.exists(db_path):
        say(
            text="No active database found for this thread. Upload a CSV file first.",
            thread_ts=thread_ts,
        )
        return

    print(f"[Session] routing question to agent for thread {thread_ts}")

    status_ts: str | None = None
    try:
        status_message = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=GRAPH_NODE_PROGRESS["generate_sql"],
        )
        status_ts = status_message["ts"]
    except Exception as exc:
        print(f"[Progress] failed to post status message: {exc}")
        try:
            say(
                text=(
                    "Working on your question... I'll post the answer here shortly. "
                    f"(Status update failed: {exc})"
                ),
                thread_ts=thread_ts,
            )
        except Exception as say_exc:
            print(f"[Progress] failed to notify user in thread {thread_ts}: {say_exc}")

    threading.Thread(
        target=_run_agent_and_reply,
        args=(client, channel_id, thread_ts, user_question, db_path, status_ts),
        daemon=True,
    ).start()


if __name__ == "__main__":
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not app_token:
        raise ValueError("Missing SLACK_APP_TOKEN")
    SocketModeHandler(app=app, app_token=app_token).start()
