"""
SheetOracle Slack bot.

Handles CSV uploads into per-thread SQLite sessions and routes @mentions
to the LangGraph analytical agent (SQL + optional charts + summary).
Uses Slack AI features: loading status, thinking steps, and streamed replies.
"""

import os
import threading
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from slack_bolt import Ack, App, BoltContext, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.context.set_status import SetStatus
from slack_sdk import WebClient

load_dotenv()

from agent_graph import analytical_agent, build_initial_state
from data_profiler import process_and_seed_csv
from slack_ai import (
    FEEDBACK_ACTION_ID,
    LOADING_MESSAGES,
    SUGGESTED_PROMPTS,
    advance_thinking_step,
    build_task_order,
    create_response_stream,
    format_agent_response,
    set_thread_status,
    start_thinking_steps,
    stream_agent_response,
    stream_error_response,
    upload_chart_if_present,
)
from ui_builder import build_schema_summary_card

app = App(token=os.getenv("SLACK_BOT_TOKEN"))


# ---------------------------------------------------------------------------
# Agent → Slack helpers
# ---------------------------------------------------------------------------


def _run_agent_and_reply(
    client: WebClient,
    channel_id: str,
    thread_ts: str,
    user_question: str,
    db_path: str,
    *,
    team_id: Optional[str],
    user_id: Optional[str],
) -> None:
    """Run the LangGraph agent in a background thread and stream the result."""
    initial_state = build_initial_state(user_question, db_path)
    accumulated = dict(initial_state)
    task_order = build_task_order(initial_state["chart_requested"])
    streamer = create_response_stream(
        client,
        channel_id=channel_id,
        thread_ts=thread_ts,
        team_id=team_id,
        user_id=user_id,
    )

    try:
        start_thinking_steps(streamer, task_order)

        for event in analytical_agent.stream(initial_state):
            for node_name, node_output in event.items():
                accumulated.update(node_output)
                if node_name in task_order:
                    advance_thinking_step(
                        streamer,
                        completed_task=node_name,
                        task_order=task_order,
                    )

        response_text = format_agent_response(
            accumulated.get(
                "final_response",
                "I encountered an error while processing your question.",
            ),
            chart_requested=accumulated.get("chart_requested", False),
            chart_file_path=accumulated.get("chart_file_path", ""),
            chart_error=accumulated.get("chart_error", ""),
            chart_warning=accumulated.get("chart_warning", ""),
        )
        stream_agent_response(streamer, response_text=response_text)
        upload_chart_if_present(
            client,
            channel_id=channel_id,
            thread_ts=thread_ts,
            chart_file_path=accumulated.get("chart_file_path", ""),
        )
        print(f"[Success] answered thread {thread_ts}")

    except Exception as exc:
        print(f"[Error] agent failed for thread {thread_ts}: {exc}")
        stream_error_response(
            streamer,
            f"An error occurred while processing your question: {exc}",
        )


def _handle_agent_question(
    event: Dict[str, Any],
    say: Say,
    client: WebClient,
    context: BoltContext,
    *,
    set_status: Optional[SetStatus] = None,
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

    if not user_question:
        say(
            text="Ask me a question about the uploaded spreadsheet, e.g. *What are the top expenses?*",
            thread_ts=thread_ts,
        )
        return

    print(f"[Session] routing question to agent for thread {thread_ts}")

    if set_status:
        set_status(
            status="Analyzing your data...",
            loading_messages=LOADING_MESSAGES,
        )
    else:
        set_thread_status(
            client,
            channel_id=channel_id,
            thread_ts=thread_ts,
            status="Analyzing your data...",
        )

    team_id = event.get("team") or context.team_id
    user_id = event.get("user") or context.user_id

    threading.Thread(
        target=_run_agent_and_reply,
        kwargs={
            "client": client,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "user_question": user_question,
            "db_path": db_path,
            "team_id": team_id,
            "user_id": user_id,
        },
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Slack event handlers
# ---------------------------------------------------------------------------


@app.event("app_home_opened")
def handle_app_home_opened(client: WebClient, event: Dict[str, Any]) -> None:
    """Set suggested prompts when a user opens the agent DM."""
    if event.get("tab") != "messages":
        return

    try:
        client.assistant_threads_setSuggestedPrompts(
            channel_id=event["channel"],
            title="Ask SheetOracle about your data",
            prompts=SUGGESTED_PROMPTS,
        )
    except Exception as exc:
        print(f"[Slack AI] failed to set suggested prompts: {exc}")


@app.event("message")
def handle_incoming_file_uploads(
    event: Dict[str, Any],
    say: Say,
    client: WebClient,
    context: BoltContext,
) -> None:
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    # @mentions are handled by app_mention only — Slack also emits a message event
    # for the same post, which would otherwise run the agent twice.

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
    context: BoltContext,
    set_status: Optional[SetStatus] = None,
) -> None:
    _handle_agent_question(
        event,
        say,
        client,
        context,
        set_status=set_status,
    )


@app.action(FEEDBACK_ACTION_ID)
def handle_feedback(
    ack: Ack,
    body: Dict[str, Any],
    client: WebClient,
    context: BoltContext,
) -> None:
    ack()

    channel_id = context.channel_id
    user_id = context.user_id
    message_ts = body["message"]["ts"]
    feedback_value = body["actions"][0]["value"]

    if feedback_value == "good-feedback":
        text = "Glad that was helpful! :tada:"
    else:
        text = (
            "Sorry that wasn't helpful. Try rephrasing your question or ask for a chart "
            "to visualize the data differently."
        )

    try:
        client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            thread_ts=message_ts,
            text=text,
        )
    except Exception as exc:
        print(f"[Slack AI] feedback response failed: {exc}")


if __name__ == "__main__":
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not app_token:
        raise ValueError("Missing SLACK_APP_TOKEN")
    SocketModeHandler(app=app, app_token=app_token).start()
