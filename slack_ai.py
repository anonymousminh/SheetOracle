"""
Slack AI helpers: streaming, thinking steps, feedback, and suggested prompts.
"""

from typing import List, Optional, Sequence

from slack_sdk import WebClient
from slack_sdk.models.blocks import (
    Block,
    ContextActionsBlock,
    FeedbackButtonObject,
    FeedbackButtonsElement,
)
from slack_sdk.models.messages.chunk import MarkdownTextChunk, TaskUpdateChunk
from slack_sdk.web.chat_stream import ChatStream

from agent_graph import GRAPH_NODE_PROGRESS, get_agent_task_order

SUGGESTED_PROMPTS = [
    {"title": "Summarize Dataset", "message": "Give me a high-level summary of this dataset"},
    {"title": "Plot Trends", "message": "Plot the main trends over time"},
    {"title": "Top Categories", "message": "What are the top 5 categories by value?"},
]

LOADING_MESSAGES = [
    "Crunching spreadsheet rows…",
    "Teaching SQL to speak finance…",
    "Polishing charts for executives…",
    "Consulting the data oracle…",
    "Summoning insights from your CSV…",
]

FEEDBACK_ACTION_ID = "sheetoracle_feedback"


def build_feedback_blocks() -> List[Block]:
    return [
        ContextActionsBlock(
            elements=[
                FeedbackButtonsElement(
                    action_id=FEEDBACK_ACTION_ID,
                    positive_button=FeedbackButtonObject(
                        text="Good Response",
                        accessibility_label="Submit positive feedback on this response",
                        value="good-feedback",
                    ),
                    negative_button=FeedbackButtonObject(
                        text="Bad Response",
                        accessibility_label="Submit negative feedback on this response",
                        value="bad-feedback",
                    ),
                )
            ]
        )
    ]


def set_thread_status(
    client: WebClient,
    *,
    channel_id: str,
    thread_ts: str,
    status: str,
    loading_messages: Optional[Sequence[str]] = None,
) -> None:
    try:
        client.assistant_threads_setStatus(
            channel_id=channel_id,
            thread_ts=thread_ts,
            status=status,
            loading_messages=list(loading_messages or LOADING_MESSAGES),
        )
    except Exception as exc:
        print(f"[Slack AI] setStatus failed: {exc}")


def create_response_stream(
    client: WebClient,
    *,
    channel_id: str,
    thread_ts: str,
    team_id: Optional[str],
    user_id: Optional[str],
) -> ChatStream:
    return client.chat_stream(
        channel=channel_id,
        thread_ts=thread_ts,
        recipient_team_id=team_id,
        recipient_user_id=user_id,
        task_display_mode="plan",
        buffer_size=1,
    )


def _task_chunk(task_id: str, status: str) -> TaskUpdateChunk:
    return TaskUpdateChunk(
        id=task_id,
        title=GRAPH_NODE_PROGRESS[task_id],
        status=status,
    )


def start_thinking_steps(streamer: ChatStream, task_order: Sequence[str]) -> None:
    if not task_order:
        return
    streamer.append(chunks=[_task_chunk(task_order[0], "in_progress")])


def advance_thinking_step(
    streamer: ChatStream,
    *,
    completed_task: str,
    task_order: Sequence[str],
) -> None:
    chunks: List[TaskUpdateChunk] = [_task_chunk(completed_task, "complete")]
    try:
        completed_index = task_order.index(completed_task)
    except ValueError:
        streamer.append(chunks=chunks)
        return

    if completed_index + 1 < len(task_order):
        next_task = task_order[completed_index + 1]
        chunks.append(_task_chunk(next_task, "in_progress"))

    streamer.append(chunks=chunks)


def stream_agent_response(
    streamer: ChatStream,
    *,
    response_text: str,
    include_feedback: bool = True,
) -> None:
    streamer.append(markdown_text=response_text)
    streamer.stop(blocks=build_feedback_blocks() if include_feedback else None)


def stream_error_response(streamer: ChatStream, error_text: str) -> None:
    streamer.append(
        chunks=[
            TaskUpdateChunk(
                id="error",
                title="Analysis failed",
                status="error",
                output=error_text,
            ),
            MarkdownTextChunk(text=error_text),
        ]
    )
    streamer.stop()


def format_agent_response(
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


def build_task_order(chart_requested: bool) -> List[str]:
    return get_agent_task_order(chart_requested)


def upload_chart_if_present(
    client: WebClient,
    *,
    channel_id: str,
    thread_ts: str,
    chart_file_path: str,
) -> None:
    import os

    if chart_file_path and os.path.exists(chart_file_path):
        client.files_upload_v2(
            channel=channel_id,
            thread_ts=thread_ts,
            file=chart_file_path,
            title="Data Visualization",
            initial_comment="📊 Chart generated from your dataset.",
        )
