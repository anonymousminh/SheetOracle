from json import load
import os
import requests
from typing import Dict, List, Any 
from slack_bolt import App, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

from data_profiler import process_and_seed_csv
from ui_builder import build_schema_summary_card
from agent_graph import analytical_agent


# Initialize your app with your environment tokens
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

@app.event("message")
def handle_incoming_file_uploads(event: Dict[str, Any], say: Say, client: WebClient, context: Dict[str, Any]):
    # Ignore messages the bot itself posted to avoid feedback loops
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    # Fallback routing: if the bot is mentioned inside a message event
    # (e.g. the app_mention event is not subscribed), hand off to the Q&A handler.
    bot_user_id = context.get("bot_user_id")
    text = event.get("text", "")
    if bot_user_id and f"<@{bot_user_id}>" in text:
        handle_conversational_questions(event, say, client, context)
        return

    # Check if the message contains any files
    if "files" not in event:
        return

    for file_info in event["files"]:
        file_name = file_info.get("name", "")
        file_id = file_info.get("id")
        download_url = file_info.get("url_private_download")
        
        # Intercept CSV file
        if not file_name.endswith(".csv") or not download_url:
            continue
            
        # Determine the thread timestamp to keep the analysis isolated to this thread
        thread_ts = event.get("thread_ts", event["ts"])
        channel_id = event["channel"]
        
        # 1. Immediately send an interactive UI update to the user
        say(
            text=f"📥 **File detected:** `{file_name}`. Initializing secure data stream...",
            thread_ts=thread_ts
        )
        
        try:
            # 2. Authenticate and download the file from Slack's servers
            headers = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
            response = requests.get(download_url, headers=headers, stream=True)
            
            if response.status_code == 200:
                # Ensure a local temp directory exists
                os.makedirs("./tmp_spreadsheets", exist_ok=True)
                
                # Save the file using the unique thread timestamp as the filename
                target_csv_path = f"./tmp_spreadsheets/{thread_ts}.csv"
                
                with open(target_csv_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Run profiler & Database Seeding
                total_rows, columns_metadata = process_and_seed_csv(target_csv_path, thread_ts)

                # Build the UI Block Kit Presentation Card
                ui_blocks = build_schema_summary_card(
                    file_name=file_name,
                    total_rows=total_rows,
                    columns_metadata=columns_metadata
                )
                        
                # 3. Inform the user that the data layer is locked in
                client.chat_postMessage(
                    channel=event["channel"],
                    thread_ts=thread_ts,
                    blocks= ui_blocks,
                    text=f"Profile for {file_name} generated successfully"
                )
                print(f"Data pipeline fully wired for session thread: {thread_ts}")
                
            else:
                say(text=f"❌ Failed to download file. Status code: {response.status_code}", thread_ts=thread_ts)
                
        except Exception as e:
            say(text=f"⚠️ An error occurred during file ingestion: {str(e)}", thread_ts=thread_ts)

# Listen for user analytics questions
@app.event("app_mention")
def handle_conversational_questions(event: Dict[str, Any], say: Say, client: WebClient, context: Dict[str, Any]) -> None:
    """
    Listen for follow-up data questions from users,
    validates that an active SQLite session database exists for this thread,
    and prepares the handoff to LangGraph.
    """
    # Extract the values from payload and strip the bot mention from the question
    raw_text = event.get("text", "")
    bot_user_id = context.get("bot_user_id")
    if bot_user_id:
        raw_text = raw_text.replace(f"<@{bot_user_id}>", "")
    user_question = raw_text.strip()
    channel_id = event.get("channel", "")

    # Extract the thread timestamp
    thread_ts = event.get("thread_ts")

    # Session Gatekeeper Check
    if not thread_ts:
        main_ts = event.get("ts", "")
        say(
            text="Sheetr runs inside file threads! Please upload a .csv file first then tag me inside its thread to ask questions.",
            thread_ts=main_ts
        )
        return 
    
    # Verify if a temporary database table has already been built for this thread
    expected_db_path = f"./tmp_databases/{thread_ts}.db"

    # 3. Perform Session validation check
    if os.path.exists(expected_db_path):
        print(f"[Session Approved] Routing to LangGraph for thread {thread_ts}")

        # Keep user engaged while LangGraph is processing
        say(
            text="Thinking... Executing graped-based analytical reasoning...",
            thread_ts=thread_ts
        )

        try:
            # Invoke the graph execution loop
            final_state = analytical_agent.invoke({
                "user_question": user_question,
                "db_path": expected_db_path,
                "generated_sql": "",
                "query_result": "",
                "error_message": "",
                "retry_count": 0
            })

            # Extract the answer from the state
            bot_response = final_state.get("final_response", "I encountered an error while processing your question.")
            say(text=bot_response, thread_ts=thread_ts)
            print(f"[Success] Successfully posted analytical answer to thread {thread_ts}")
        except Exception as e:
            print(f"[Error] Graph invocation failed: {str(e)}")
            say(text=f"❌ An error occurred while processing your question: {str(e)}", thread_ts=thread_ts)


    else:
        say(
            text="No active database found for this thread. Upload a CSV file first, then ask your question here.",
            thread_ts=thread_ts
        )


if __name__ == "__main__":
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not app_token:
        raise ValueError("Missing SLACK_APP_TOKEN")
    # Start your app using Socket Mode Handler
    handler = SocketModeHandler(app=app, app_token=app_token)
    handler.start()