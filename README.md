# SheetOracle

A Slack bot that turns spreadsheets into conversational analytics. Upload a CSV or paste a Google Sheet URL in a thread, then @mention SheetOracle to ask questions in plain English. The bot profiles your data, runs SQL against a per-thread SQLite session, optionally generates charts, and streams a natural-language summary back into Slack.

## Features

- **CSV ingestion** — Upload a `.csv` file in a Slack thread; SheetOracle downloads it, infers column types, and seeds an ephemeral SQLite database.
- **Google Sheets** — Paste a Google Sheets URL; data is fetched via the [mcp-google-sheets](https://github.com/xing5/mcp-google-sheets) MCP server.
- **Text-to-SQL agent** — A [LangGraph](https://github.com/langchain-ai/langgraph) pipeline uses GPT-4o-mini to generate read-only SQL, execute it safely, and summarize results.
- **Charts** — Ask for plots, trends, or visualizations; matplotlib renders bar, line, pie, or scatter charts uploaded to the thread.
- **Slack AI UX** — Loading status, thinking steps, streamed replies, suggested prompts, and feedback buttons.

## How it works

```
User uploads CSV or pastes Google Sheet URL
        │
        ▼
data_profiler ──► per-thread SQLite DB (tmp_databases/{thread_ts}.db)
        │
        ▼
User @mentions SheetOracle with a question
        │
        ▼
LangGraph agent:
  1. Generate SQL (LLM)
  2. Execute SQL (read-only, validated)
  3. Generate chart (optional)
  4. Formulate response (LLM)
        │
        ▼
Streamed reply + optional chart image in Slack thread
```

Each Slack thread gets its own isolated database session. Data is stored locally under `tmp_databases/`, `tmp_spreadsheets/`, and `tmp_charts/` (gitignored).

## Prerequisites

- Python 3.11+
- A [Slack app](https://api.slack.com/apps) with **Socket Mode** enabled
- An [OpenAI API key](https://platform.openai.com/)
- (Optional) [uv](https://docs.astral.sh/uv/) / `uvx` for Google Sheets MCP
- (Optional) Google Cloud service account with Sheets API access

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd SheetOracle

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install slack-bolt slack-sdk python-dotenv requests pandas \
            langchain-openai langgraph matplotlib mcp
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Required
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
OPENAI_API_KEY=sk-...

# Google Sheets (optional)
SERVICE_ACCOUNT_PATH=/path/to/service-account.json
DRIVE_FOLDER_ID=          # optional
ENABLED_TOOLS=list_sheets,get_sheet_data
UVX_COMMAND=uvx           # optional; defaults to uvx on PATH
```

### 3. Configure the Slack app

Enable **Socket Mode** and create an app-level token (`SLACK_APP_TOKEN`).

**Bot Token Scopes** (minimum):

- `app_mentions:read`
- `chat:write`
- `files:read`
- `assistant:write` (for Slack AI streaming and status)

**Event Subscriptions** (subscribe to bot events):

- `app_home_opened`
- `app_mention`
- `message.channels` (or `message.groups` / `message.im` as needed)

**Interactivity** — enable interactivity for the feedback buttons (`sheetoracle_feedback` action).

Invite the bot to your workspace and the channels where you want to use it.

### 4. Google Sheets (optional)

1. Create a Google Cloud service account with the Google Sheets API enabled.
2. Download the JSON key and set `SERVICE_ACCOUNT_PATH` in `.env`.
3. Share target spreadsheets with the service account email (Viewer access is enough).
4. Ensure `uvx` is installed (`pip install uv` or follow [uv docs](https://docs.astral.sh/uv/)).

## Running

```bash
python app.py
```

The bot connects via Socket Mode and listens for file uploads, Google Sheet URLs, and @mentions.

## Usage

1. **Start a thread** — Upload a CSV file or paste a Google Sheets URL in a Slack message.
2. **Review the profile** — SheetOracle replies with a Block Kit card showing row count and inferred column types.
3. **Ask questions** — @mention SheetOracle in that same thread, for example:
   - *What are the top 5 categories by revenue?*
   - *Plot expenses over time*
   - *Give me a high-level summary of this dataset*

SheetOracle only answers questions inside threads that already have an active database session.

## Local agent testing

`test_agent.py` runs the LangGraph pipeline against an existing session database without Slack:

```bash
# Point db_path at a file under tmp_databases/ and set OPENAI_API_KEY
python test_agent.py
```

## Security notes

- Generated SQL is validated: only single `SELECT`/`WITH` statements are allowed; mutating keywords are rejected.
- Session databases are opened in **read-only** mode during query execution.
- Queries are capped at 10,000 rows and a 20-second timeout.
- Never commit `.env`, service account JSON files, or runtime data (`tmp_*` directories) to version control.

## License

MIT — see [LICENSE](LICENSE) for details.
