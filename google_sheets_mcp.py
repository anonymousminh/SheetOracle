"""
Google Sheets ingestion via the mcp-google-sheets MCP server.

Spawns `uvx mcp-google-sheets@latest` over stdio and calls list_sheets / get_sheet_data.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

SHEET_URL_PATTERN = re.compile(
    r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)"
)

DEFAULT_ENABLED_TOOLS = "list_sheets,get_sheet_data"


class GoogleSheetsMCPError(Exception):
    """Raised when Google Sheets MCP operations fail."""


def extract_spreadsheet_id(text: str) -> Optional[str]:
    match = SHEET_URL_PATTERN.search(text or "")
    return match.group(1) if match else None


def _uvx_command() -> str:
    return os.getenv("UVX_COMMAND") or shutil.which("uvx") or "uvx"


def _mcp_server_env() -> Dict[str, str]:
    service_account_path = os.getenv("SERVICE_ACCOUNT_PATH", "").strip()
    if not service_account_path:
        raise GoogleSheetsMCPError(
            "Missing SERVICE_ACCOUNT_PATH. Add your Google service account JSON path to .env."
        )
    if not os.path.exists(service_account_path):
        raise GoogleSheetsMCPError(
            f"SERVICE_ACCOUNT_PATH does not exist: {service_account_path}"
        )

    env = os.environ.copy()
    env["SERVICE_ACCOUNT_PATH"] = service_account_path
    env["ENABLED_TOOLS"] = os.getenv("ENABLED_TOOLS", DEFAULT_ENABLED_TOOLS)

    drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    if drive_folder_id:
        env["DRIVE_FOLDER_ID"] = drive_folder_id

    return env


def _parse_tool_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    texts: List[str] = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, TextContent):
            texts.append(block.text)

    if not texts:
        return None

    combined = "\n".join(texts).strip()
    try:
        return json.loads(combined)
    except json.JSONDecodeError:
        return combined


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    server_params = StdioServerParameters(
        command=_uvx_command(),
        args=["mcp-google-sheets@latest"],
        env=_mcp_server_env(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if getattr(result, "isError", False):
                payload = _parse_tool_payload(result)
                raise GoogleSheetsMCPError(f"MCP tool {tool_name} failed: {payload}")
            return _parse_tool_payload(result)


def _sheet_names_from_list_sheets(payload: Any) -> List[str]:
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, str) for item in payload):
            return payload
        if all(isinstance(item, dict) for item in payload):
            names = [item.get("title") or item.get("name") for item in payload]
            return [name for name in names if name]
    if isinstance(payload, dict):
        sheets = payload.get("sheets") or payload.get("sheet_names") or payload.get("names")
        if isinstance(sheets, list):
            return _sheet_names_from_list_sheets(sheets)
    raise GoogleSheetsMCPError(f"Unexpected list_sheets response: {payload!r}")


def _rows_from_get_sheet_data(payload: Any) -> List[List[Any]]:
    if isinstance(payload, dict):
        values = payload.get("values")
        if isinstance(values, list):
            return values
        if "range" in payload and isinstance(payload.get("values"), list):
            return payload["values"]

    if isinstance(payload, list):
        if payload and all(isinstance(row, list) for row in payload):
            return payload

    raise GoogleSheetsMCPError(f"Unexpected get_sheet_data response: {payload!r}")


def sheet_rows_to_dataframe(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        raise GoogleSheetsMCPError("Google Sheet is empty.")

    headers = [str(cell).strip() for cell in rows[0]]
    if not any(headers):
        raise GoogleSheetsMCPError("Google Sheet header row is empty.")

    data_rows = rows[1:] if len(rows) > 1 else []
    return pd.DataFrame(data_rows, columns=headers)


async def _fetch_sheet_rows(spreadsheet_id: str, sheet_name: Optional[str] = None) -> tuple[str, List[List[Any]]]:
    resolved_sheet = sheet_name
    if not resolved_sheet:
        list_payload = await _call_mcp_tool(
            "list_sheets",
            {"spreadsheet_id": spreadsheet_id},
        )
        sheet_names = _sheet_names_from_list_sheets(list_payload)
        if not sheet_names:
            raise GoogleSheetsMCPError("Spreadsheet has no readable sheets.")
        resolved_sheet = sheet_names[0]

    data_payload = await _call_mcp_tool(
        "get_sheet_data",
        {
            "spreadsheet_id": spreadsheet_id,
            "sheet": resolved_sheet,
        },
    )
    rows = _rows_from_get_sheet_data(data_payload)
    if not rows:
        raise GoogleSheetsMCPError(f"Sheet '{resolved_sheet}' returned no data.")

    return resolved_sheet, rows


def fetch_google_sheet_dataframe(
    spreadsheet_id: str,
    sheet_name: Optional[str] = None,
) -> tuple[str, pd.DataFrame]:
    """Fetch a Google Sheet through MCP and return the sheet name and DataFrame."""
    resolved_sheet, rows = asyncio.run(_fetch_sheet_rows(spreadsheet_id, sheet_name))
    return resolved_sheet, sheet_rows_to_dataframe(rows)
