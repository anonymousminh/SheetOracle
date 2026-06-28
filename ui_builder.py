from typing import List, Dict, Any

def build_schema_summary_card(
    file_name: str, 
    total_rows: int, 
    columns_metadata: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Constructs a native Slack Block Kit payload layout displaying 
    the indexed spreadsheet's metadata and inferred column profile.
    
    Args:
        file_name (str): The human-readable name of the original file.
        total_rows (int): Total number of rows parsed from the data grid.
        columns_metadata (List[Dict[str, str]]): A list of dicts, where each dict
            contains 'name' and 'type' keys for a parsed column.
            
    Returns:
        List[Dict[str, Any]]: A list of structured block dictionaries 
                              ready for Slack's chat API surfaces.
    """
    # 1. Initialize the root layout block list with a prominent Header
    blocks: List[Dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📊 Dataset Profile Indexed Successfully",
                "emoji": True
            }
        },
        # 2. Add a clean Context row to summarize high-level document stats
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📁 *Source File:* `{file_name}`  |  🔢 *Total Scope:* `{total_rows:,} rows`"
                }
            ]
        },
        # 3. Add a Divider line to cleanly separate metadata from column layouts
        {
            "type": "divider"
        }
    ]
    
    # 4. Construct a multi-column 'fields' list for the discovered schema layout
    # Slack allows up to 10 markdown field blocks inside a single section element.
    ui_fields: List[Dict[str, str]] = []
    
    for col in columns_metadata[:10]:  # Cap at 10 items to stay safe within single-block boundaries
        col_name: str = col.get("name", "Unknown")
        col_type: str = col.get("type", "Text")
        
        # Determine an appropriate emoji based on data classification types
        emoji: str = "🔤"
        if col_type == "Integer":
            emoji = "🔢"
        elif col_type == "Decimal/Float":
            emoji = "💵"
        elif col_type == "Date/Time":
            emoji = "📅"
            
        # Format the field layout neatly using Slack markdown syntax
        ui_fields.append({
            "type": "mrkdwn",
            "text": f"{emoji} *{col_name}*\n`Type: {col_type}`"
        })
        
    # Append the fields matrix block to the main layout tracking tree
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*🤖 Discovered Database Schema Blueprint:*"
        },
        "fields": ui_fields
    })
    
    # 5. Add a professional footer note indicating conversational readiness
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "⚡ *SheetOracle Brain Engaged:* Reply to this thread to ask analytics commands."
            }
        ]
    })
    
    return blocks