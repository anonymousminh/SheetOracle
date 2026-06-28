import os
import sqlite3
import pandas as pd
from typing import List, Dict

def process_and_seed_csv(file_path: str, thread_ts: str) -> tuple[int, List[Dict[str, str]]]:
    """
    Ingest a CSV file, infers its column schema, and seeds it into an isolated,
    temporary SQLite database.

    Returns:
        str: A clean markdown-formatted string representing the schema, ready for the LangGraph state loop and Slack UI
    """
    df: pd.DataFrame = pd.read_csv(file_path)
    
    total_rows: int = len(df)

    # Initialize an empty list to hold column dictionaries
    columns_metadata: List[Dict[str, str]] = []
    
    # We loop over every column to inspect its inferred data type
    for col_name in df.columns:
        # Get the underlying data type determined by pandas
        pandas_type: str = str(df[col_name].dtype)
        
        # Convert pandas type descriptors to clear, friendly user schemas
        friendly_type: str = "Text"
        if "int" in pandas_type:
            friendly_type = "Integer"
        elif "float" in pandas_type:
            friendly_type = "Decimal/Float"
        elif "datetime" in pandas_type or "object" in col_name.lower() and "date" in col_name.lower():
            friendly_type = "Date/Time"
            
        # Populate the list
        columns_metadata.append({
            "name": str(col_name),
            "type": friendly_type
        })

    # Ensure our target database directory framework exists
    db_directory: str = "./tmp_databases"
    os.makedirs(db_directory, exist_ok=True)
    
    # Construct our deterministic database pathway tied to the conversation session
    target_db_path: str = f"{db_directory}/{thread_ts}.db"
    
    # Establish a clean local transaction connection hook with SQLite
    connection: sqlite3.Connection = sqlite3.connect(target_db_path)
    
    try:
        # Stream the full in-memory grid matrix directly into a read-only table 
        # named 'uploaded_data'. If it exists, overwrite it. Do not include row indexes.
        df.to_sql(
            name="uploaded_data", 
            con=connection, 
            if_exists="replace", 
            index=False
        )
        print(f"[Profiler] Successfully seeded ephemeral database warehouse at: {target_db_path}")
        
    finally:
        connection.close()

    return total_rows, columns_metadata