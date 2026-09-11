import os
import sqlite3
import pandas as pd
from typing import List, Dict


def _infer_columns_metadata(df: pd.DataFrame) -> List[Dict[str, str]]:
    columns_metadata: List[Dict[str, str]] = []

    for col_name in df.columns:
        pandas_type: str = str(df[col_name].dtype)

        friendly_type: str = "Text"
        if "int" in pandas_type:
            friendly_type = "Integer"
        elif "float" in pandas_type:
            friendly_type = "Decimal/Float"
        elif "datetime" in pandas_type or "object" in col_name.lower() and "date" in col_name.lower():
            friendly_type = "Date/Time"

        columns_metadata.append({
            "name": str(col_name),
            "type": friendly_type,
        })

    return columns_metadata


def process_and_seed_dataframe(df: pd.DataFrame, thread_ts: str) -> tuple[int, List[Dict[str, str]]]:
    """
    Infer column schema from a DataFrame and seed it into a per-thread SQLite database.
    """
    total_rows: int = len(df)
    columns_metadata = _infer_columns_metadata(df)

    db_directory: str = "./tmp_databases"
    os.makedirs(db_directory, exist_ok=True)

    target_db_path: str = f"{db_directory}/{thread_ts}.db"
    connection: sqlite3.Connection = sqlite3.connect(target_db_path)

    try:
        df.to_sql(
            name="uploaded_data",
            con=connection,
            if_exists="replace",
            index=False,
        )
        print(f"[Profiler] Successfully seeded ephemeral database warehouse at: {target_db_path}")
    finally:
        connection.close()

    return total_rows, columns_metadata


def process_and_seed_csv(file_path: str, thread_ts: str) -> tuple[int, List[Dict[str, str]]]:
    """
    Ingest a CSV file, infer its column schema, and seed it into an isolated,
    temporary SQLite database.
    """
    return process_and_seed_dataframe(pd.read_csv(file_path), thread_ts)
