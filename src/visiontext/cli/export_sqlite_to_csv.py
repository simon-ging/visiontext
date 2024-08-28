import os

import pandas as pd
import sqlite3
from attrs import define
from loguru import logger
from pathlib import Path

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import VerboseQuietArgs, add_argument, TypedParser


@define
class Args(VerboseQuietArgs):
    sqlite_file: Path = add_argument(positional=True, help="sqlite file")
    run_export: bool = add_argument(action="store_true", help="Run the export", shortcut="-e")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    conn = sqlite3.connect(args.sqlite_file)

    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables in the database:")
    print(tables)

    # For each table, display the first few rows
    for table_name in tables["name"]:
        print(f"\nTable: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
        print(df)
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Count: {count}")

    if args.run_export:
        export_dir = Path(f"{args.sqlite_file}_export")
        if export_dir.is_dir():
            logger.warning(f"Export directory already exists: {export_dir}")
        os.makedirs(export_dir, exist_ok=True)
        for table_name in tables["name"]:
            print(f"\nTable: {table_name}")
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(df.shape)
            csv_file = export_dir / f"{table_name}.csv"
            df.to_csv(csv_file, sep=",", index=False)
            print(f"Exported to: {csv_file}")

    conn.close()


if __name__ == "__main__":
    main()
