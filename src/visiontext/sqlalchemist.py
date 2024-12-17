import os
from copy import deepcopy
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    InstrumentedAttribute,
    ColumnProperty,
    Session,
    sessionmaker,
    DeclarativeBase,
)

from packg.log import logger


class DBOpener:
    def __init__(
        self,
        db_model,
        db_url: str = None,
        sqlite_db_file: str | Path | None = None,
        engine_kwargs: dict | None = None,
    ):
        """
        Initialize the database manager either with an SQLite file, or any other SQL database URL.

        Args:
            db_model: DeclarativeBase for sqlalchemy ORM
            db_url: URL for the database e.g. 'postgresql+psycopg2://user:pw@server:5432/dbname'
            sqlite_db_file: Path to the SQLite database file instead of a URL
        """
        if db_url is not None and sqlite_db_file is not None:
            raise ValueError(f"Can only provide one of {db_url=} or {sqlite_db_file=}, got both.")
        if db_url is not None:
            self.db_url: str = db_url
            self.sqlite_db_file: Path | None = None
        elif sqlite_db_file is not None:
            self.db_url: str = (f"sqlite:///{self.sqlite_db_file.as_posix()}",)
            self.sqlite_db_file: Path = Path(sqlite_db_file)
        self.db_model: DeclarativeBase = db_model
        self.engine_kwargs: dict | None = {} if engine_kwargs is None else engine_kwargs
        self.engine: Engine = None
        self.session: Session = None

    def connect(self, create_all: bool = True):
        """Connect to the SQLite database."""
        if self.sqlite_db_file is not None and not self.sqlite_db_file.is_file():
            logger.warning(f"Database file not found, recreating: {self.sqlite_db_file}")
            os.makedirs(self.sqlite_db_file.parent, exist_ok=True)
        logger.info(f"Connecting to database: {self.db_url}")
        self.engine: Engine = create_engine(
            self.db_url,
            # connect_args={"check_same_thread": False}
            **self.engine_kwargs,
        )
        self.session: Session = sessionmaker(bind=self.engine)()
        if create_all:
            self.db_model.metadata.create_all(self.engine, checkfirst=True)

        # self.connection = sqlite3.connect(self.db_file)
        # self.connection.row_factory = sqlite3.Row  # This allows us to get rows as dictionaries
        # self.cursor = self.connection.cursor()

    def close(self):
        """Close the database connection."""
        if self.session is not None:
            self.session.close()


def get_orm_class_by_table_name(base, table_name):
    for cls in base.registry.mappers:
        if cls.class_.__tablename__ == table_name:
            return cls.class_
    raise ValueError(f"No ORM class found for table name {table_name}")


def convert_sqlalchemy_orm_objects_to_dict(result: list[object]):
    """
    Convert SQLAlchemy data objects to dictionaries. These are still database objects e.g.
    medpc.pacs.database_model.Studies and not Row.

    Here, Studies is a mapped class (likely an ORM class). This means that studies[0] is an instance
    of the Studies class, which corresponds to a row in the database. The object has attributes
    that correspond to the columns in the Studies table, and you can access them as object
    attributes, like studies[0].StudyDate.
    """
    output = []
    for i, r in enumerate(result):
        try:
            r_dict = r.__dict__
        except AttributeError as e:
            raise ValueError(
                f"Object {r} type {type(r)} in list at position {i} does not have a __dict__ "
                f"attribute so it is not an ORM object as expected. If the type is "
                f"sqlalchemy.engine.row.Row then the solution is to use the function "
                f"convert_sqlalchemy_rows_to_dict instead."
            ) from e
        r_dict_new = {k: v for k, v in r_dict.items() if not k.startswith("_")}
        output.append(r_dict_new)
    return output


def convert_sqlalchemy_rows_to_dict(result: list[Row]):
    """
    Convert SQLAlchemy Row objects to dictionaries. These are results which have been completely
    resolved and fetched, so there are no more dependencies to other tables.

    In this case, the table selected from is from SQLAlchemy's Table construct (not an ORM
    class). When you query it, the result is not an instance of a mapped class but a Row object.
    The Row object is a key-value mapping of column names to values, and you access data using
    dictionary-like syntax, like row["StudyDescription"].
    """
    # return [dict(row._mapping) for row in result]  # noqa
    output = []
    for i, r in enumerate(result):
        try:
            r_dict = dict(r._mapping)  # noqa
        except AttributeError as e:
            raise ValueError(
                f"Object {r} type {type(r)} in list at position {i} does not have a _mapping "
                f"attribute so it is not a Row object as expected. If the type is an sqlalchemy "
                f"ORM object like module.database_model.TableName then the solution is to use the "
                f"function convert_sqlalchemy_orm_objects_to_dict instead."
            ) from e
        r_dict_new = {k: v for k, v in r_dict.items() if not k.startswith("_")}
        output.append(r_dict_new)
    return output


def bulk_insert_mappings_ignore_dups_singlethreaded(
    session: Session, orm_table_class, unique_id_column_str, data: list[dict]
):
    """
    insert with e.g. sqlite and ignore duplicates by checking for the ids in the table. in
    multi-write this might crash if new duplicates are inserted between select and insert

    Args:
        session: sqlalchemy session
        orm_table_class: Studies
        unique_id_column_str: "StudyInstanceUID"
        data: list of dict with each dict representing a row to insert

    Returns:

    """
    # remove duplicates in the input data
    id2data = {}
    for d in data:
        did = d[unique_id_column_str]
        if did in id2data:
            logger.warning(
                f"Duplicate input id: {unique_id_column_str}={did} found. Ignoring the second "
                f"entry.\nExisting: {id2data[did]}\nDuplicate: {d}"
            )
            continue
        id2data[did] = d
    ids = list(id2data.keys())

    # select existing ids from db and skip all input data that already exists
    column = getattr(orm_table_class, unique_id_column_str)
    existing_ids_rows = session.query(column).filter(column.in_(ids)).all()
    existing_ids = set(
        [a[unique_id_column_str] for a in convert_sqlalchemy_rows_to_dict(existing_ids_rows)]
    )
    new_data = [d for did, d in id2data.items() if did not in existing_ids]
    if len(new_data) != len(data):
        logger.warning(
            f"Ignoring {len(data) - len(new_data)} duplicates out of {len(data)} "
            f"records for table {orm_table_class.__tablename__}"
        )

    # sqlite does not give enough information on errors. catch the error and reraise with more info
    errors_to_check = (OverflowError, IntegrityError)
    try:
        session.bulk_insert_mappings(orm_table_class, new_data)
    except errors_to_check as e:
        session.rollback()
        logger.error(f"Error inserting data: {type(e).__name__} {e}")
        logger.info(f"Inserting one by one to find the problematic entry...")
        for datapoint in new_data:
            try:
                session.bulk_insert_mappings(orm_table_class, [datapoint])
            except errors_to_check as e:
                logger.error(f"Error inserting single datapoint: {type(e).__name__} {e}")
                logger.error(f"Problematic entry: {datapoint}")
                raise e
        raise RuntimeError("The error above did not reappear when inserting data 1 by 1!") from e


def bulk_insert_mappings_ignore_dups_postgresql(
    session: Session,
    orm_table_class,
    index_elements: str | list[str],
    data: list[dict],
    count_inserts: bool = False,
    dedup_input: bool = True,
):
    """
    postgresql dialect has a special insert statement that can ignore duplicates.
    with count_inserts true

    Args:
        session: sqlalchemy session
        orm_table_class: Studies
        index_elements: name of the unique id column(s)
        data: list of dict with each dict representing a row to insert
        count_inserts: if True, count and log the number of ignored duplicates.
            this is ~10-20x slower than without counting and logging.
        dedup_input: if True, remove duplicates in the input data
    Returns:

    """
    n_inputs_before_dedup = len(data)
    if isinstance(index_elements, str):
        if dedup_input:
            id2data = {d[index_elements]: d for d in data}
            n_inputs_before_dedup = len(data)
            data = list(id2data.values())
        index_elements = [index_elements]
    else:
        index_elements = list(index_elements)
        if dedup_input:
            id2data = {d[tuple(d[i] for i in index_elements)]: d for d in data}
            n_inputs_before_dedup = len(data)
            data = list(id2data.values())

    stmt = pg_insert(orm_table_class).values(data)
    # count the inserts
    stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
    if not count_inserts:
        session.execute(stmt)
        return
    orm_field = getattr(orm_table_class, index_elements[0])
    stmt = stmt.returning(orm_field)
    result = session.execute(stmt)
    num_inserts = len(list(result))
    if num_inserts < len(data):
        i_in = len(data) - n_inputs_before_dedup
        i_db = n_inputs_before_dedup - num_inserts
        logger.warning(f"Input data: {len(data)}, ignored in input: {i_in}, already in db: {i_db}")


def get_orm_column_types(orm_table_class) -> dict[str, type]:
    field2type = {}
    supported_types = set((str, int, float))
    for field, value in orm_table_class.__dict__.items():
        if field.startswith("_"):
            continue
        value: InstrumentedAttribute
        if not isinstance(value.property, ColumnProperty):
            # ignore relationships etc.
            continue
        proper: ColumnProperty = value.property
        columns = proper.columns
        assert len(columns) == 1, f"Expected one column for {field} but got {len(columns)}"
        column = columns[0]
        column_type = column.type.python_type
        if column_type not in supported_types:
            raise ValueError(f"Unsupported type {column_type} for field {field}")
        field2type[field] = column_type
    return field2type


def convert_data_types(orm_table_class, data: list[dict], str_replace_0x00: bool = False):
    """
    Convert data to the table structure. E.g. if input is int, and table field is str, convert
    to str. E.g.: studies_insert_list = convert_data_types(Studies, studies_insert_list)
    """
    field2type = get_orm_column_types(orm_table_class)
    new_data = []
    for d in data:
        dnew = deepcopy(d)
        for k in list(dnew.keys()):
            v = dnew[k]
            if v is None:
                continue
            target_type = field2type[k]
            if not isinstance(v, target_type):
                v = target_type(v)
            if str_replace_0x00 and target_type == str:
                # avoid postgresql error: ValueError: A string literal cannot contain NUL (0x00)
                v = v.replace("\x00", " ")
            dnew[k] = v
        new_data.append(dnew)
    return new_data


def _migrate_sqlite_to_postgresql_example():
    import sqlite3
    import pandas as pd

    # 1. connect to sqlite
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()

    # 2. list all existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(tables)

    Example = None  # assuming this is the ORM class for the table
    session: Session = None  # assuming this session is connected to postgresql with sqlalchemy ORM
    chunk_size = 50000
    for table_name, table_orm in [
        ("example", Example),
    ]:
        print("*" * 70)
        print("*" * 70, table_name)
        print("*" * 70)
        total = 0
        # 3. select chunks
        for chunk in pd.read_sql_query(f"SELECT * FROM {table_name}", conn, chunksize=chunk_size):
            total += len(chunk)
            print(f"Chunk with {len(chunk)} rows loaded. total {total}")
            if total < 23150000:
                continue
            records: list[dict] = chunk.to_dict(orient="records")
            # 4. convert to target column types and insert
            records_conv = convert_data_types(table_orm, records, str_replace_0x00=True)
            session.bulk_insert_mappings(table_orm, records_conv)
            session.commit()

    # conn.close()
