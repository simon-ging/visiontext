import re
from collections import defaultdict
from copy import deepcopy
from io import StringIO
from types import GenericAlias
from typing import get_args, get_origin

import pandas as pd
from sqlalchemy import ARRAY
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import ColumnProperty, InstrumentedAttribute, Session
from sqlalchemy.schema import Column, CreateTable

from packg.log import logger


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
            id2data = {tuple(d[i] for i in index_elements): d for d in data}
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
    """

    Args:
        orm_table_class:

    Returns:
        dict of column -> column type. the column type is either a standard python type like
        str, int, float or a type annotation like list[str], list[int], list[float]

    """
    field2type = {}
    supported_types = set((str, int, float, bool))
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
        column: Column = columns[0]
        if isinstance(column.type, ARRAY):
            array_content = column.type.item_type.python_type
            field2type[field] = list[array_content]
        elif column.type.python_type in supported_types:
            field2type[field] = column.type.python_type
        else:
            raise ValueError(f"Unsupported type {column.type} for field {field}")
    return field2type


def replace_0x00(x):
    # avoid postgresql error: ValueError: A string literal cannot contain NUL (0x00)
    if isinstance(x, str):
        return x.replace("\x00", " ")


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
            if isinstance(target_type, GenericAlias) and get_origin(target_type) == list:
                # convert given a type like list[int]
                args = get_args(target_type)
                assert len(args) == 1, f"Expected 1 arg for typed list {target_type} but got {args}"
                inner_type = args[0]
                new_list = []
                for i in v:
                    if not isinstance(i, inner_type):
                        i = inner_type(i)
                    if str_replace_0x00:
                        i = replace_0x00(i)
                    new_list.append(i)
                v = new_list
            else:
                # convert a simple type like int, str, float
                if not isinstance(v, target_type):
                    v = target_type(v)
                if str_replace_0x00:
                    v = replace_0x00(v)
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
            records: list[dict] = chunk.to_dict(orient="records")
            # 4. convert to target column types and insert
            records_conv = convert_data_types(table_orm, records, str_replace_0x00=True)
            session.bulk_insert_mappings(table_orm, records_conv)
            session.commit()

    # conn.close()


def get_table_creation_sql(db_model, engine):
    output = StringIO()
    for table in db_model.metadata.sorted_tables:
        sql = str(CreateTable(table).compile(engine))
        output.write(sql + ";\n\n")
    return output.getvalue()


def split_postgresql_db_uri(dburi: str) -> tuple[str, str, str, str, str]:
    """
    Split dburi into it's components, e.g. to give the components to psycopg2.connect()

    Args:
        dburi: postgresql+psycopg2://USER:PASS@SERVER:PORT/DBNAME

    Returns:
        tuple of (user, password, server, port, dbname)

    """
    re_db = re.compile(r"postgresql\+psycopg2://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)")
    m = re_db.match(dburi)
    assert m is not None, f"DBURI {dburi} does not match the expected format"
    user, password, server, port, dbname = m.groups()
    return user, password, server, port, dbname


def get_deletion_order(Base) -> list[str]:
    """
    Get the order in which tables should be deleted to avoid foreign key constraint violations.

    Args:
        Base: declarative_base() object

    Returns:
        list of table names in the order they should be deleted

    """

    # Get all ORM classes (tables) from Base's metadata
    orm_classes = list(Base.metadata.sorted_tables)  # Sorted tables in the metadata

    # Build dependencies graph
    table_dependencies = defaultdict(set)
    for cls in orm_classes:
        # Inspect the columns for foreign key relationships
        for column in cls.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    referenced_table = fk.column.table
                    # Add the dependent table to the list of dependencies for the referenced table
                    table_dependencies[referenced_table].add(cls)

    # Perform a topological sort on the table dependencies
    visited = set()
    deletion_order = []

    def visit(cls):
        if cls not in visited:
            visited.add(cls)
            # Recursively visit dependent tables
            for dependency in table_dependencies[cls]:
                visit(dependency)
            deletion_order.append(cls)

    # Start the visitation from all tables
    for cls in orm_classes:
        visit(cls)

    return [a.name for a in deletion_order]


def pd_read_sql_table(
    engine, base, table_name, sort_column: str | None = None, sort_ascending: bool = True
):
    table_orm = get_orm_class_by_table_name(base, table_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    for column, dtype in get_orm_column_types(table_orm).items():
        if dtype == int:
            dtype = "Int64"
        if dtype == str:
            dtype = "string"
        if dtype == list or get_origin(dtype) == list:  # list or list[int] etc.
            # pandas does not support list of int
            dtype = "object"

        # print("map", column, "to", dtype)
        df[column] = df[column].astype(dtype)
    if sort_column is not None:
        df.sort_values(sort_column, inplace=True, ascending=sort_ascending)
        df.reset_index(drop=True, inplace=True)  # sorting only works together with index reset
        # assert df["index"].tolist() == list(range(df.shape[0])), "index not contiguous"
        # for i, row in df.iterrows():
        #     assert i == row["index"], f"{i=} {row['index']=}"
    return df
