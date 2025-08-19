import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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
