"""
Database Session Management
Provides SQLAlchemy engine, connection pooling, session factory, context managers,
and FastAPI dependencies for PostgreSQL with SQLite development/testing fallback.
"""

from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger

# Determine engine parameters based on database dialect
db_url = settings.DATABASE_URL
engine_kwargs = {
    "pool_pre_ping": True,
    "echo": settings.DB_ECHO,
}

if db_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL Connection Pooling options
    engine_kwargs.update({
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_timeout": settings.DB_POOL_TIMEOUT,
        "pool_recycle": settings.DB_POOL_RECYCLE,
    })

engine = create_engine(db_url, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def init_db() -> None:
    """
    Initializes all database tables from registered SQLAlchemy models.
    Safe to run repeatedly (uses CREATE TABLE IF NOT EXISTS).
    """
    from app.models import Base
    Base.metadata.create_all(bind=engine)


# Automatically ensure tables exist on engine initialization
try:
    init_db()
except Exception as _e:
    logger.warning(f"Auto-init db notice: {_e}")



def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a managed database session per request.
    Rolls back transaction on unhandled exception and guarantees session closure.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions outside of HTTP request lifecycle
    (e.g., background tasks, worker threads, migration scripts, CLI, tests).
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database context rolled back due to error: {e}")
        raise
    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Health check utility verifying active database connectivity using a lightweight query.
    Returns True if database is reachable, False otherwise.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection health check failed: {e}")
        return False
