"""
SQLAlchemy Declarative Base Model
"""

from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass
