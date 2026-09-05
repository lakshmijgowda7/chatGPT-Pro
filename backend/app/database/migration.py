"""
Alembic Programmatic Migration Runner
Allows running database schema migrations during container startup, CLI maintenance, or test suites.
"""

import os
from alembic.config import Config
from alembic import command
from app.core.config import settings
from app.core.logging import logger


def get_alembic_config() -> Config:
    """Creates an Alembic Config object pointing to alembic.ini with dynamic db url."""
    # Locate alembic.ini relative to backend root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ini_path = os.path.join(base_dir, "alembic.ini")
    
    alembic_cfg = Config(ini_path)
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
    alembic_cfg.set_main_option("script_location", os.path.join(base_dir, "alembic"))
    return alembic_cfg


def run_migrations(target_revision: str = "head") -> None:
    """Upgrades the database schema to the target revision (defaults to 'head')."""
    logger.info(f"Running database migrations to revision: {target_revision}...")
    alembic_cfg = get_alembic_config()
    command.upgrade(alembic_cfg, target_revision)
    logger.info("Database migrations completed successfully.")


def rollback_migration(target_revision: str = "-1") -> None:
    """Downgrades database schema by relative steps or to target revision."""
    logger.info(f"Rolling back database migration to: {target_revision}...")
    alembic_cfg = get_alembic_config()
    command.downgrade(alembic_cfg, target_revision)
    logger.info("Database rollback completed.")
