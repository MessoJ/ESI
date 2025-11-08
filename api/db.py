import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError


DATABASE_URL = os.getenv("ALERTS_DATABASE_URL", "sqlite:///./esi_alerts.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    from .models import AlertRule  # noqa
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as e:
        # Avoid crashing on concurrent create when multiple workers start
        if "already exists" in str(e).lower():
            return
        raise



