import datetime as dt
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text

from .db import Base


class AlertRule(Base):
    __tablename__ = "alert_rules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric: Mapped[str] = mapped_column(String(64))  # 'ESI' or component like 'A_financial'
    op: Mapped[str] = mapped_column(String(2))  # >,<,>=,<=
    threshold: Mapped[float] = mapped_column(Float)
    country: Mapped[str] = mapped_column(String(8), default="US")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)



