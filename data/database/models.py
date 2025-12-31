"""
SQLAlchemy 2.0 models for the RDT Trading System.
"""

from datetime import datetime, date
from enum import Enum
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, Date, Boolean, Text,
    Enum as SQLEnum, Index, CheckConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    EOD = "end_of_day"


class SignalStatus(str, Enum):
    PENDING = "pending"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    IGNORED = "ignored"


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rrs_at_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(SQLEnum(TradeStatus, native_enum=False), default=TradeStatus.OPEN)
    exit_reason: Mapped[Optional[str]] = mapped_column(SQLEnum(ExitReason, native_enum=False), nullable=True)

    __table_args__ = (
        Index("ix_trades_symbol_entry_time", "symbol", "entry_time"),
        CheckConstraint("shares > 0", name="ck_trades_shares_positive"),
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True, index=True)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rrs_at_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    rrs: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(SQLEnum(SignalStatus, native_enum=False), default=SignalStatus.PENDING)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    atr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    daily_strong: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    daily_weak: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    starting_balance: Mapped[float] = mapped_column(Float, nullable=False)
    ending_balance: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winners: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True, index=True)
    added_date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
