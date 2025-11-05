# main.py
import os
import requests
import datetime
import math
import logging
from typing import Optional

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import (create_engine, Column, Integer, String, Float,
                        DateTime, Boolean)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# -------------------------
# Configuration (via ENV)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tradingviewbot:nmStBvlf2QhKPyYscxBSSoaw2UwXIiIa@dpg-d45hrd95pdvs73c2p8kg-a/tradingviewsignals")
PRICE_API = os.getenv("PRICE_API", "binance")  # 'binance' supported by default
REPORT_DAILY_CRON = os.getenv("REPORT_DAILY_CRON", "0 0 * * *")
REPORT_WEEKLY_CRON = os.getenv("REPORT_WEEKLY_CRON", "0 0 * * 0")


if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. The app will still run, but notifications will fail.")

# -------------------------
# DB setup (SQLAlchemy)
# -------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String, index=True)
    side = Column(String)  # BUY / SELL / CLOSE_SELL / CLOSE_BUY / EXIT
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    resolved = Column(Boolean, default=False)  # whether closed or manually resolved
    closed_price = Column(Float, nullable=True)
    closed_time = Column(DateTime, nullable=True)
    notes = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)

# -------------------------
# FastAPI Init
# -------------------------
app = FastAPI(title="TradingView → Telegram Signal Tracker")

# -------------------------
# Helpers
# -------------------------
def send_telegram_message(text: str):
    """Send message via Telegram bot. Non-blocking caller should run in background."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials not set. Can't send message.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logging.exception("Failed to send telegram message: %s", e)
        return False


def fetch_price_binance(symbol: str) -> Optional[float]:
    """Try Binance public /api/v3/ticker/price (symbol must be like BTCUSDT)."""
    try:
        s = symbol.replace("/", "").replace("-", "").upper()
        # Binance uses symbols like BTCUSDT, ETHUSDT, etc
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={s}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            j = r.json()
            return float(j.get("price"))
    except Exception:
        logging.exception("Binance price fetch failed for %s", symbol)
    return None


def get_current_price(symbol: str) -> Optional[float]:
    """Return current price using configured PRICE_API (only binance implemented)."""
    if PRICE_API == "binance":
        return fetch_price_binance(symbol)
    # future providers or fallback can be added
    return None


def compute_pnl_percent(side: str, entry: float, current: float) -> Optional[float]:
    """Compute percentage PnL given side 'BUY' or 'SELL'."""
    if not entry or not current:
        return None
    try:
        if side.upper() == "BUY":
            return ((current - entry) / entry) * 100.0
        elif side.upper() == "SELL":
            return ((entry - current) / entry) * 100.0
    except Exception:
        logging.exception("Error computing pnl")
    return None


# -------------------------
# Request model for webhook
# -------------------------
class TVWebhook(BaseModel):
    pair: Optional[str] = None  # or "symbol"
    symbol: Optional[str] = None
    signal: str
    price: float
    stop_loss: Optional[float] = None
    timestamp: Optional[str] = None  # tradingview timenow
    interval: Optional[str] = None


# -------------------------
# Webhook endpoint
# -------------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint to receive TradingView webhook JSON.
    Expect payload like: {"pair":"BTCUSDT","signal":"BUY","price":68000,"stop_loss":67500,"timestamp":"..."}
    """
    payload = await request.json()
    # Accept multiple field names for compatibility
    pair = payload.get("pair") or payload.get("symbol") or payload.get("ticker") or payload.get("symbol_full") or payload.get("s")
    signal = payload.get("signal") or payload.get("alert_message") or payload.get("alert") or "UNKNOWN"
    price = payload.get("price") or payload.get("close") or None
    stop_loss = payload.get("stop_loss") or payload.get("sl") or None
    interval = payload.get("interval") or payload.get("timeframe") or payload.get("resolution") or None
    timestamp = payload.get("timestamp") or payload.get("time") or payload.get("timenow") or datetime.datetime.utcnow().isoformat()

    # Normalise
    if isinstance(price, str):
        try:
            price = float(price)
        except:
            price = None
    if isinstance(stop_loss, str):
        try:
            stop_loss = float(stop_loss)
        except:
            stop_loss = None

    db = SessionLocal()
    try:
        s = Signal(
            pair=str(pair).upper() if pair else "UNKNOWN",
            side=str(signal).upper(),
            entry_price=float(price) if price else None,
            stop_loss=float(stop_loss) if stop_loss else None,
            timestamp=datetime.datetime.utcnow(),
            resolved=False
        )
        db.add(s)
        db.commit()
        db.refresh(s)

        # Send instant Telegram notification (background)
        def notify(sig_id: int):
            cur = SessionLocal()
            try:
                record = cur.query(Signal).filter(Signal.id == sig_id).first()
                price_display = f"{record.entry_price:.8g}" if record.entry_price is not None else "N/A"
                sl_display = f"{record.stop_loss:.8g}" if record.stop_loss is not None else "N/A"
                interval_text = interval or "N/A"
                msg = (
                    f"📡 *New Signal Received*\n\n"
                    f"🪙 *Pair:* {record.pair}\n"
                    f"⚡ *Signal:* {record.side}\n"
                    f"⏱ *Interval:* {interval_text}\n"
                    f"💰 *Entry Price:* {price_display}\n"
                    f"🛡 *Stop Loss:* {sl_display}\n"
                    f"🕒 *Time:* {timestamp}\n"
                )
                send_telegram_message(msg)
            finally:
                cur.close()

        background_tasks.add_task(notify, s.id)

        return {"status": "ok", "stored_id": s.id}
    except Exception as e:
        logging.exception("Error storing signal")
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


# -------------------------
# Reports generation
# -------------------------
def generate_report(period: str = "daily"):
    """
    period: 'daily' or 'weekly'
    This function:
    - select signals from time window (last 24h for daily, last 7 days for weekly)
    - compute current P/L for open signals & for closed signals compute closed P/L
    - create summary message and send to Telegram
    """
    db = SessionLocal()
    try:
        now = datetime.datetime.utcnow()
        if period == "daily":
            since = now - datetime.timedelta(days=1)
            label = now.strftime("%Y-%m-%d")
        else:
            since = now - datetime.timedelta(days=7)
            week_start = (now - datetime.timedelta(days=now.weekday())).date()
            label = f"Week of {week_start.isoformat()}"

        # Query signals in window
        records = db.query(Signal).filter(Signal.timestamp >= since).all()

        total = len(records)
        profitable = 0
        losing = 0
        total_pct = 0.0
        details = []

        for rec in records:
            entry = rec.entry_price
            side = rec.side.upper() if rec.side else "UNKNOWN"

            # determine price to compute PnL: if closed use closed_price else fetch current
            current_price = rec.closed_price if rec.closed_price is not None else get_current_price(rec.pair)
            pnl_pct = None
            if entry and current_price is not None and side in ("BUY", "SELL"):
                pnl_pct = compute_pnl_percent(side, entry, current_price)
                if pnl_pct is not None:
                    total_pct += pnl_pct
                    if pnl_pct >= 0:
                        profitable += 1
                    else:
                        losing += 1
            details.append((rec.pair, side, entry, rec.stop_loss, current_price, pnl_pct))

        overall = (total_pct) if total > 0 else 0.0
        avg = overall / total if total > 0 else 0.0

        # Build message
        header = f"📈 *{period.title()} Signal Report* — {label}\n\n"
        summary = f"Total Signals: {total}\nProfitable: {profitable}\nLosing: {losing}\nAverage P/L across signals: {avg:.2f}%\n\n"
        msg = header + summary

        # include top performers (sorted by pnl)
        details_with_pnl = [d for d in details if d[5] is not None]
        details_with_pnl.sort(key=lambda x: x[5], reverse=True)
        top_n = details_with_pnl[:5]
        if top_n:
            msg += "*Top performers:*\n"
            for p, side, entry, sl, curp, pnl in top_n:
                msg += f"- {p} {side}: {pnl:.2f}% (entry {entry}, now {curp})\n"
            msg += "\n"

        # worst performers
        worst = details_with_pnl[-5:]
        if worst:
            msg += "*Worst performers:*\n"
            for p, side, entry, sl, curp, pnl in reversed(worst):
                msg += f"- {p} {side}: {pnl:.2f}% (entry {entry}, now {curp})\n"
            msg += "\n"

        # optional include number of unresolved open trades
        open_trades = [r for r in records if not r.resolved]
        msg += f"Open trades in window: {len(open_trades)}\n"

        send_telegram_message(msg)
        return msg
    finally:
        db.close()


# -------------------------
# HTTP routes to trigger reports manually
# -------------------------
@app.get("/reports/daily")
async def report_daily():
    msg = generate_report("daily")
    return {"status": "ok", "report": msg}


@app.get("/reports/weekly")
async def report_weekly():
    msg = generate_report("weekly")
    return {"status": "ok", "report": msg}


# -------------------------
# Scheduler setup
# -------------------------
scheduler = AsyncIOScheduler(timezone="UTC")


@app.on_event("startup")
def startup_event():
    # schedule daily and weekly reports using cron strings
    try:
        # If using default CRON env values above, convert them into CronTrigger
        # REPORT_DAILY_CRON default "0 0 * * *" -> minute hour dom month dow
        daily_parts = REPORT_DAILY_CRON.strip().split()
        if len(daily_parts) == 5:
            minute, hour, dom, month, dow = daily_parts
            scheduler.add_job(lambda: generate_report("daily"),
                              trigger=CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow, timezone="UTC"),
                              id="daily_report", replace_existing=True)
        weekly_parts = REPORT_WEEKLY_CRON.strip().split()
        if len(weekly_parts) == 5:
            minute, hour, dom, month, dow = weekly_parts
            scheduler.add_job(lambda: generate_report("weekly"),
                              trigger=CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow, timezone="UTC"),
                              id="weekly_report", replace_existing=True)

        scheduler.start()
        logging.info("Scheduler started (UTC).")
    except Exception:
        logging.exception("Failed to start scheduler")


@app.on_event("shutdown")
def shutdown_event():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        logging.exception("Error shutting down scheduler")
