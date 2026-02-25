"""
Signal Quality Metrics Tracker

Tracks counters and rates for signal quality over time and persists them
to a JSON file.  Deliberately lightweight — no database dependency.

Pattern: file-based persistence with fcntl.flock, matching the project's
signal file conventions in active_signals.json / signal_history.json.
"""

import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Default metrics file location (mirrors active_signals.json convention)
# ---------------------------------------------------------------------------
_DEFAULT_METRICS_FILE = Path("data/signals/signal_metrics.json")


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


class SignalMetricsTracker:
    """
    Lightweight in-process tracker for signal quality metrics.

    Counters are accumulated in memory and flushed to a JSON file after
    every call to ``record_scan`` so that the API can read them at any time.

    Thread/process safety: file writes use fcntl.flock (shared-lock for
    reads, exclusive-lock for writes) — the same pattern used by
    realtime_scanner.py for active_signals.json.

    Hit-rate tracking
    -----------------
    A "hit" is recorded when the caller explicitly calls
    ``record_outcome(signal_id, hit=True|False)``.  The tracker does *not*
    automatically monitor prices; outcome registration must be driven by
    external logic (e.g. a price-check background task).  This keeps the
    class dependency-free and easy to test.
    """

    def __init__(self, metrics_file: Optional[Path] = None):
        self._metrics_file: Path = Path(metrics_file) if metrics_file else _DEFAULT_METRICS_FILE
        self._metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # ---- in-memory state loaded from / persisted to disk ----
        self._data: Dict = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_scan(
        self,
        signals: List[Dict],
    ) -> None:
        """
        Record aggregate statistics for one completed scanner run.

        Args:
            signals: The list of signal dicts produced by ``save_signals``
                     (after quality validation has been applied).
                     Each signal must contain at minimum:
                       - ``direction``   : 'long' | 'short'
                       - ``rrs``         : float
                       - ``warnings``    : list[str]  (may be absent / empty)
        """
        if not signals:
            self._data["scans_with_no_signals"] = (
                self._data.get("scans_with_no_signals", 0) + 1
            )
            self._data["total_scans"] = self._data.get("total_scans", 0) + 1
            self._data["last_scan_at"] = _utc_now_iso()
            self._flush()
            return

        total = len(signals)
        longs = sum(1 for s in signals if s.get("direction") == "long")
        shorts = total - longs

        # A signal is "flagged" if the validator attached any warnings
        flagged = sum(
            1 for s in signals if s.get("warnings") or _has_warning_keys(s)
        )
        clean = total - flagged

        rrs_values = [abs(s["rrs"]) for s in signals if isinstance(s.get("rrs"), (int, float))]
        avg_rrs = sum(rrs_values) / len(rrs_values) if rrs_values else 0.0

        # confidence field is optional — not all signals carry it
        conf_values = [
            s["confidence"]
            for s in signals
            if isinstance(s.get("confidence"), (int, float))
        ]
        avg_confidence = sum(conf_values) / len(conf_values) if conf_values else None

        # ---- update rolling counters ----
        d = self._data
        d["total_scans"] = d.get("total_scans", 0) + 1
        d["total_signals"] = d.get("total_signals", 0) + total
        d["total_long_signals"] = d.get("total_long_signals", 0) + longs
        d["total_short_signals"] = d.get("total_short_signals", 0) + shorts
        d["total_clean_signals"] = d.get("total_clean_signals", 0) + clean
        d["total_flagged_signals"] = d.get("total_flagged_signals", 0) + flagged
        d["scans_with_no_signals"] = d.get("scans_with_no_signals", 0)

        # running sum for avg_rrs (compute as ratio later)
        d["_rrs_sum"] = d.get("_rrs_sum", 0.0) + (avg_rrs * total)
        d["_rrs_count"] = d.get("_rrs_count", 0) + total

        if avg_confidence is not None:
            d["_conf_sum"] = d.get("_conf_sum", 0.0) + (avg_confidence * len(conf_values))
            d["_conf_count"] = d.get("_conf_count", 0) + len(conf_values)

        # last scan snapshot
        d["last_scan_at"] = _utc_now_iso()
        d["last_scan_signal_count"] = total
        d["last_scan_avg_rrs"] = round(avg_rrs, 4)
        d["last_scan_flagged"] = flagged

        self._flush()
        logger.debug(
            f"[SignalMetrics] scan recorded: total={total} long={longs} "
            f"short={shorts} flagged={flagged} avg_rrs={avg_rrs:.2f}"
        )

    def record_outcome(self, signal_id: str, hit: bool) -> None:
        """
        Register whether a signal's target was reached (hit) or stopped out.

        Args:
            signal_id: An identifier for the signal (e.g. ``symbol + generated_at``).
            hit:       True  → price reached target before stop.
                       False → price hit the stop before target.
        """
        d = self._data
        d["total_outcomes"] = d.get("total_outcomes", 0) + 1
        if hit:
            d["target_hits"] = d.get("target_hits", 0) + 1
        else:
            d["stop_outs"] = d.get("stop_outs", 0) + 1
        self._flush()
        logger.debug(f"[SignalMetrics] outcome recorded: id={signal_id} hit={hit}")

    def get_summary(self) -> Dict:
        """
        Return a snapshot of all current metrics.

        Hit-rate and false-positive-rate are computed on the fly from the
        stored counters so they are always up-to-date without needing a
        separate update step.
        """
        d = self._data
        total_signals = d.get("total_signals", 0)
        total_flagged = d.get("total_flagged_signals", 0)
        total_clean = d.get("total_clean_signals", 0)
        total_outcomes = d.get("total_outcomes", 0)
        target_hits = d.get("target_hits", 0)
        stop_outs = d.get("stop_outs", 0)

        # Hit rate: targets reached / total resolved outcomes
        hit_rate = (target_hits / total_outcomes) if total_outcomes > 0 else None

        # False-positive rate: flagged (warned) / total signals
        false_positive_rate = (
            (total_flagged / total_signals) if total_signals > 0 else None
        )

        # Average RRS across all recorded signals
        rrs_count = d.get("_rrs_count", 0)
        avg_rrs_all_time = (
            round(d.get("_rrs_sum", 0.0) / rrs_count, 4) if rrs_count > 0 else None
        )

        # Average confidence (only present if signals carried it)
        conf_count = d.get("_conf_count", 0)
        avg_confidence_all_time = (
            round(d.get("_conf_sum", 0.0) / conf_count, 4) if conf_count > 0 else None
        )

        return {
            "total_scans": d.get("total_scans", 0),
            "scans_with_no_signals": d.get("scans_with_no_signals", 0),
            "total_signals": total_signals,
            "total_long_signals": d.get("total_long_signals", 0),
            "total_short_signals": d.get("total_short_signals", 0),
            "total_clean_signals": total_clean,
            "total_flagged_signals": total_flagged,
            "false_positive_rate": (
                round(false_positive_rate, 4) if false_positive_rate is not None else None
            ),
            "total_outcomes": total_outcomes,
            "target_hits": target_hits,
            "stop_outs": stop_outs,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "avg_rrs_all_time": avg_rrs_all_time,
            "avg_confidence_all_time": avg_confidence_all_time,
            "last_scan_at": d.get("last_scan_at"),
            "last_scan_signal_count": d.get("last_scan_signal_count", 0),
            "last_scan_avg_rrs": d.get("last_scan_avg_rrs"),
            "last_scan_flagged": d.get("last_scan_flagged", 0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict:
        """Load metrics from disk, returning an empty dict on any failure."""
        if not self._metrics_file.exists():
            return {}
        try:
            with open(self._metrics_file, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"[SignalMetrics] could not load metrics file: {exc}")
        return {}

    def _flush(self) -> None:
        """Atomically persist current metrics to disk."""
        metrics_dir = self._metrics_file.parent
        temp_fd, temp_path = tempfile.mkstemp(dir=str(metrics_dir), suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(self._data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(temp_path, str(self._metrics_file))  # atomic on POSIX
        except Exception as exc:
            logger.error(f"[SignalMetrics] failed to flush metrics: {exc}")
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton (imported by the scanner and the API route)
# ---------------------------------------------------------------------------

_tracker: Optional[SignalMetricsTracker] = None


def get_metrics_tracker() -> SignalMetricsTracker:
    """Return the shared SignalMetricsTracker instance (created on first call)."""
    global _tracker
    if _tracker is None:
        _tracker = SignalMetricsTracker()
    return _tracker


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _has_warning_keys(signal: Dict) -> bool:
    """
    Detect signals that carry individual warning keys set by signal_validator.py
    (e.g. ``gap_warning``, ``low_volume_warning``) rather than a ``warnings``
    list, for backward compatibility.
    """
    warning_keys = {
        "gap_warning",
        "low_volume_warning",
        "wide_spread_warning",
        "rrs_outlier_warning",
        "price_sanity_error",
    }
    return any(signal.get(k) for k in warning_keys)
