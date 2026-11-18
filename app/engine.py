"""Pure publication computation.

``compute_day`` is deliberately free of I/O and locks: given one consistent
snapshot of a probe's readings and calibration certificates for a UTC day it
returns the calibrated segments above 8 C and the minutes that have readings
but no covering certificate. The service layer calls it inside a single
write transaction, so one published version always derives from one committed
state of readings and certificates.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import namedtuple

from .timeutil import day_bounds, day_to_iso, minute_to_iso

THRESHOLD_C = 8.0

Calibration = namedtuple("Calibration", ["valid_from", "valid_to", "offset_c"])
Reading = namedtuple("Reading", ["minute", "temperature_c"])


class ComputeError(RuntimeError):
    """Raised when a day cannot be computed (the publish tx rolls back)."""


def _round3(value: float) -> float:
    return round(float(value), 3)


def find_offset(calibrations: list[Calibration], minute: int) -> float | None:
    """Return the offset covering *minute* or ``None`` (certificate gap).

    Intervals are half-open ``[valid_from, valid_to)`` and non-overlapping.
    """
    starts = [c.valid_from for c in calibrations]
    idx = bisect_right(starts, minute) - 1
    if idx < 0:
        return None
    cal = calibrations[idx]
    if cal.valid_from <= minute < cal.valid_to:
        return float(cal.offset_c)
    return None


def compute_day(
    probe_id: str,
    day: int,
    readings: list[Reading],
    calibrations: list[Calibration],
) -> dict:
    """Compute the publication payload for one probe / UTC day."""
    payload, _trace = _compute(probe_id, day, readings, calibrations)
    return payload


def compute_day_with_trace(
    probe_id: str,
    day: int,
    readings: list[Reading],
    calibrations: list[Calibration],
) -> tuple[dict, list[dict]]:
    """Return ``(payload, minute_trace)`` computed from one snapshot.

    The trace is the per-minute evidence retained with each publication:
    for every minute that has a reading it records the raw value, the
    certificate offset applied (``None`` on a certificate gap), the
    calibrated value and the over-threshold judgment, exactly as computed
    from the same committed snapshot that produced the payload.
    """
    return _compute(probe_id, day, readings, calibrations)


def _compute(
    probe_id: str,
    day: int,
    readings: list[Reading],
    calibrations: list[Calibration],
) -> tuple[dict, list[dict]]:
    """Compute the publication payload for one probe / UTC day.

    Rules implemented:

    * calibrated value = raw reading + certificate offset;
    * a minute without a reading simply does not exist; a minute with a
      reading but without a covering certificate is listed as uncalibrated;
    * consecutive (difference == 1 minute) calibrated minutes strictly above
      8 C form one segment; a missing minute or an uncalibrated minute breaks
      the run (half-open ``[start, end)`` in UTC minutes).
    """
    day_start, day_end = day_bounds(day)

    # Defensive snapshot sanity checks; service layer normally prevents these.
    in_day = [r for r in readings if day_start <= r.minute < day_end]
    if any(r.minute < day_start or r.minute >= day_end for r in readings):
        raise ComputeError("reading outside the requested day reached compute_day")
    for cal in calibrations:
        if cal.valid_from >= cal.valid_to:
            raise ComputeError("certificate interval must be half-open and non-empty")

    # Non-overlap is enforced on writes, but re-check the snapshot to keep the
    # offset lookup unambiguous.
    ordered = sorted(calibrations, key=lambda c: c.valid_from)
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.valid_from < prev.valid_to:
            raise ComputeError(
                f"overlapping calibration certificates for probe {probe_id}"
            )

    by_minute = {r.minute: float(r.temperature_c) for r in in_day}
    minutes = sorted(by_minute)

    uncalibrated: list[int] = []
    calibrated: list[tuple[int, float]] = []
    trace: list[dict] = []
    for minute in minutes:
        offset = find_offset(ordered, minute)
        if offset is None:
            uncalibrated.append(minute)
            trace.append(
                {
                    "minute_utc": minute,
                    "raw_c": by_minute[minute],
                    "offset_c": None,
                    "calibrated_c": None,
                    "above_threshold": None,
                }
            )
        else:
            value = by_minute[minute] + offset
            calibrated.append((minute, value))
            trace.append(
                {
                    "minute_utc": minute,
                    "raw_c": by_minute[minute],
                    "offset_c": float(offset),
                    "calibrated_c": _round3(value),
                    # judgment uses the full-precision value (strictly > 8 C)
                    "above_threshold": value > THRESHOLD_C,
                }
            )

    segments: list[dict] = []
    run: list[tuple[int, float]] = []

    def flush() -> None:
        if not run:
            return
        start_minute = run[0][0]
        end_minute = run[-1][0] + 1  # half-open
        peak = max(v for _, v in run)
        segments.append(
            {
                "start_minute": start_minute,
                "end_minute": end_minute,
                "start_utc": minute_to_iso(start_minute),
                "end_utc": minute_to_iso(end_minute),
                "duration_minutes": end_minute - start_minute,
                "max_calibrated_c": _round3(peak),
            }
        )

    # Explicit state machine over calibrated minutes:
    # a run continues only while consecutive minutes stay strictly > 8 C;
    # any non-hot or non-contiguous minute closes the current run.
    prev_minute: int | None = None
    for minute, value in calibrated:
        above = value > THRESHOLD_C
        contiguous = prev_minute is not None and minute == prev_minute + 1
        if above and contiguous:
            run.append((minute, value))
        else:
            flush()
            run = [(minute, value)] if above else []
        prev_minute = minute
    flush()

    return {
        "probe_id": probe_id,
        "day": day,
        "day_utc": day_to_iso(day),
        "threshold_c": THRESHOLD_C,
        "total_reading_minutes": len(minutes),
        "calibrated_minutes": len(calibrated),
        "uncalibrated_minutes": uncalibrated,
        "uncalibrated_minutes_utc": [minute_to_iso(m) for m in uncalibrated],
        "segments": segments,
    }, trace


# --------------------------------------------------------------------------- #
# Version-to-version difference tracing
# --------------------------------------------------------------------------- #
#: change_type values used by :func:`diff_traces`
CHANGE_NONE = "none"
CHANGE_READING = "reading"          # raw reading changed (incl. appeared/removed)
CHANGE_CERTIFICATE = "certificate"  # applicable certificate offset changed
CHANGE_BOTH = "both"                # reading and certificate changed together

_TOL = 1e-9


def _num_eq(a, b) -> bool:
    """None-aware numeric equality with a small float tolerance."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= _TOL


def _side(row: dict | None) -> dict | None:
    """One version's per-minute view; ``None`` means no reading that minute."""
    if row is None:
        return None
    return {
        "raw_c": row["raw_c"],
        "offset_c": row["offset_c"],
        "calibrated_c": row["calibrated_c"],
        "above_threshold": row["above_threshold"],
    }


def diff_traces(trace_from: list[dict], trace_to: list[dict]) -> dict:
    """Minute-by-minute diff of two retained evidence traces.

    Both traces must come from publications of the same probe / UTC day.
    For every minute present in either trace the diff reports both sides
    (raw / offset / calibrated / over-threshold) and classifies the change
    as ``reading`` (raw value changed, incl. a reading appearing or
    disappearing), ``certificate`` (the applicable offset changed) or
    ``both``. A certificate change is only attributed when both versions
    have a reading at that minute — a missing reading is itself the
    reading change. Consecutive changed minutes of one type are merged
    into half-open ``changed_periods``.
    """
    by_from = {r["minute_utc"]: r for r in trace_from}
    by_to = {r["minute_utc"]: r for r in trace_to}

    minutes: list[dict] = []
    for minute in sorted(set(by_from) | set(by_to)):
        a = by_from.get(minute)
        b = by_to.get(minute)
        reading_changed = (a is None) != (b is None) or (
            a is not None
            and b is not None
            and not _num_eq(a["raw_c"], b["raw_c"])
        )
        certificate_changed = (
            a is not None
            and b is not None
            and not _num_eq(a["offset_c"], b["offset_c"])
        )
        if reading_changed and certificate_changed:
            change_type = CHANGE_BOTH
        elif reading_changed:
            change_type = CHANGE_READING
        elif certificate_changed:
            change_type = CHANGE_CERTIFICATE
        else:
            change_type = CHANGE_NONE

        above_a = a["above_threshold"] if a else None
        above_b = b["above_threshold"] if b else None
        cal_a = a["calibrated_c"] if a else None
        cal_b = b["calibrated_c"] if b else None
        minutes.append(
            {
                "minute_utc": minute,
                "minute_iso": minute_to_iso(minute),
                "from": _side(a),
                "to": _side(b),
                "reading_changed": reading_changed,
                "certificate_changed": certificate_changed,
                "calibrated_changed": not _num_eq(cal_a, cal_b),
                "status_changed": above_a != above_b,
                "change_type": change_type,
            }
        )

    # Merge consecutive changed minutes of one type into half-open periods.
    changed_periods: list[dict] = []
    current: dict | None = None
    for row in minutes:
        if row["change_type"] == CHANGE_NONE:
            continue
        if (
            current is not None
            and current["change_type"] == row["change_type"]
            and row["minute_utc"] == current["end_minute"]
        ):
            current["end_minute"] += 1
            current["minutes"] += 1
            continue
        if current is not None:
            changed_periods.append(current)
        current = {
            "change_type": row["change_type"],
            "start_minute": row["minute_utc"],
            "end_minute": row["minute_utc"] + 1,
            "minutes": 1,
        }
    if current is not None:
        changed_periods.append(current)
    for period in changed_periods:
        period["start_utc"] = minute_to_iso(period["start_minute"])
        period["end_utc"] = minute_to_iso(period["end_minute"])
        period["duration_minutes"] = period["end_minute"] - period["start_minute"]

    summary = {
        "minutes_compared": len(minutes),
        "minutes_changed": sum(1 for r in minutes if r["change_type"] != CHANGE_NONE),
        "reading_changes": sum(
            1 for r in minutes if r["change_type"] == CHANGE_READING
        ),
        "certificate_changes": sum(
            1 for r in minutes if r["change_type"] == CHANGE_CERTIFICATE
        ),
        "both_changes": sum(1 for r in minutes if r["change_type"] == CHANGE_BOTH),
        "status_changes": sum(1 for r in minutes if r["status_changed"]),
    }
    return {
        "summary": summary,
        "changed_periods": changed_periods,
        "minutes": minutes,
    }
