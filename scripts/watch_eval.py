#!/usr/bin/env python3
"""Watch evaluation logs for errors or early-stop triggers.

Usage: .venv/bin/python scripts/watch_eval.py
"""
import time
from pathlib import Path
from datetime import datetime

BASE = Path("experiments/results/round2")
EVAL_LOG = BASE / "eval.log"
STATUS = BASE / "status.txt"
ALERT_LOG = BASE / "monitor_alerts.log"

KEYWORDS = [
    "traceback",
    "error",
    "exception",
    "earlystoperror",
    "early stopping",
    "early stop",
    "cannot convert float",
    "kernel panic",
    "cuda out of memory",
    "out of memory",
    "fatal",
]


def check_line(line: str) -> bool:
    t = line.lower()
    return any(k in t for k in KEYWORDS)


def tail_file(path: Path, pos: int):
    if not path.exists():
        return pos, []
    with path.open("r", errors="ignore") as f:
        f.seek(pos)
        lines = f.read().splitlines()
        pos = f.tell()
    return pos, lines


def alert(msg: str):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    line = f"[{ts}] ALERT: {msg}"
    print(line)
    with ALERT_LOG.open("a") as f:
        f.write(line + "\n")


def monitor():
    eval_pos = 0
    status_pos = 0
    print("Starting evaluation monitor — watching:", EVAL_LOG, STATUS)
    while True:
        eval_pos, lines = tail_file(EVAL_LOG, eval_pos)
        for ln in lines:
            if check_line(ln):
                alert(f"log match -> {ln[:200]}")

        # also check the status file for early-stop indicators
        status_pos, s_lines = tail_file(STATUS, status_pos)
        for ln in s_lines:
            if check_line(ln):
                alert(f"status match -> {ln[:200]}")

        time.sleep(6)


if __name__ == "__main__":
    try:
        BASE.mkdir(parents=True, exist_ok=True)
        monitor()
    except KeyboardInterrupt:
        print("Monitor stopped by user.")