from __future__ import annotations

import os
import subprocess
import sys
import time
import argparse
from pathlib import Path


def read_secret_lines(count: int) -> list[str]:
    disabled_echo = False
    if sys.stdin.isatty():
        try:
            import termios

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            new = old[:]
            new[3] = new[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            disabled_echo = True
        except Exception:
            old = None
    else:
        old = None

    try:
        values = [sys.stdin.readline().strip() for _ in range(count)]
    finally:
        if disabled_echo and old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
            print("", flush=True)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or resume the full simulation in the background.")
    parser.add_argument("--resume", type=Path, default=None, help="Existing run directory to resume")
    parser.add_argument("--key-count", type=int, default=5, help="Number of API keys to read from stdin")
    parser.add_argument("--key-env-prefix", default="MODEL_API_KEY", help="Environment variable prefix for keys")
    parser.add_argument("--per-key-concurrency", type=int, default=100)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    log_dir = root / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    keys = read_secret_lines(args.key_count)
    if any(not key for key in keys):
        raise SystemExit(f"expected {args.key_count} API keys on stdin")

    ts = time.strftime("%Y%m%d_%H%M%S")
    label = "resume" if args.resume else "full_run"
    log = log_dir / f"{label}_{ts}.log"
    pidfile = log_dir / f"{label}_{ts}.pid"

    env = os.environ.copy()
    for index, key in enumerate(keys, start=1):
        env[f"{args.key_env_prefix}_{index}"] = key
    total_concurrency = args.key_count * args.per_key_concurrency
    env.update({
        "NUMBA_CACHE_DIR": "/tmp/numba-cache",
        "CONCURRENCY": str(total_concurrency),
        "NEWS_CONCURRENCY": str(total_concurrency),
        "EMBED_CONCURRENCY": str(total_concurrency),
        "MAX_INFLIGHT_FACTOR": "1",
    })

    stream = open(log, "ab", buffering=0)
    command = [sys.executable, "job_sim.py"]
    if args.resume:
        command += ["--resume", str(args.resume)]
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pidfile.write_text(str(process.pid) + "\n", encoding="utf-8")
    print(f"STARTED pid={process.pid} log={log} pidfile={pidfile}", flush=True)


if __name__ == "__main__":
    main()
