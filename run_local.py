"""Local dev launcher — starts gateway + streamlit without Docker.

Usage:
    python run_local.py          # both services
    python run_local.py --app    # streamlit only
    python run_local.py --gateway  # gateway only

Requires: pip install aiosqlite uvicorn
Database: auto-creates SQLite file ./forecast.db (set via DATABASE_URL in .env)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Start E-Forecast locally")
    parser.add_argument("--app", action="store_true", help="Start only Streamlit app")
    parser.add_argument("--gateway", action="store_true", help="Start only Gateway API")
    args = parser.parse_args()

    start_app = args.app or not args.gateway
    start_gw = args.gateway or not args.app

    processes = []

    if start_gw:
        print("🚀 Starting Gateway on http://localhost:8000 ...")
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn",
                 "forecast.gateway.app:app",
                 "--host", "0.0.0.0", "--port", "8000",
                 "--reload", "--log-level", "info"],
            )
        )
        time.sleep(2)  # let gateway start first

    if start_app:
        print("🚀 Starting Streamlit on http://localhost:8501 ...")
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run",
                 "app.py",
                 "--server.port", "8501",
                 "--server.headless", "true"],
            )
        )

    print("\n✅ Services starting...")
    print("   Streamlit: http://localhost:8501")
    print("   Gateway:   http://localhost:8000")
    print("   Docs:      http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services.\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait(timeout=5)
        print("Done.")


if __name__ == "__main__":
    main()
