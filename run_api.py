"""
FastAPI Server Launcher
=======================
Launch with:
    python3 run_api.py
or:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uvicorn

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import config


def main():
    host = config.get("api.host", "0.0.0.0")
    port = int(config.get("api.port", 8000))
    reload = config.get("api.reload", True)

    print(f"🚀 Starting RAG FastAPI Chat Server on http://{host}:{port}")
    print(f"📖 OpenAPI Swagger Documentation at http://{host}:{port}/docs")

    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

