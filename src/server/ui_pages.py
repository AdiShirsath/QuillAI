"""
src/api/ui_pages.py
────────────────────────────────────────────────────────────────
All UI page routes in one place.
Every route reads its HTML from  src/api/static/<name>.html
so you can edit pages without touching application logic.

Routes registered:
  GET /            → static/home.html
  GET /docs        → static/docs.html   (custom dark Swagger UI)
  GET /dashboard   → static/dashboard.html
  GET /ws-test     → static/ws_test.html
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

# All static HTML files live next to this file, inside  static/
STATIC_DIR = Path(__file__).parent / "static"

ui_router = APIRouter()


def _serve(filename: str) -> HTMLResponse:
    """Read an HTML file from the static directory and return it."""
    path = STATIC_DIR / filename
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return HTMLResponse(
        f"<h2>404 — {filename} not found.</h2>"
        f"<p>Place it at <code>src/api/static/{filename}</code></p>",
        status_code=404,
    )


@ui_router.get("/", include_in_schema=False)
async def home():
    """Home page — API overview, all endpoints, live stats."""
    return _serve("home.html")


@ui_router.get("/docs", include_in_schema=False)
async def docs():
    """Custom dark-themed Swagger UI."""
    return _serve("docs.html")


@ui_router.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Real-time agent dashboard."""
    return _serve("dashboard.html")


@ui_router.get("/ws-test", include_in_schema=False)
async def ws_test():
    """Terminal-style WebSocket tester."""
    return _serve("ws_test.html")
