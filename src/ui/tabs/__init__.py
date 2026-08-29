"""Tabs package for the RAG Admin Dashboard."""

from src.ui.tabs.tab_etl import render_etl_tab
from src.ui.tabs.tab_db import render_db_tab
from src.ui.tabs.tab_eval import render_eval_tab
from src.ui.tabs.tab_playground import render_playground_tab

__all__ = [
    "render_etl_tab",
    "render_db_tab",
    "render_eval_tab",
    "render_playground_tab",
]

