"""
RAG Pipeline Admin Dashboard
=============================
Launch with:  streamlit run admin_app.py
"""

import os
import sys
import streamlit as st

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.tabs import (
    render_etl_tab,
    render_db_tab,
    render_eval_tab,
    render_playground_tab,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Admin Dashboard", page_icon="🔬", layout="wide")
st.title("🔬 RAG Pipeline Admin Dashboard")

tab_etl, tab_db, tab_eval, tab_playground = st.tabs([
    "⚙️ ETL Pipeline",
    "🗄️ Database Explorer",
    "📊 Evaluation",
    "💬 Playground & Feedback",
])

with tab_etl:
    render_etl_tab()

with tab_db:
    render_db_tab()

with tab_eval:
    render_eval_tab()

with tab_playground:
    render_playground_tab()
