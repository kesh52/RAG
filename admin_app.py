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

from src.ui.auth import check_password, render_logout_button
from src.ui.tabs import (
    render_guide_tab,
    render_etl_tab,
    render_db_tab,
    render_playground_tab,
    render_eval_tab,
    render_settings_tab,
)

# ---------------------------------------------------------------------------
# Page configuration & Authentication Gate
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Admin Dashboard", page_icon="🔬", layout="wide")

# Enforce authentication
check_password()
render_logout_button()

st.title("🔬 RAG Pipeline Admin Dashboard")

tab_guide, tab_etl, tab_db, tab_playground, tab_eval, tab_settings = st.tabs([
    "📖 Guide & Workflow",
    "⚙️ ETL Pipeline",
    "🗄️ Database Explorer",
    "💬 Playground & Feedback",
    "📊 Evaluation",
    "⚙️ System Settings",
])

with tab_guide:
    render_guide_tab()

with tab_etl:
    render_etl_tab()

with tab_db:
    render_db_tab()

with tab_playground:
    render_playground_tab()

with tab_eval:
    render_eval_tab()

with tab_settings:
    render_settings_tab()

