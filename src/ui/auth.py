"""Authentication module for Streamlit Admin Dashboard."""

import hmac
import streamlit as st
from src.utils.config import config


def check_password() -> bool:
    """Returns `True` if the user is authenticated, else displays the login form and stops execution."""
    auth_enabled = config.get("admin_auth.enabled", True)
    if isinstance(auth_enabled, str):
        auth_enabled = auth_enabled.lower() in ("true", "1", "yes")

    if not auth_enabled:
        return True

    if st.session_state.get("authenticated", False):
        return True

    # Render Login Card
    col_l, col_center, col_r = st.columns([1, 1.6, 1])
    with col_center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("🔒 Admin Dashboard Login")
        st.caption("Please enter your administrative credentials to access the RAG pipeline operations.")

        with st.form("admin_login_form"):
            username = st.text_input("Username", value="", placeholder="admin", key="login_username")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
            submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit:
                expected_user = config.get("admin_auth.username", "admin")
                expected_pass = config.get("admin_auth.password", "admin")

                # Timing-attack safe credential comparison
                is_user_valid = hmac.compare_digest(username.strip(), expected_user)
                is_pass_valid = hmac.compare_digest(password.strip(), expected_pass)

                if is_user_valid and is_pass_valid:
                    st.session_state["authenticated"] = True
                    st.session_state["auth_user"] = username.strip()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

    st.stop()


def render_logout_button():
    """Render a clean logout button in the sidebar when authenticated."""
    auth_enabled = config.get("admin_auth.enabled", True)
    if isinstance(auth_enabled, str):
        auth_enabled = auth_enabled.lower() in ("true", "1", "yes")

    if auth_enabled and st.session_state.get("authenticated", False):
        with st.sidebar:
            st.markdown("---")
            current_user = st.session_state.get("auth_user", "Admin")
            st.caption(f"👤 Signed in as: **{current_user}**")
            if st.button("🚪 Log Out", key="btn_admin_logout", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state.pop("auth_user", None)
                st.rerun()

