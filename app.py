"""GreenScope navigation."""

from __future__ import annotations

import streamlit as st

from src.ui.assessment import render_assessment
from src.ui.common import DISCLOSURE_TEXT, inject_css
from src.ui.demo_lab import render_demo
from src.ui.navstate import register
from src.ui.overview import render_overview
from src.ui.research import render_research

st.set_page_config(
    page_title="GreenScope — Lawn Health & Service Planning",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

pages = [
    st.Page(render_overview, title="Overview", url_path="overview", default=True),
    st.Page(render_assessment, title="Lawn assessment", url_path="assessment"),
    st.Page(render_demo, title="Demo test lab", url_path="demo"),
    st.Page(render_research, title="Model results and data", url_path="results"),
]
register(pages)
navigation = st.navigation(pages)
with st.sidebar:
    st.markdown("**GreenScope**")
    st.caption("Lawn Health & Service Planning")
    st.markdown(f"<div class='gs-disclosure'>{DISCLOSURE_TEXT}</div>", unsafe_allow_html=True)
navigation.run()
