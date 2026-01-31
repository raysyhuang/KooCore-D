"""
KooCore-D Performance Dashboard

Read-only observability dashboard that visualizes outputs/ data.
Design principles:
- Read-only: never writes to outputs/
- No model logic or learning side-effects
- Pure visualization from frozen data
"""

import streamlit as st

st.set_page_config(
    page_title="KooCore-D Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import views after page config
from views.overview import render_overview
from views.picks import render_picks
from views.phase5 import render_phase5


def main():
    st.title("📊 KooCore-D Performance Dashboard")
    
    st.sidebar.markdown("### Navigation")
    
    PAGES = {
        "📈 Overview": render_overview,
        "🎯 Daily Picks": render_picks,
        "🧠 Phase-5 Learning": render_phase5,
    }
    
    page = st.sidebar.radio("Select Page", list(PAGES.keys()), label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Dashboard Info**
        - Read-only visualization
        - Data source: `outputs/`
        - No writes, no model logic
        """
    )
    
    # Render selected page
    PAGES[page]()


if __name__ == "__main__":
    main()
