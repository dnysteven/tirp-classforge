# utils/ui_utils.py
import streamlit as st

def apply_global_styles():
  """Injects shared CSS styles for all pages."""
  st.markdown(
    """
    <style>
    /* Tighten spacing between paragraphs in st.info(), st.success(), etc. */
    .stAlert p {
      margin-bottom: 0.8rem;   /* default is ~1rem */
    }
    </style>
    """,
    unsafe_allow_html=True,
  )

def render_footer():
    st.markdown(
        """
        <style>
            div.st-emotion-cache-uf99v8 {
                padding-bottom: 3rem;  /* prevent overlap with footer */
            }
            .footer-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                display: flex;
                justify-content: center;
                z-index: 9999;
                width: 100%;
                background-color: #0E1117;
            }
            .footer-content {
                max-width: 1200px;  /* match Streamlit's main content width */
                margin: 0 auto;
                text-align: center;
                font-size: 0.9em;
                color: #888;
                padding: 0.5em 1em;
            }
        </style>
        <div class="footer-container">
            <div class="footer-content">
              ClassForge &middot; AI-Powered Classroom Allocation System &middot; © 2025
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
