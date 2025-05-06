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
