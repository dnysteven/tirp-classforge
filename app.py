import streamlit as st
from utils.ui_utils import render_sidebar

st.set_page_config(page_title="ClassForge Demo", layout="wide")
render_sidebar()

st.title("ClassForge Demo – Home")
st.write(
  """
  Use the sidebar to navigate to **Group Allocation** (native grid),
  **Ag-Grid Allocation**, or **Smart Group Allocation**.
  Upload a CSV/Excel and interactively edit the suggested groups.
  """
)
