import streamlit as st

_TW = '<link href="https://cdn.jsdelivr.net/npm/tailwindcss@3.4.4/dist/tailwind.min.css" rel="stylesheet">'
_HIDE = "[data-testid=stSidebarNav],[data-testid=stSidebarCollapseControl]{display:none!important}"
_CSS  = ".cf-sidebar{@apply p-4 h-full bg-gray-800 text-white;} .cf-link{@apply block px-3 py-2 rounded transition hover:bg-gray-700;} .cf-brand{@apply text-xl font-semibold mb-4;}"

def _inject_once():
  if "_tailwind" not in st.session_state:
    st.markdown(_TW + f"<style>{_HIDE}{_CSS}</style>", unsafe_allow_html=True)
    st.session_state["_tailwind"] = True

def render_sidebar():
  _inject_once()
  st.sidebar.markdown("""
  <div class="cf-sidebar">
    <h2 class="cf-brand">ClassForge</h2>
    <ul class="space-y-2">
      <li><a href="/" class="cf-link">Home</a></li>
      <li><a href="/gnn_group_allocation" class="cf-link">GNN Allocation</a></li>
    </ul>
  </div>
  """, unsafe_allow_html=True)