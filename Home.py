# app.py  – Home / Upload + Engine Launcher
import os, streamlit as st, pandas as pd

st.set_page_config(page_title="ClassForge Home", layout="wide")
st.title("ClassForge – Classroom Allocation Toolkit")

if st.session_state.pop("redirect_warning", None):
    st.warning("⚠️  Please upload a CSV before opening an allocation page.")

# ───────────────── 1. Upload (once per session) ──────────────────────
upl = st.file_uploader("Upload your student CSV once:", type="csv")

if upl:
    try:
        df_uploaded = pd.read_csv(upl)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.session_state["uploaded_df"] = df_uploaded
    st.success(f"✅ Loaded {len(df_uploaded)} rows.")
else:
    # use previous upload if still stored
    df_uploaded = st.session_state.get("uploaded_df")

# ───────────────── 2. Detect available engines ───────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ENGINES = []

ENGINES.append(("CP-SAT Model for Classroom allocation", "pages/CP_SAT_Allocation.py"))

if os.path.exists(os.path.join(MODELS_DIR, "gnn_model1.pth")):
    ENGINES.append(("GNN Model for Classroom allocation", "pages/GNN_Allocation.py"))

if os.path.exists(os.path.join(MODELS_DIR, "deep_rl_model.pth")):
    ENGINES.append(("Deep Reinforcement Learning for Classroom allocation", "pages/Deep_RL_Allocation.py"))

# ───────────────── 3. Show engine buttons after upload ───────────────
if df_uploaded is not None:
    st.markdown("## Choose an allocation engine")

    cols = st.columns(len(ENGINES))
    for (label, slug), col in zip(ENGINES, cols):
        with col:
            if st.button(label, key=slug):
                # ---- ONE line decides which API is available ----
                if hasattr(st, "switch_page"):       # Streamlit ≥ 1.25
                    st.switch_page(slug)             # slug already has .py
                else:                                # older Streamlit
                    st.experimental_set_query_params(page=slug)
                    st.experimental_rerun()
else:
    st.info("⬆️ Upload a CSV to unlock engine buttons.")

st.markdown("---")
st.write(
    """
    **Workflow**

    1. Upload your student CSV once on this page.  
    2. Click an engine button to run that allocation method.  
    3. All downstream pages reuse the same data (no re‑upload needed).
    4. To start over, reload this page and upload a new file.
    """
)
