# Home.py – upload CSV ▸ choose allocation engine ▸ compare engines
import os, streamlit as st, pandas as pd
from utils.ui_utils import apply_global_styles

# ────────────────────────────────────────────────────────────────────
# 0.  AI LLM Agent  (LangChain + Ollama)
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def llm_explain(criteria: str) -> str:
    """
    Returns one sentence that begins with
    'Run this allocation model if you want to create class roster based on …'

    Falls back to a hard-coded string if Ollama / LangChain is unavailable.
    """
    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import ChatPromptTemplate

        llm = Ollama(model="mistral")   # make sure this model exists
        prompt = ChatPromptTemplate.from_template(
            "In no more than 45 words, write ONE sentence that starts exactly with "
            "'Run this allocation model if you want to create class roster based on ' "
            "followed by {criteria}. Use simple terms understood by students, teachers, "
            "and school administrators."
        )

        chain = prompt | llm          # build runnable chain
        return chain.invoke({"criteria": criteria}).strip()
    except Exception:
        # Silent fallback keeps Home page responsive even if Ollama is offline
        return f"Run this allocation model if you want to create class roster based on {criteria}"

# ────────────────────────────────────────────────────────────────────
# 1.  Upload CSV once
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ClassForge Home", layout="wide")
apply_global_styles()
st.title("ClassForge – Classroom Allocation System")

st.info(
    "Welcome to ClassForge, an online web app to create class rosters using various AI engines.\n\n"
    "Upload your student CSV file below."
)

upl = st.file_uploader("Upload student CSV:", type="csv")
if upl:
    st.session_state["uploaded_df"] = pd.read_csv(upl)
    st.success(f"✅ Loaded {len(st.session_state.uploaded_df)} rows.")

df_uploaded = st.session_state.get("uploaded_df")

# ────────────────────────────────────────────────────────────────────
# 2.  Define engine list (label, page-path, criteria)
# ────────────────────────────────────────────────────────────────────
ENGINES = [
    ("CP-SAT Model",           "pages/CP_SAT_Allocation.py",
    "academic scores, wellbeing, friendships and respect under class-size limits"),
    ("GNN Model",              "pages/GNN_Allocation.py",
    "similarity in scores, study hours, stress, bullying and safety"),
    ("Deep-RL (DQN) Model",    "pages/Deep_RL_Allocation.py",
    "balanced performance, stress, bullying and friendships sequentially"),
    ("Genetic Algorithm Model","pages/GA_Allocation.py",
    "optimising diverse student traits through evolutionary strategies"),
    ("Scenario-Based AI Model","pages/Scenario_Allocation.py",
    "simulating outcomes across academic and social grouping scenarios"),
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ────────────────────────────────────────────────────────────────────
# 3.  Show engine buttons
# ────────────────────────────────────────────────────────────────────
if df_uploaded is not None:
    st.markdown("---")
    st.markdown("## Please choose an allocation engine to create your class roster")

    ROW_LEN = 3
    for i in range(0, len(ENGINES), ROW_LEN):
        row_engines = ENGINES[i:i + ROW_LEN]
        cols = st.columns(len(row_engines))

        for (label, slug, criteria), col in zip(row_engines, cols):
            with col:
                if st.button(label, key=slug):
                    page_path = f"{slug}"
                    if hasattr(st, "switch_page"):
                        st.switch_page(page_path)
                    else:
                        st.experimental_set_query_params(page=slug)
                        st.experimental_rerun()

                st.info(llm_explain(criteria))
else:
    st.info("⬆️ Upload a CSV to unlock engine buttons.")

# ────────────────────────────────────────────────────────────────────
# 4.  Model comparer (NEW)
# ────────────────────────────────────────────────────────────────────
if df_uploaded is not None:
    st.markdown("---")
    if st.button("🔀 Compare multiple models"):
        st.switch_page("pages/Compare_Models.py")