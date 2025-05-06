# Home.py  – upload CSV ▸ choose allocation engine ▸ brief LLM blurb
import os, streamlit as st, pandas as pd
from utils.ui_utils import apply_global_styles

# ────────────────────────────────────────────────────────────────────
# 0.  LLM helper  (uses LangChain + Ollama; graceful fallback)
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def llm_explain(criteria: str) -> str:
    """
    Returns ONE short sentence that begins with:
    'Run this model if you want to create class roster based on …'
    If Ollama / LangChain is unavailable, returns the plain criteria string.
    """
    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import ChatPromptTemplate

        llm = Ollama(model="mistral")
        prompt = ChatPromptTemplate.from_template(
            "In no more than 45 words, write ONE sentence that starts exactly with "
            "'Run this allocation model if you want to create class roster based on ' "
            "followed by {criteria}, and make sure to translate the criteria to be readable for general or new user "
            "use terminology such as students, teachers and class roster"
        )
        chain = prompt | llm
        return chain.invoke({"criteria": criteria}).strip()
    except Exception:
        return f"Run this model if you want to create class roster based on {criteria}"

# ────────────────────────────────────────────────────────────────────
# 1.  Upload CSV once
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ClassForge Home", layout="wide")
apply_global_styles()
st.title("ClassForge – Classroom Allocation System")

st.info(
    """
    Welcome to ClassForge, an Online Web App to create class rosters using various AI engine.\n
    To begin, please upload your csv file containing your student data below
    """
)

upl = st.file_uploader("Upload your student CSV:", type="csv")
if upl:
    st.session_state["uploaded_df"] = pd.read_csv(upl)
    st.success(f"✅ Loaded {len(st.session_state.uploaded_df)} rows.")

df_uploaded = st.session_state.get("uploaded_df")

# ────────────────────────────────────────────────────────────────────
# 2.  Detect available engines
# ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ENGINES = [
    (
        "CP‑SAT Model",
        "CP_SAT_Allocation",
        "academic scores, wellbeing, friendships and respect under class‑size limits",
    )
]

if os.path.exists(os.path.join(MODELS_DIR, "gnn_model1.pth")):
    ENGINES.append(
        (
            "GNN Model",
            "GNN_Allocation",
            "similarity in scores, study hours, stress, bullying and safety",
        )
    )

if os.path.exists(os.path.join(MODELS_DIR, "deep_rl_model.pth")):
    ENGINES.append(
        (
            "Deep Reinforcement Learning Model",
            "Deep_RL_Allocation",
            "balanced performance, stress, bullying and friendships sequentially",
        )
    )

ENGINES.append(
    (
        "Scenario‑Based AI Model",
        "Scenario_Allocation",
        "simulating academic and social outcomes using several grouping scenarios",
    )
)

# ────────────────────────────────────────────────────────────────────
# 3.  Show engine buttons + LLM text
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
                    page_path = f"pages/{slug}.py"
                    if hasattr(st, "switch_page"):
                        st.switch_page(page_path)
                    else:
                        st.experimental_set_query_params(page=slug)
                        st.experimental_rerun()

                st.info(llm_explain(criteria))