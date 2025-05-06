# pages/DeepRL_Allocation.py
import streamlit as st, pandas as pd
from utils.deep_rl_utils import load_model, allocate_students
from utils.cpsat_utils   import to_csv_bytes

st.set_page_config(page_title="Deep-RL Allocation", layout="wide")
st.title("🤖 Deep-RL (DQN) Classroom Allocation")

# ── 1. Get dataframe (from session or local upload) ─────────────────
if "uploaded_df" not in st.session_state:
    st.session_state["redirect_warning"] = True
    if hasattr(st, "switch_page"):
        st.switch_page("Home.py")
    else:
        st.experimental_set_query_params(page="Home.py")
        st.experimental_rerun()
else:
    df_raw = st.session_state.uploaded_df.copy()

# ── 2. Ensure a Student_Name column exists (bug‑free test) ──────────
if "Student_Name" not in df_raw.columns and {
    "First_Name", "Last_Name"
}.issubset(df_raw.columns):                         # ← fixed test
    df_raw["Student_Name"] = (
        df_raw["First_Name"].astype(str).str.strip()
        + " "
        + df_raw["Last_Name"].astype(str).str.strip()
    )

# ── 3. Parameter controls ───────────────────────────────────────────
cap_col, cls_col = st.columns(2)
capacity = cap_col.number_input("Capacity per class", 1, 40, 20, 1)
num_classrooms = cls_col.slider("Number of classrooms (≤10)", 2, 10, 10, 1)

# ── 4. Load DQN model (cached in session) ───────────────────────────
if "dqn_model" not in st.session_state:
    st.session_state.dqn_model = load_model(state_size=15, action_size=10)

with st.spinner("Allocating with DQN…"):
    assigned_df = allocate_students(
        df_raw,
        st.session_state.dqn_model,
        num_classrooms=num_classrooms,
        max_capacity=capacity,
    )

# Move key columns forward
front = ["Assigned_Classroom", "Student_ID", "Student_Name"]
assigned_df = assigned_df[front + [c for c in assigned_df.columns if c not in front]]

# ── 5. Tabs: rosters first, visualisations second ───────────────────
tab_roster, tab_vis = st.tabs(["Class rosters", "Visualisations"])

# ---------- Class rosters tab --------------------------------------
with tab_roster:
    editable = st.checkbox("Enable manual edits", value=False)
    df_edit = st.data_editor(
        assigned_df,
        disabled=not editable,
        num_rows="fixed",
        column_config={"Assigned_Classroom": st.column_config.NumberColumn(min_value=1)},
        use_container_width=True,
    )
    if editable:
        assigned_df = df_edit

    cols = st.columns(2)
    for idx, (cls, sub) in enumerate(assigned_df.groupby("Assigned_Classroom"), 1):
        with cols[(idx - 1) % 2]:
            st.markdown(f"**Class {cls}**")
            st.dataframe(
                sub[["Student_ID", "Student_Name", "Reason"]],
                hide_index=True,
                use_container_width=True,
            )

    st.download_button(
        "📥 Download allocation (CSV)",
        to_csv_bytes(assigned_df),
        file_name="deep_rl_alloc.csv",
        mime="text/csv",
    )

# ---------- Visualisations tab --------------------------------------
with tab_vis:
    st.markdown(f"#### 🏷 Number of Classrooms: `{num_classrooms}`")

    # Correctly label the bar‑chart dataframe
    counts = (
        assigned_df["Assigned_Classroom"]
        .value_counts()
        .sort_index()
        .reset_index(name="Students")
    )
    counts.columns = ["Classroom", "Students"]      # 👈 rename both cols

    st.markdown("### 📊 Students per Classroom – Bar")
    st.bar_chart(counts, x="Classroom", y="Students", use_container_width=True)

    if "Total_Score" in df_raw.columns:
        avg_df = (
            assigned_df.merge(
                df_raw[["Student_ID", "Total_Score"]],
                on="Student_ID",
                how="left",
            )
            .groupby("Assigned_Classroom", as_index=False)["Total_Score"]
            .mean()
        )
        avg_df.columns = ["Classroom", "AvgScore"]  # 👈 proper names

        st.markdown("### 📊 Average Total Score by Class")
        st.bar_chart(avg_df, x="Classroom", y="AvgScore", use_container_width=True)