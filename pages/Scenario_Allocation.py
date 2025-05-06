# Scenario_Allocation.py
# ⬇️  ORIGINAL COMMENTS ARE PRESERVED; only minimal wiring added

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

from utils.scenario_utils import (
    run_scenario_clustering,
    allocate_to_classrooms,
    SCENARIO_FEATURES,
)
from utils.cpsat_utils import to_csv_bytes  # reuse existing helper
from utils.ui_utils import apply_global_styles

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- PAGE SETUP --------------------------
st.set_page_config(page_title="ClassForge Scenario Dashboard", layout="wide")
apply_global_styles()
st.title("🎯 ClassForge: Scenario‑Based AI Modelling Dashboard")

# ----------------------- FILE SOURCE ----------------------------
if "uploaded_df" in st.session_state:
    data = st.session_state.uploaded_df.copy()
else:
    upl = st.file_uploader("📂 Upload CSV file", type=["csv"])
    if not upl:
        st.warning("⚠️ Please upload a CSV file to begin.")
        st.stop()
    data = pd.read_csv(upl)

st.success("✅ Dataset Loaded Successfully!")

# --------------------- CONTROLS (two rows) -----------------------
st.markdown("### ⚙️ Allocation Parameters")

# Row 1 – Scenario settings
c1, c2 = st.columns(2)
with c1:
    scenario_type = st.selectbox("Scenario Type", list(SCENARIO_FEATURES.keys()))
with c2:
    num_clusters = st.slider("Clusters", 2, 5, 3)

# Row 2 – Classroom planner
c3, c4 = st.columns(2)
with c3:
    class_capacity = st.number_input("Capacity / class", min_value=1, value=30)
with c4:
    num_classes = st.number_input("Available classes", min_value=1, value=10)

# ------------------------ RUN CLUSTERING ------------------------
data, group_labels, group_meanings = run_scenario_clustering(
    data, scenario_type, num_clusters
)

# Prepare summary table (used in Class Roster tab first)
group_counts = data['Scenario_Group'].value_counts().sort_index()
summary_table = pd.DataFrame({
    "Group Name": [group_labels[i] for i in range(num_clusters)],
    "Number of Students": [group_counts.get(i, 0) for i in range(num_clusters)],
    "Meaning": [group_meanings.get(i, "(no description)") for i in range(num_clusters)]
})

# ------------------------ ALLOCATE TO CLASSROOMS ------------------------
allocated_df, required_classrooms = allocate_to_classrooms(data, class_capacity)

# Add full name for class roster table
if "First_Name" in allocated_df.columns and "Last_Name" in allocated_df.columns:
    allocated_df["Name"] = allocated_df["First_Name"].str.strip() + " " + allocated_df["Last_Name"].str.strip()
elif "Name" not in allocated_df.columns:
    allocated_df["Name"] = allocated_df.get("Student_ID", "").astype(str)

# ------------------------ TABS ------------------------
tab_roster, tab_visual = st.tabs(["📋 Class Rosters", "📊 Visualisation"])

# ------------------------ CLASS ROSTERS TAB ------------------------
with tab_roster:
    st.markdown(f"### 📌 {scenario_type} – Scenario Group Summary")
    st.dataframe(summary_table)

    st.markdown("### 👥 Classroom Rosters")

    total_students = len(data)
    st.write(f"Total Students: {total_students}")
    st.write(f"Required Classrooms: {required_classrooms}")
    st.write(f"Available Classrooms: {num_classes}")

    if required_classrooms > num_classes:
        st.error(f"❗ {required_classrooms - num_classes} more classroom(s) needed!")
    else:
        st.success("✅ Classrooms are sufficient.")

    # ---------- editable full table ----------
    editable = st.checkbox("Enable manual edits", value=False)
    edited_df = st.data_editor(
        allocated_df[["Student_ID", "Name", "Scenario_Group", "Classroom"]],
        disabled=not editable,
        column_config={
            "Scenario_Group": st.column_config.NumberColumn(min_value=0),
            "Classroom":      st.column_config.NumberColumn(min_value=1),
        },
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state["scenario_edited_df"] = edited_df.copy()

    # ---------- per‑class tables ----------
    st.markdown("#### 🗂️  Class‑by‑Class View")
    cols_tbl = st.columns(2)
    for idx, (cls, sub) in enumerate(edited_df.groupby("Classroom"), 1):
        with cols_tbl[(idx - 1) % 2]:
            st.markdown(f"**Classroom {cls}** ({len(sub)} students)")
            st.dataframe(
                sub[["Student_ID", "Name"]],
                hide_index=True,
                use_container_width=True,
            )

    st.download_button(
        "📥 Download Class Roster (CSV)",
        to_csv_bytes(edited_df),
        "Final_Classroom_Assignment.csv",
        "text/csv",
    )

# ------------------------ VISUALISATION TAB ------------------------
with tab_visual:
    # ------------------ HEATMAP ----------------------
    st.markdown("### 🧊 Feature Averages by Scenario Group")
    try:
        heatmap_data = data.groupby("Group_Name")[SCENARIO_FEATURES[scenario_type]].mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    except Exception:
        st.warning("⚠️ Heatmap could not be displayed due to incompatible data.")

    # -------------------- TREND COMPARISON ---------------------
    st.markdown("### 📈 Feature Trend Comparison")
    features_to_compare = st.multiselect(
        "Select Features to Compare",
        options=SCENARIO_FEATURES[scenario_type]
    )
    if features_to_compare:
        for feature in features_to_compare:
            avg_data = data.groupby("Group_Name")[[feature]].mean().reset_index()
            avg_data["Meaning"] = avg_data["Group_Name"].map(
                {group_labels[i]: group_meanings[i] for i in range(num_clusters)}
            )
            bar = px.bar(
                avg_data, x="Group_Name", y=feature, color="Group_Name",
                title=f"Average {feature} by Scenario Group",
                hover_data=["Meaning"],
                color_discrete_sequence=px.colors.sequential.Blues
            )
            st.plotly_chart(bar, use_container_width=True)

st.markdown("---")
st.caption("💡 ClassForge - Scenario‑Based Allocation Tool | AI‑Powered 2025")
