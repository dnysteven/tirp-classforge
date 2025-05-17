# pages/DeepRL_Allocation.py
import streamlit as st, pandas as pd, altair as alt
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

# ── 2. Rebuild Student_Name (always) ────────────────────────────────
if {"First_Name", "Last_Name"}.issubset(df_raw.columns):
    df_raw["Student_Name"] = (
        df_raw["First_Name"].astype(str).str.strip()
        + " "
        + df_raw["Last_Name"].astype(str).str.strip()
    )
else:
    df_raw["Student_Name"] = df_raw["Student_ID"].astype(str)

# ── 3. Parameter controls ───────────────────────────────────────────
#cap_col, cls_col = st.columns(1)
#capacity = cap_col.number_input("Capacity per class", 1, 40, 20, 1)
#num_classrooms = cls_col.slider("Number of classrooms (≤10)", 2, 10, 10, 1)
num_classrooms = st.slider("Number of classrooms", 2, 10, 6, 1)

# ── 4. Load DQN model (cached in session) ───────────────────────────
if "dqn_model" not in st.session_state:
    st.session_state.dqn_model = load_model(state_size=15, action_size=10)

with st.spinner("Allocating with DQN…"):
    assigned_df = allocate_students(
        df_raw,
        st.session_state.dqn_model,
        num_classrooms=num_classrooms,
        #max_capacity=capacity,
    )
    
    # Ensure Student_Name is restored (in case allocator strips it)
    if "Student_Name" in df_raw.columns:
        assigned_df = assigned_df.merge(
            df_raw[["Student_ID", "Student_Name"]],
            on="Student_ID",
            how="left"
        )

# Move key columns forward
front = ["Assigned_Classroom", "Student_ID"]
if "Student_Name" in assigned_df.columns:
    front.append("Student_Name")
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

    # Students per Classroom
    counts = (
        assigned_df["Assigned_Classroom"]
        .value_counts()
        .sort_index()
        .reset_index(name="Students")
    )
    counts.columns = ["Classroom", "Students"]

    st.markdown("### 📊 Students per Classroom")
    st.bar_chart(counts, x="Classroom", y="Students", use_container_width=True)

    # Allocation Reasons
    st.markdown("### 📝 Allocation Reasons (by Category)")
    # Compute counts
    reason_counts = (
        assigned_df["Reason"]
            .value_counts()
            .reset_index(name="Count")
            .rename(columns={"index": "Reason"})
    )
    # Map each raw reason into a broader category by keyword matching
    def map_reason_category(reason: str) -> str:
        rl = reason.lower()
        if "profile" in rl:
            return "Skill Balance"
        # anything about disrespect, bullying, stress or satisfaction → Wellbeing
        elif any(term in rl for term in ["disrespect", "bullied", "stress", "satisfaction"]):
            return "Wellbeing"
        # friendship cues → Social Fit
        elif "friend" in rl:
            return "Social Fit"
        # high performers, mentors, leaders, challenges → Performance
        elif any(term in rl for term in ["performer", "mentor", "leadership", "challenge"]):
            return "Performance"
        # fall-back bucket
        else:
            return "Other"

    reason_counts["Category"] = reason_counts["Reason"].apply(map_reason_category)

    # Sort descending
    reason_counts = reason_counts.sort_values("Count", ascending=False)

    # Build Altair chart
    chart = (
        alt.Chart(reason_counts)
          .mark_bar(cornerRadiusTopLeft=3, cornerRadiusBottomLeft=3)
          .encode(
             y=alt.Y(
                 "Reason:N", sort="-x", title=None,
                 axis=alt.Axis(labelFontSize=12, labelLimit=300)
             ),
             x=alt.X(
                 "Count:Q", title="Number of Students",
                 axis=alt.Axis(labelFontSize=12, titleFontSize=14)
             ),
             color=alt.Color(
                 "Category:N", title="Reason Category",
                 legend=alt.Legend(orient="right", labelFontSize=12, titleFontSize=14)
             ),
             tooltip=[
                 alt.Tooltip("Reason:N", title="Reason"),
                 alt.Tooltip("Count:Q", title="Count"),
                 alt.Tooltip("Category:N", title="Category")
             ]
          )
          .properties(height=600, width=800)
          .configure_view(strokeOpacity=0)
          .configure_axis(grid=False)
    )
    st.altair_chart(chart, use_container_width=True)
