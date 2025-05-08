# pages/CP_SAT_Allocation.py

import math
import streamlit as st
import pandas as pd
import altair as alt

from utils.cpsat_utils import (
    load_csv,
    compute_fitness,
    solve_constraints,
    to_csv_bytes
)

st.set_page_config(page_title="CP-SAT Classroom Optimiser", layout="wide")
st.title("🎯 CP-SAT Constraint Optimisation")

# — 1) Redirect if no data in session state —
if "uploaded_df" not in st.session_state:
    st.session_state["redirect_warning"] = True
    if hasattr(st, "switch_page"):
        st.switch_page("Home.py")
    else:
        st.experimental_set_query_params(page="Home.py")
        st.experimental_rerun()

# — 2) Grab the uploaded DataFrame —
df_raw = st.session_state.uploaded_df.copy()

# ==== 3️⃣ Configure Fitness Weights ====
st.header("1️⃣ Configure Fitness Weights")

presets = {
    "Balanced":         (0.4, 0.3, 0.15, 0.15),
    "Academic Focus":   (0.6, 0.2, 0.10, 0.10),
    "Well-being Focus": (0.2, 0.5, 0.15, 0.15),
    "Social Focus":     (0.2, 0.2, 0.40, 0.20),
    "Custom":           None
}
col_p, col_s = st.columns([2,5])
with col_p:
    preset = st.selectbox("Choose a preset:", list(presets.keys()), index=0)
with col_s:
    if preset != "Custom":
        w_acad, w_well, w_friend, w_disrespect = presets[preset]
        st.markdown(f"**Preset:** {preset}")
        st.markdown(
            f"- Academic: **{w_acad}**  •  "
            f"Well-being: **{w_well}**  •  "
            f"Friendship: **{w_friend}**  •  "
            f"Respect: **{w_disrespect}**"
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        w_acad      = c1.slider("Academic",    0.0,1.0,0.4,0.05)
        w_well      = c2.slider("Well-being",  0.0,1.0,0.3,0.05)
        w_friend    = c3.slider("Friendship",  0.0,1.0,0.15,0.05)
        w_disrespect= c4.slider("Respect",     0.0,1.0,0.15,0.05)

# ==== 4️⃣ Configure Classrooms ====
st.header("2️⃣ Configure Classrooms")
N = len(df_raw)
cap_col, cls_col = st.columns(2)
capacity    = cap_col.number_input("Capacity per class:", min_value=1, value=30, step=1)
min_classes = math.ceil(N / capacity)
num_classes = cls_col.number_input(
    "Number of classes:",
    min_value=min_classes,
    max_value=1000,
    value=min_classes,
    step=1
)

# ==== 5️⃣ Compute Fitness ====
df = compute_fitness(df_raw, w_acad, w_well, w_friend, w_disrespect)

# ==== 6️⃣ Preview Fitness ====
st.markdown("#### Sample of Computed Fitness")
st.dataframe(
    df[[
        "Student_ID",
        "Total_Score",
        "life_satisfaction",
        "closest_friend_count",
        "disrespected_by_peers",
        "fitness"
    ]].head(8),
    use_container_width=True
)

# Show top/bottom fitness to observe weight effects
st.markdown("##### Top 5 Fitness Scores")
st.dataframe(df.nlargest(5, "fitness")[["Student_ID","fitness"]], use_container_width=True)
st.markdown("##### Bottom 5 Fitness Scores")
st.dataframe(df.nsmallest(5, "fitness")[["Student_ID","fitness"]], use_container_width=True)

# ==== 7️⃣ Run Allocation ====
if st.button("🚀 Run allocation"):
    with st.spinner("Optimising allocations…"):
        assignment = solve_constraints(df, num_classes, capacity)

    df["Class"] = df["Student_ID"].map(assignment).astype(int) + 1
    st.success("✅ Allocation complete!")

    tab_rosters, tab_viz = st.tabs(["Class Rosters", "Visualizations"])

    # --- Tab: Class Rosters ---
    with tab_rosters:
        st.markdown("### Editable Class Rosters")
        edit = st.checkbox("Allow manual edits", value=False)
        df_edit = st.data_editor(
            df,
            disabled=not edit,
            use_container_width=True,
            column_config={"Class": st.column_config.NumberColumn(min_value=1)}
        )
        if edit:
            df = df_edit

        cols = st.columns(2)
        for idx, (cls, group) in enumerate(df.groupby("Class"), start=1):
            with cols[(idx-1) % 2]:
                st.markdown(f"**Class {cls}**")
                st.dataframe(
                    group[["Student_ID","fitness","Total_Score","life_satisfaction"]],
                    hide_index=True,
                    use_container_width=True
                )

        st.download_button(
            "Download full roster CSV",
            to_csv_bytes(df),
            file_name="cpsat_allocation.csv",
            mime="text/csv"
        )

    # --- Tab: Visualizations ---
    with tab_viz:
        # 1) Class Size Distribution
        st.markdown("#### Class Size Distribution")
        size_df = df["Class"].value_counts().sort_index().reset_index()
        size_df.columns = ["Class","Count"]
        chart1 = (
            alt.Chart(size_df)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Class:O", title="Class"),
                y=alt.Y("Count:Q", title="Number of Students"),
                tooltip=["Class","Count"]
            )
            .properties(width=600, height=300)
        )
        st.altair_chart(chart1, use_container_width=True)

        # 2) Bullying Rate: per-class + baseline
        st.markdown("#### Bullying Rate: Before vs After by Class")
        before_pct = df_raw["disrespected_by_peers"].map({"Yes":1,"No":0}).mean()*100
        bully_by_class = (
            df.groupby("Class")["disrespected_by_peers"]
              .apply(lambda x: (x=="Yes").mean()*100)
              .reset_index(name="Bullied %")
        )
        chart2 = (
            alt.Chart(bully_by_class)
            .mark_bar(color="#E45756")
            .encode(
                x=alt.X("Class:O", title="Class"),
                y=alt.Y("Bullied %:Q", title="Bullying Rate (%)"),
                tooltip=["Class","Bullied %"]
            )
            .properties(width=600, height=300)
        )
        baseline = (
            alt.Chart(pd.DataFrame([{"y": before_pct}]))
            .mark_rule(color="black", strokeDash=[4,2])
            .encode(y="y:Q")
        )
        st.altair_chart(chart2 + baseline, use_container_width=True)
        st.caption(f"Black dashed line = overall bullied rate before allocation ({before_pct:.1f}%)")

        # 3) Academic Score Distribution by Class
        st.markdown("#### Academic Score Distribution by Class")
        chart3 = (
            alt.Chart(df)
            .mark_boxplot(color="#59A14F")
            .encode(
                x=alt.X("Class:O", title="Class"),
                y=alt.Y("Total_Score:Q", title="Total Score"),
                tooltip=["Class","Total_Score"]
            )
            .properties(width=600, height=300)
        )
        st.altair_chart(chart3, use_container_width=True)

        # 4) Average Fitness per Class
        st.markdown("#### Average Fitness per Class")
        fit_df = df.groupby("Class")["fitness"].mean().reset_index()
        chart4 = (
            alt.Chart(fit_df)
            .mark_line(point=True, color="#79706E")
            .encode(
                x=alt.X("Class:O", title="Class"),
                y=alt.Y("fitness:Q", title="Average Fitness"),
                tooltip=["Class","fitness"]
            )
            .properties(width=600, height=300)
        )
        st.altair_chart(chart4, use_container_width=True)

        # ➡️ Explain what "fitness" means
        st.markdown(
            """
            **What is the “Fitness” score?**  
            The fitness score is a single, composite metric we use to balance each classroom. It combines:
            - **Academic performance** (Total Score)  
            - **Student well-being** (self-reported life satisfaction)  
            - **Social connections** (number of close friends)  
            - **Peer respect** (low levels of reported disrespect)  

            Each component is normalized to the same 0–1 scale and weighted (e.g. 40% academics, 30% well-being, 
            15% friendships, 15% respect). A higher fitness means a student is doing well across all four areas.  
            The “Average Fitness per Class” chart shows how evenly balanced each room is in terms of overall 
            student readiness and support needs.
            """
        )
