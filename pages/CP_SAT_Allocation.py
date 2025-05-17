# pages/CP_SAT_Allocation.py

import math
import streamlit as st
import pandas as pd
import altair as alt
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

from utils.cpsat_utils import (
    compute_fitness,
    solve_constraints,
    to_csv_bytes
)
from utils.compare_utils import build_social_graph

st.set_page_config(page_title="CP-SAT Classroom Optimiser", layout="wide")
st.title("🎯 CP-SAT Constraint Optimisation")

# — 1) Redirect if no data in session state —
if "uploaded_df" not in st.session_state:
    st.session_state["redirect_warning"] = True
    if hasattr(st, "switch_page"):
        st.switch_page("Home.py")
    else:
        st.experimental_set_query_params(page="Home.py")
        st.stop()

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
    if presets[preset]:
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
st.markdown("##### Top 5 Fitness Scores")
st.dataframe(df.nlargest(5, "fitness")[["Student_ID","fitness"]], use_container_width=True)
st.markdown("##### Bottom 5 Fitness Scores")
st.dataframe(df.nsmallest(5, "fitness")[["Student_ID","fitness"]], use_container_width=True)

# ==== 7️⃣ Run Allocation & persist ====
if "alloc_df" not in st.session_state:
    if st.button("🚀 Run allocation"):
        with st.spinner("Optimising allocations…"):
            assignment = solve_constraints(df, num_classes, capacity)
        df["Class"] = df["Student_ID"].map(assignment).astype(int) + 1
        st.session_state.alloc_df = df
        st.success("✅ Allocation complete!")
    else:
        st.info("Click 🚀 Run allocation to assign students to classes.")
        st.stop()

# ==== 8️⃣ Once allocated, show Rosters & Visualizations ====
df = st.session_state.alloc_df

tab_rosters, tab_viz = st.tabs(["Class Rosters","Visualizations"])

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
        st.session_state.alloc_df = df_edit
        df = df_edit

    cols = st.columns(2)
    for idx, (cls, group) in enumerate(df.groupby("Class"), start=1):
        with cols[(idx-1)%2]:
            st.markdown(f"**Class {cls}**")
            st.dataframe(
                group[["Student_ID","fitness","Total_Score","life_satisfaction"]],
                hide_index=True, use_container_width=True
            )
    st.download_button(
        "Download full roster CSV",
        to_csv_bytes(df),
        file_name="cpsat_allocation.csv",
        mime="text/csv"
    )

with tab_viz:
    # 1) Class Size
    st.markdown("#### Class Size Distribution")
    size_df = df["Class"].value_counts().sort_index().reset_index()
    size_df.columns = ["Class","Count"]
    chart1 = (
        alt.Chart(size_df)
        .mark_bar(color="#4C78A8")
        .encode(x="Class:O", y="Count:Q", tooltip=["Class","Count"])
        .properties(width=600, height=300)
    )
    st.altair_chart(chart1, use_container_width=True)

    # 2) Bullying
    st.markdown("#### Bullying Rate: Before vs After by Class")
    before = df_raw["disrespected_by_peers"].map({"Yes":1,"No":0}).mean()*100
    by_class = (
        df.groupby("Class")["disrespected_by_peers"]
          .apply(lambda x: (x=="Yes").mean()*100)
          .reset_index(name="Bullied %")
    )
    chart2 = (
        alt.Chart(by_class)
        .mark_bar(color="#E45756")
        .encode(x="Class:O", y="Bullied %:Q", tooltip=["Class","Bullied %"])
        .properties(width=600, height=300)
    )
    base = (
        alt.Chart(pd.DataFrame([{"y": before}]))
        .mark_rule(color="black", strokeDash=[4,2])
        .encode(y="y:Q")
    )
    st.altair_chart(chart2 + base, use_container_width=True)
    st.caption(f"Black dashed line = overall bullied rate before allocation ({before:.1f}%)")

    # 3) Academic Scores
    st.markdown("#### Academic Score Distribution by Class")
    chart3 = (
        alt.Chart(df)
        .mark_boxplot(color="#59A14F")
        .encode(x="Class:O", y="Total_Score:Q", tooltip=["Class","Total_Score"])
        .properties(width=600, height=300)
    )
    st.altair_chart(chart3, use_container_width=True)

    # 4) Average Fitness
    st.markdown("#### Average Fitness per Class")
    fit_df = df.groupby("Class")["fitness"].mean().reset_index()
    chart4 = (
        alt.Chart(fit_df)
        .mark_line(point=True, color="#79706E")
        .encode(x="Class:O", y="fitness:Q", tooltip=["Class","fitness"])
        .properties(width=600, height=300)
    )
    st.altair_chart(chart4, use_container_width=True)

    st.markdown(
        """
        **What is the “Fitness” score?**  
        A single composite metric combining:
        - Academic performance (Total Score)  
        - Student well-being (life satisfaction)  
        - Social connections (close friends)  
        - Peer respect (low disrespect reports)
        """
    )

        # ==== 5️⃣ Social Network Graph ====
    st.markdown("#### Social Network Graph by Class")
    class_choice = st.selectbox("Select class", sorted(df["Class"].unique()))
    # get just those students
    sub_ids = set(df[df["Class"] == class_choice]["Student_ID"])
    sub_df  = df_raw[df_raw["Student_ID"].isin(sub_ids)]

    # build SNA graph (friend vs. conflict)
    G = build_social_graph(sub_df, use_synthetic=False)

    # count edges
    friend_edges   = [(u,v) for u,v,d in G.edges(data=True) if d["relation_type"]=="friend"]
    conflict_edges = [(u,v) for u,v,d in G.edges(data=True) if d["relation_type"]=="disrespect"]

    # simple metrics
    num_students     = len(sub_ids)
    num_friends      = len(friend_edges)
    num_conflicts    = len(conflict_edges)
    # average friends per student
    # count how many friend links each student has
    friend_degree = {n:0 for n in sub_ids}
    for u,v in friend_edges:
        friend_degree[u] += 1
        friend_degree[v] += 1
    avg_friends = sum(friend_degree.values()) / num_students if num_students else 0
    # how many students have zero friends
    num_isolated = sum(1 for v in friend_degree.values() if v == 0)
    # largest friend circle: biggest group connected by friendship only
    friend_graph = G.edge_subgraph(friend_edges).copy()
    largest_circle = max(
        (len(c) for c in nx.connected_components(friend_graph)),
        default=0
    )

    # display simple metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Students in Class",      num_students)
    m2.metric("Friend Links",           num_friends)
    m3.metric("Conflict Links",         num_conflicts)
    m4.metric("Avg. Friends per Student", f"{avg_friends:.1f}")
    m5.metric("Students with No Friends", num_isolated)
    m6.metric("Largest Friend Circle",   largest_circle)

    # build interactive PyVis network
    net = Network(
        height="600px", width="100%",
        bgcolor="#f8f9fa", font_color="#333",
        notebook=False
    )
    for n in G.nodes:
        net.add_node(n, label=str(n))
    colors = {"friend":"#4C78A8", "disrespect":"#E45756"}
    for u, v, d in G.edges(data=True):
        rel = d.get("relation_type","")
        net.add_edge(u, v, color=colors.get(rel,"#888"), title=rel)

    # spread nodes out
    net.repulsion(
        node_distance=250,
        central_gravity=0.01,
        spring_length=300,
        spring_strength=0.01
    )

    html = net.generate_html()
    components.html(html, height=650, scrolling=True)

    # friendly explanation
    st.markdown(
        f"In Class {class_choice}, there are **{num_students}** students.  \n"
        f"- They share **{num_friends}** friendship connections and **{num_conflicts}** conflicts.  \n"
        f"- On average, each student has **{avg_friends:.1f}** friends.  \n"
        f"- **{num_isolated}** students have no friends in this network.  \n"
        f"- The largest friend circle includes **{largest_circle}** students.  \n\n"
        "Use this graph and these numbers to quickly see how connected your students are, "
        "and where you might need to foster more friendships or address conflicts."
    )

