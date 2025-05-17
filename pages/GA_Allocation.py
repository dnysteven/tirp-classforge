# GA_Allocation.py
# src/streamlit_app.py  – adapted for ClassForge multipage layout
# Original comments kept; only UI placement & naming updated

import streamlit as st
import pandas as pd
from utils.ga_utils import load_and_scale, simulate_graph, setup_deap, run_ga
from utils.ui_utils import apply_global_styles
from utils.cpsat_utils import to_csv_bytes  # existing helper for CSV download
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="GA‑Powered Classroom Allocation", layout="wide")
apply_global_styles()
st.title("🧬 ClassForge: Genetic‑Algorithm Classroom Allocation")

# ---------------------------------------------------------------
# 1. DATA SOURCE
# ---------------------------------------------------------------
if "uploaded_df" in st.session_state:
    df_raw = st.session_state.uploaded_df.copy()
else:
    upl = st.file_uploader("📂 Upload student CSV", type="csv")
    if not upl:
        st.warning("Please upload a CSV file first (or return to Home).")
        st.stop()
    df_raw = pd.read_csv(upl)

# ---------------------------------------------------------------
# 2. CONTROLS  (formerly sidebar)  – two rows
# ---------------------------------------------------------------
row1c1, row1c2, row1c3 = st.columns(3)
with row1c1:
    K = st.number_input("Number of classes", 2, 10, 4)
with row1c2:
    pop = st.slider("Population size", 2, 100, 20)
with row1c3:
    gens = st.slider("Generations", 1, 50, 5)

row2c1, row2c2, row2c3 = st.columns(3)
with row2c1:
    w_acad = st.slider("Academic balance weight", 0.0, 1.0, 0.33)
with row2c2:
    w_fri = st.slider("Friendship retention weight", 0.0, 1.0, 0.33)
with row2c3:
    w_neg = st.slider("Disrespect penalty weight", 0.0, 1.0, 0.33)

# ---------------------------------------------------------------
# 3. PREP DATA  (scale + graph)
# ---------------------------------------------------------------
df_scaled, feature_cols = load_and_scale(df_raw)
G = simulate_graph(df_scaled)
# Debug: print total number of disrespect edges in full graph
total_disrespect = sum(
    1 for _, _, d in G.edges(data=True)
    if d.get("relation_type") == "disrespect"
)
print(f"[DEBUG] Total disrespect edges in graph: {total_disrespect}")

# ---------------------------------------------------------------
# 4. RUN GA & TABS
# ---------------------------------------------------------------
tab_roster, tab_vis = st.tabs(["📋 Class Rosters", "📊 Visualisation"])

with tab_roster:
    if st.button("🚀 Run Genetic Algorithm"):
        st.info("Running GA… this can take a few seconds.")
        toolbox = setup_deap(
            df_scaled, G,
            num_classes=K,
            weights=(w_acad, w_fri, w_neg)
        )
        best_ind, _log = run_ga(toolbox, pop_size=pop, gens=gens)

        # Decode into DataFrame
        df_alloc = pd.DataFrame({
            "Student_ID": df_scaled["Student_ID"],
            "Classroom":  [cls + 1 for cls in best_ind]  # 1‑indexed
        })

        # Add Name column if available
        if {"First_Name", "Last_Name"}.issubset(df_raw.columns):
            df_alloc["Name"] = (
                df_raw["First_Name"].str.strip()
                + " "
                + df_raw["Last_Name"].str.strip()
            )
        elif "Name" in df_raw.columns:
            df_alloc["Name"] = df_raw["Name"]
        else:
            df_alloc["Name"] = df_alloc["Student_ID"].astype(str)

        st.success("✅ GA complete!")

        # Editable master table
        editable = st.checkbox("Enable manual edits", value=False)
        df_edit = st.data_editor(
            df_alloc[["Student_ID", "Name", "Classroom"]],
            disabled=not editable,
            column_config={"Classroom": st.column_config.NumberColumn(min_value=1)},
            num_rows="dynamic",
            use_container_width=True,
        )
        st.session_state.ga_df_edit = df_edit

        # Per‑class tables
        st.markdown("#### 🗂️  Class‑by‑Class View")
        cols_tbl = st.columns(2)
        for idx, (cls, sub) in enumerate(df_edit.groupby("Classroom"), 1):
            with cols_tbl[(idx - 1) % 2]:
                st.markdown(f"**Class {cls}** ({len(sub)} students)")
                st.dataframe(
                    sub[["Student_ID", "Name"]],
                    hide_index=True,
                    use_container_width=True,
                )

        # Download
        st.download_button(
            "📥 Download allocation CSV",
            data=to_csv_bytes(df_edit),
            file_name="ga_allocation.csv",
            mime="text/csv",
        )
    else:
        st.info("Adjust parameters, then press **Run Genetic Algorithm**.")

# ---------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------
with tab_vis:
    if "ga_df_edit" in st.session_state:
        class_counts = (
            st.session_state.ga_df_edit["Classroom"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "Classroom", "count": "Students"})
        )
        
        st.subheader("💗 Wellbeing Composite by Class")
        st.bar_chart(class_counts, x="Classroom", y="Students", use_container_width=True)

        st.subheader("🌐 Full Student Network")
        def plot_student_network(G: nx.Graph, df: pd.DataFrame) -> go.Figure:
            pos = nx.spring_layout(G, seed=42)

            edge_traces = []
            colors = {"friend": "green", "disrespect": "red"}

            for rel_type, color in colors.items():
                edge_x = []
                edge_y = []
                for u, v, d in G.edges(data=True):
                    if d.get("relation_type") == rel_type:
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
                edge_traces.append(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1, color=color),
                    hoverinfo="none",
                    mode="lines",
                    name=rel_type.capitalize()
                ))

            # Draw nodes
            node_x = []
            node_y = []
            node_text = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=node_text,
                textposition="top center",
                marker=dict(color="yellow", size=10, line_width=2),
                name="Students"
            )

            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                showlegend=True,
                margin=dict(l=0, r=0, t=20, b=0),
                hovermode="closest"
            )
            return fig

        def plot_class_subgraph(G: nx.Graph, df: pd.DataFrame, class_label: int) -> go.Figure:
            ids = df[df["Assigned_Class"] == class_label]["Student_ID"].tolist()
            subG = G.subgraph(ids)
            
            intra_conflicts = sum(
                1 for _, _, d in subG.edges(data=True)
                if d.get("relation_type") == "disrespect"
            )
            print(f"[DEBUG] Class {cls} – Intra-class conflicts: {intra_conflicts}")
            
            return plot_student_network(subG, df)

        if st.checkbox("Show full network"):
            fig_full = plot_student_network(G, st.session_state.ga_df_edit.rename(columns={"Classroom": "Assigned_Class"}))
            st.plotly_chart(fig_full, use_container_width=True)

        st.subheader("🔍 Social Subgraphs by Class")
        classes = sorted(st.session_state.ga_df_edit["Classroom"].unique())
        for cls in classes:
            st.markdown(f"---\n**Class {cls}**")
            fig_sub = plot_class_subgraph(G, st.session_state.ga_df_edit.rename(columns={"Classroom": "Assigned_Class"}), cls)
            st.plotly_chart(fig_sub, use_container_width=True)

            ids = st.session_state.ga_df_edit.loc[st.session_state.ga_df_edit["Classroom"] == cls, "Student_ID"]
            subG = G.subgraph(ids)
            possible = len(ids) * (len(ids) - 1) / 2
            kept = sum(1 for _,_,d in subG.edges(data=True) if d["relation_type"] == "friend")
            conf = sum(1 for _,_,d in subG.edges(data=True) if d["relation_type"] == "disrespect")
            st.markdown(
                f"• **Friendship edges retained:** {kept}/{int(possible)} ({kept / possible:.0%})  \n"
                f"• **Intra-class conflicts:** {conf}/{int(possible)} ({conf / possible:.0%})  \n"
                f"{'Great cohesion!' if kept / possible > 0.5 else 'Consider adjusting friendship weight.'}"
            )
    else:
        st.info("Run the algorithm to see visualisations.")
