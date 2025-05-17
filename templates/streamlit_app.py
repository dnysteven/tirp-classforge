# src/streamlit_app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from data_prep import load_and_scale, simulate_graph
from ga_engine import setup_deap, run_ga

def plot_student_network(G: nx.Graph, assignment: pd.DataFrame):
    """Plot the entire social network, colored by class."""
    pos = nx.spring_layout(G, seed=42)
    cls_map = dict(zip(assignment.Student_ID, assignment.Assigned_Class))

    # Edge traces
    edge_traces = []
    for relation, color in [("friend","green"), ("disrespect","red")]:
        xs, ys = [], []
        for u, v, d in G.edges(data=True):
            if d["relation_type"] == relation:
                x0,y0 = pos[u]; x1,y1 = pos[v]
                xs += [x0, x1, None]; ys += [y0, y1, None]
        edge_traces.append(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="none",
                name=relation
            )
        )

    # Node trace
    node_x, node_y, node_color, node_text = [], [], [], []
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        c = cls_map[n]
        node_color.append(c)
        node_text.append(f"ID: {n}<br>Class: {c}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(
            size=8,
            color=node_color,
            colorscale="Viridis",
            colorbar=dict(title="Class"),
            line=dict(width=1, color="#333")
        ),
        text=[str(cls_map[n]) for n in G.nodes()],
        textposition="bottom center",
        hovertext=node_text, hoverinfo="text",
        name="Students"
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=True,
            title="Full Social Network: Nodes by Assigned Class",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    return fig

def plot_class_subgraph(G: nx.Graph, assignment: pd.DataFrame, class_id: int):
    """Plot only the subgraph for a single class."""
    ids = assignment.loc[assignment.Assigned_Class == class_id, "Student_ID"].tolist()
    subG = G.subgraph(ids)
    pos = nx.spring_layout(subG, seed=42)

    # Edge traces
    edge_traces = []
    for relation, color in [("friend","green"), ("disrespect","red")]:
        xs, ys = [], []
        for u, v, d in subG.edges(data=True):
            if d["relation_type"] == relation:
                x0,y0 = pos[u]; x1,y1 = pos[v]
                xs += [x0, x1, None]; ys += [y0, y1, None]
        edge_traces.append(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="none",
                name=relation
            )
        )

    # Node trace
    node_x, node_y, node_text = [], [], []
    for n in subG.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(f"ID: {n}<br>Class: {class_id}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(
            size=10,
            color=class_id,
            colorscale="Viridis",
            line=dict(width=1, color="#333")
        ),
        text=[str(class_id)] * len(node_x),
        textposition="bottom center",
        hovertext=node_text, hoverinfo="text",
        name=f"Class {class_id}"
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=True,
            title=f"Social Subgraph for Class {class_id}",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    return fig

def main():
    st.set_page_config(page_title="ClassForge", layout="wide")
    st.title("ClassForge: GA-Powered Classroom Allocation")

    # ─── Sidebar Controls ────────────────────────────────────────────
    K        = st.sidebar.number_input("Number of classes", 2, 10, 10)
    w_acad   = st.sidebar.slider("Academic balance weight",      0.0, 1.0, 0.33)
    w_fri    = st.sidebar.slider("Friendship retention weight", 0.0, 1.0, 0.33)
    w_neg    = st.sidebar.slider("Disrespect penalty weight",   0.0, 1.0, 0.33)
    pop_size = st.sidebar.slider("Population size",  2, 100, 20)
    gens     = st.sidebar.slider("Generations",     1,  50,  5)

    # ─── Load & Simulate ──────────────────────────────────────────────
    df, feature_cols = load_and_scale()
    G                = simulate_graph(df)

    # ─── Run Genetic Algorithm ───────────────────────────────────────
    if st.button("Run Genetic Algorithm"):
        st.info("Running GA… please wait.")
        toolbox = setup_deap(df, G, num_classes=K, weights=(w_acad, w_fri, w_neg))
        pop, best_ind, log = run_ga(toolbox, pop_size=pop_size, gens=gens)

        # Decode assignment
        assignment = pd.DataFrame({
            "Student_ID":     df["Student_ID"],
            "Assigned_Class": best_ind
        })
        st.success("✅ GA complete!")

        # Assignment table + download
        st.subheader("📋 Best Assignment")
        st.dataframe(assignment)
        csv = assignment.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "allocations.csv")

        # ─── Academic Balance ────────────────────────────────────────
        class_avgs = (
            assignment
            .merge(df[["Student_ID","Academic_Composite"]], on="Student_ID")
            .groupby("Assigned_Class")["Academic_Composite"]
            .mean()
        )
        st.subheader("📊 Academic Composite by Class")
        st.bar_chart(class_avgs)

        # Dynamic explanation
        min_cls, max_cls = class_avgs.idxmin(), class_avgs.idxmax()
        min_val, max_val = class_avgs[min_cls], class_avgs[max_cls]
        spread = max_val - min_val
        st.markdown(
            f"**📈 Academic Summary:** Class **{max_cls}** has the highest average "
            f"composite ({max_val:.2f}), Class **{min_cls}** the lowest ({min_val:.2f}). "
            f"Gap of **{spread:.2f}** indicates "
            f"{'excellent balance.' if spread<0.1 else 'room for improvement.'}"
        )

        # ─── Wellbeing Balance ───────────────────────────────────────
        wellbeing_cols = [
            "Stress_Level (1-10)", "Sleep_Hours_per_Night",
            "life_satisfaction", "feels_nervous_frequency",
            "feels_depressed_frequency", "effort_exhaustion_level",
            "worthless_feeling_frequency"
        ]
        df_wb = df[["Student_ID"] + wellbeing_cols].copy()
        df_wb["Wellbeing_Composite"] = df_wb[wellbeing_cols].sum(axis=1)
        wb_avgs = (
            assignment
            .merge(df_wb[["Student_ID","Wellbeing_Composite"]], on="Student_ID")
            .groupby("Assigned_Class")["Wellbeing_Composite"]
            .mean()
        )
        st.subheader("💗 Wellbeing Composite by Class")
        st.bar_chart(wb_avgs)

        # Dynamic explanation
        min_wb_cls, max_wb_cls = wb_avgs.idxmin(), wb_avgs.idxmax()
        min_wb, max_wb = wb_avgs[min_wb_cls], wb_avgs[max_wb_cls]
        wb_spread = max_wb - min_wb
        st.markdown(
            f"**💗 Wellbeing Summary:** Class **{max_wb_cls}** has highest average "
            f"wellbeing ({max_wb:.2f}), Class **{min_wb_cls}** lowest ({min_wb:.2f}). "
            f"Spread **{wb_spread:.2f}** means "
            f"{'well-distributed wellbeing.' if wb_spread<0.1 else 'consider adjusting weights.'}"
        )

        # ─── Full Network (optional) ─────────────────────────────────
        if st.checkbox("Show full network"):
            st.subheader("🌐 Full Student Social Network")
            fig_full = plot_student_network(G, assignment)
            st.plotly_chart(fig_full, use_container_width=True)

        # ─── Subgraphs by Class ──────────────────────────────────────
        st.subheader("🔍 Social Subgraphs by Class")
        classes = sorted(assignment.Assigned_Class.unique())
        for cls in classes:
            st.markdown(f"---\n**Class {cls}**")
            fig_sub = plot_class_subgraph(G, assignment, cls)
            st.plotly_chart(fig_sub, use_container_width=True)

            # Dynamic network stats
            nodes = assignment.loc[assignment.Assigned_Class == cls, "Student_ID"]
            subG = G.subgraph(nodes)
            possible = len(nodes)*(len(nodes)-1)/2
            kept = sum(1 for _,_,d in subG.edges(data=True) if d["relation_type"]=="friend")
            conf = sum(1 for _,_,d in subG.edges(data=True) if d["relation_type"]=="disrespect")
            retention = kept/possible if possible>0 else 0
            conflict = conf/possible if possible>0 else 0
            st.markdown(
                f"• **Friendship edges retained:** {kept}/{int(possible)} "
                f"({retention:.0%})  \n"
                f"• **Intra-class conflicts:** {conf}/{int(possible)} "
                f"({conflict:.0%})  \n"
                f"{'Great cohesion!' if retention>0.5 else 'Consider boosting friendship weight.'}"
            )

if __name__ == "__main__":
    main()
