# pages/Compare_Models.py
import streamlit as st
import plotly.graph_objects as go
from utils.ui_utils import apply_global_styles
from utils.compare_utils import (
    ENGINE_IDS, run_comparison, friend_conflict_counts
)

st.set_page_config(page_title="Compare Models", layout="wide")
apply_global_styles()
st.title("🔀 Model Comparer")

# --- data check
if "uploaded_df" not in st.session_state:
    st.warning("Upload a CSV on Home first.")
    st.stop()

full_df = st.session_state.uploaded_df

# --- engine selection
label2id = {lbl: mid for mid, lbl in ENGINE_IDS.items()}
labels = st.multiselect("Select ≥ 2 models", list(label2id.keys()))
model_ids = [label2id[lbl] for lbl in labels]

# --- sampling
a, b, c = st.columns(3)
with a: frac = st.slider("Fraction", 0.1, 1.0, 0.25, 0.05)
with b: max_n = st.number_input("Max rows", 50, 3000, 200, 50)
with c: seed = st.number_input("Seed", 0, 9999, 42)

run_btn = st.button("🚀 Run comparison", disabled=len(model_ids) < 2)
if not run_btn:
    st.stop()

# --- run comparison via utils
sample, G_base, pos, results, errors = run_comparison(
    full_df, model_ids, frac, max_n, seed
)

if errors:
    for mid, msg in errors.items():
        st.error(f"{ENGINE_IDS.get(mid, mid)} failed: {msg}")

if len(results) < 2:
    st.error("Need ≥ 2 successful engines.")
    st.stop()

st.success(f"Sample size: {len(sample)} students")

# --- visualisation
st.markdown("## Model Comparison (Side by Side – Max 2 Columns)")

model_items = list(results.items())
column_sets = [model_items[i:i+2] for i in range(0, len(model_items), 2)]

CLASS_COLORS = ["red", "green", "blue"]

for row_models in column_sets:
    cols = st.columns(len(row_models))
    for (mid, df_alloc), col in zip(row_models, cols):
        with col:
            model_label = ENGINE_IDS[mid]
            st.markdown(f"### {model_label}")

            cmap = dict(zip(df_alloc["Student_ID"], df_alloc["Classroom"]))
            class_groups = df_alloc.groupby("Classroom")
            top_classes = sorted(class_groups.groups.keys())[:3]

            for i, cls in enumerate(top_classes):
                sub_df = class_groups.get_group(cls)
                sub_nodes = set(sub_df["Student_ID"])
                subgraph = G_base.subgraph(sub_nodes).copy()

                # colored edge traces
                edge_traces = []
                f_in = d_in = 0

                # Friend edges
                x_f, y_f = [], []
                for u, v, d in subgraph.edges(data=True):
                    if d.get("relation_type") == "friend":
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        x_f += [x0, x1, None]
                        y_f += [y0, y1, None]
                        f_in += 1
                if x_f:
                    edge_traces.append(go.Scatter(
                        x=x_f, y=y_f, mode="lines",
                        line=dict(width=1, color="green"),
                        hoverinfo="none", name="friend"
                    ))

                # Disrespect edges
                x_d, y_d = [], []
                for u, v, d in subgraph.edges(data=True):
                    if d.get("relation_type") == "disrespect":
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        x_d += [x0, x1, None]
                        y_d += [y0, y1, None]
                        d_in += 1
                if x_d:
                    edge_traces.append(go.Scatter(
                        x=x_d, y=y_d, mode="lines",
                        line=dict(width=1, color="red"),
                        hoverinfo="none", name="conflict"
                    ))

                # node trace (yellow)
                x_node, y_node, texts = [], [], []
                for n in sub_nodes:
                    x, y = pos[n]
                    x_node.append(x)
                    y_node.append(y)
                    texts.append(str(n))

                node_trace = go.Scatter(
                    x=x_node, y=y_node,
                    mode="markers",
                    marker=dict(size=8, color="yellow", showscale=False),
                    text=texts,
                    hoverinfo="text"
                )

                fig = go.Figure(data=edge_traces + [node_trace],
                                layout=go.Layout(
                                    title=f"Class {cls}",
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    hovermode="closest",
                                    showlegend=False
                                ))

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"✅ Friends kept: **{f_in}** | ❌ Conflicts kept: **{d_in}**")

            # Full network
            edge_traces = {
                "friend": {"x": [], "y": [], "color": "green"},
                "disrespect": {"x": [], "y": [], "color": "red"},
                "other": {"x": [], "y": [], "color": "#bbb"},
            }

            for u, v, d in G_base.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                same_class = cmap.get(u) == cmap.get(v)

                if d["relation_type"] == "friend" and same_class:
                    etype = "friend"
                elif d["relation_type"] == "disrespect" and same_class:
                    etype = "disrespect"
                else:
                    etype = "other"

                edge_traces[etype]["x"] += [x0, x1, None]
                edge_traces[etype]["y"] += [y0, y1, None]

            edge_scatter = []
            for etype, data in edge_traces.items():
                edge_scatter.append(go.Scatter(
                    x=data["x"], y=data["y"],
                    mode="lines",
                    line=dict(width=1, color=data["color"]),
                    hoverinfo="none",
                    name=etype
                ))

            node_x, node_y, node_text = [], [], []
            for n in G_base.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{n} → Class {cmap.get(n, '?')}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                marker=dict(
                    size=6,
                    color=[cmap.get(n, 0) for n in G_base.nodes()],
                    colorscale="Viridis",
                    showscale=False
                ),
                text=node_text,
                hoverinfo="text"
            )

            fig = go.Figure(
                data=edge_scatter + [node_trace],
                layout=go.Layout(
                    title=f"Full Network – {model_label}",
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="closest",
                    showlegend=False
                )
            )

            f_all, d_all = friend_conflict_counts(df_alloc, G_base)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"✅ Friends kept: **{f_all}** | ❌ Conflicts kept: **{d_all}**")

st.markdown("Green = friendship • Red = conflict • Grey = between-class")
