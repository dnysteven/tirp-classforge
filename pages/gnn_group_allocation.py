# pages/gnn_group_allocation.py
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_utils import render_sidebar
from utils.gnn_utils import smart_allocate_groups, to_csv_bytes

# ── page config & sidebar ────────────────────────────────────────────
st.set_page_config(page_title="GNN Group Allocation", layout="wide")
render_sidebar()
st.title("Smart GNN Group Allocation")

# ── File upload ──────────────────────────────────────────────────────
uploaded = st.file_uploader(
	"Upload CSV / Excel with Student_ID, First_Name, Last_Name, Final_Score …",
	type=["csv", "xls", "xlsx"],
)

if not uploaded:
	st.info("Upload a file to begin.")
	st.stop()

df_raw = (
	pd.read_csv(uploaded)
	if uploaded.name.lower().endswith(".csv")
	else pd.read_excel(uploaded)
)

# ── run allocator once per session ───────────────────────────────────
if "gnn_alloc" not in st.session_state:
	df_alloc, graph, labels = smart_allocate_groups(df_raw)
	st.session_state.update(gnn_alloc=df_alloc, graph=graph, labels=labels)

# ── UI tabs ────────────────────────────────────────
tab_tables, tab_graph = st.tabs(["Group Tables", "Network Graph"])

# ——————————————————— Group-Tables tab ————————————————————————
with tab_tables:
	edited_df = st.data_editor(
		st.session_state.gnn_alloc,
		column_config={"group": st.column_config.NumberColumn(min_value=1)},
		num_rows="fixed", use_container_width=True,
	)
	if not edited_df.equals(st.session_state.gnn_alloc):
		edited_df["group"] = edited_df["group"].astype(int)
		st.session_state.gnn_alloc = edited_df

	st.markdown("### Current groups")
	cols = st.columns(3)
	for idx, (g, sub) in enumerate(st.session_state.gnn_alloc.groupby("group"), 1):
		with cols[(idx - 1) % 3]:
			st.markdown(f"**Group {g}**")
			st.table(sub[["student_id", "name"]])

	st.download_button(
		"Download allocation (CSV)",
		data=to_csv_bytes(st.session_state.gnn_alloc),
		file_name="gnn_group_allocation.csv",
		mime="text/csv",
	)

# ——————————————————— Network-Graph tab ————————————————————————
with tab_graph:
  # ---- gnn similarity network chart -----------------------------
	st.markdown("### GNN Similarity Network Graph")

	G = st.session_state.graph
	labels = st.session_state.labels
	pos = nx.spring_layout(G, seed=42)

	edge_x, edge_y = [], []
	for u, v in G.edges():
		x0, y0 = pos[u]; x1, y1 = pos[v]
		edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y, mode="lines",
		line=dict(width=0.5, color="#888"), hoverinfo="none"
	)

	node_x, node_y, node_c = [], [], []
	for i, node in enumerate(G.nodes()):
		x, y = pos[node]
		node_x.append(x); node_y.append(y); node_c.append(labels[i])

	node_trace = go.Scatter(
		x=node_x, y=node_y, mode="markers",
		marker=dict(size=10, color=node_c, colorscale="Viridis",
								showscale=True, line_width=1),
		hoverinfo="text"
	)

	fig = go.Figure(
		data=[edge_trace, node_trace],
		layout=go.Layout(
			title="GNN-Derived Student Similarity Graph",
			margin=dict(l=0, r=0, t=40, b=0),
			hovermode="closest", showlegend=False,
		),
	)
	st.plotly_chart(fig, use_container_width=True)

	# ---- intra-group density chart -----------------------------
	st.markdown("### Friendship Level within Group")

	densities = []
	for g, sub in st.session_state.gnn_alloc.groupby("group"):
		nodes   = sub["student_id"].tolist()
		subg    = G.subgraph(nodes)
		n, m    = subg.number_of_nodes(), subg.number_of_edges()
		density = 0 if n <= 1 else 2 * m / (n * (n - 1))
		densities.append({"group": g, "density": density})

	dens_fig = px.bar(
		densities, x="group", y="density",
		labels={"group": "Group", "density": "Density"},
		title="Edge Density within Each Group",
	)
	dens_fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
	st.plotly_chart(dens_fig, use_container_width=True)