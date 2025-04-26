import streamlit as st, pandas as pd, networkx as nx, plotly.graph_objects as go
from utils.ui_utils import render_sidebar
from utils.gnn_utils import smart_allocate_groups, to_csv_bytes

# ── Streamlit page ---------------------------------------------------
st.set_page_config(page_title="GNN Group Allocation", layout="wide")
render_sidebar()
st.title("Smart GNN Group Allocation")

uploaded = st.file_uploader(
	"Upload CSV / Excel with Student_ID, First_Name, Last_Name, Final_Score …",
	type=["csv", "xls", "xlsx"],
)

if not uploaded:
	st.info("Upload a file to begin.")
	st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
if "gnn_alloc" not in st.session_state:
	df_alloc, G, labels = smart_allocate_groups(df_raw)
	st.session_state.update(gnn_alloc=df_alloc, G=G, labels=labels)

# editable grid
df_edit = st.data_editor(
	st.session_state.gnn_alloc,
	column_config={"group": st.column_config.NumberColumn(min_value=1)},
	num_rows="fixed", use_container_width=True,
)
if not df_edit.equals(st.session_state.gnn_alloc):
	df_edit["group"] = df_edit["group"].astype(int)
	st.session_state.gnn_alloc = df_edit

# network viz
pos, G = nx.spring_layout(st.session_state.G, seed=42), st.session_state.G
edge_x, edge_y = [], []

for u, v in G.edges():
	x0, y0 = pos[u]; x1, y1 = pos[v]
	edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888"), hoverinfo="none")
node_x, node_y, node_c = [], [], []

for node in G.nodes():
	x, y = pos[node]; node_x.append(x); node_y.append(y)
	node_c.append(st.session_state.labels[list(G.nodes()).index(node)])
node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=10, color=node_c,
												colorscale="Viridis", showscale=True, line_width=1))
st.plotly_chart(go.Figure(data=[edge_trace, node_trace], layout=go.Layout(margin=dict(l=0, r=0, t=25, b=0))), 
                use_container_width=True)

# tables + download
st.markdown("### Current groups")
cols = st.columns(3)
for idx, (g, sub) in enumerate(st.session_state.gnn_alloc.groupby("group"), 1):
	with cols[(idx-1) % 3]:
		st.markdown(f"**Group {g}**")
		st.table(sub[["student_id", "name"]])

st.download_button(
	"Download allocation (CSV)",
	data=to_csv_bytes(st.session_state.gnn_alloc),
	file_name="gnn_group_allocation.csv",
	mime="text/csv",
)