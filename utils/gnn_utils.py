import io, numpy as np, pandas as pd, torch
import networkx as nx, community as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ── column normaliser ────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if "student_id" not in df.columns:
		if "Student_ID" in df.columns:
			df = df.rename(columns={"Student_ID": "student_id"})
		else:
			raise ValueError("Missing Student_ID / student_id")
	if "name" not in df.columns:
		if {"First_Name", "Last_Name"} <= set(df.columns):
			df["name"] = df["First_Name"].str.strip() + " " + df["Last_Name"].str.strip()
		else:
			df["name"] = df["student_id"].astype(str)
	if "Final_Score" not in df.columns:
		raise ValueError("Missing Final_Score")
	if "life_satisfaction" not in df.columns:
		df["life_satisfaction"] = pd.NA
	return df[["student_id", "name", "Final_Score", "life_satisfaction"]]

# ── tiny utility for downloads ───────────────────────────────────────
def to_csv_bytes(df: pd.DataFrame) -> bytes:
	buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

# ── GNN model definition ─────────────────────────────────────────────
class GNNModel(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim=8, output_dim=2):
		super().__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, output_dim)

	def forward(self, data: Data):
		x, edge_index = data.x, data.edge_index
		x = F.relu(self.conv1(x, edge_index))
		return self.conv2(x, edge_index)

# ── main allocator ───────────────────────────────────────────────────
def smart_allocate_groups(df_raw: pd.DataFrame, k_neighbors: int = 5):
	df = normalise_columns(df_raw)
	
	# ── SAMPLE ONLY 2000 STUDENTS FOR TESTING ───────────────────────
	if len(df) > 2000:
		df = df.sample(n=2000, random_state=42).reset_index(drop=True)
	
	df["Final_Score"]       = df["Final_Score"].fillna(50)
	df["life_satisfaction"] = df["life_satisfaction"].fillna(5)

	# build k-NN graph on Final_Score
	nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(df[["Final_Score"]])
	adj  = nbrs.kneighbors_graph(mode="connectivity")
	G    = nx.from_scipy_sparse_array(adj)
	G    = nx.relabel_nodes(G, {i: sid for i, sid in enumerate(df.student_id)})

	# node features
	feats   = StandardScaler().fit_transform(df[["Final_Score", "life_satisfaction"]])
	x_tensor = torch.tensor(feats, dtype=torch.float)
	id_map   = {sid: idx for idx, sid in enumerate(df.student_id)}
	e0, e1   = zip(*((id_map[u], id_map[v]) for u, v in G.edges()))
	edge_idx = torch.tensor([e0, e1], dtype=torch.long)

	data = Data(x=x_tensor, edge_index=edge_idx)

	# GNN training (tiny)
	model = GNNModel(input_dim=2)
	optim = torch.optim.Adam(model.parameters(), lr=0.01)
	model.train()
	for _ in range(10):
		optim.zero_grad()
		loss = F.mse_loss(model(data), data.x)
		loss.backward(); optim.step()

	emb = model(data).detach().numpy()

	# k-means clustering into ~5-person groups
	n_groups = max(2, len(df) // 5)
	labels   = KMeans(n_clusters=n_groups, random_state=42).fit_predict(emb)
	df["group"] = labels + 1        # groups start at 1
	return df[["student_id", "name", "group"]], G, labels
