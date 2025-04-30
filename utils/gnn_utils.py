"""
GNN-based classroom allocator.
• Loads a pretrained GCN encoder, feature scaler and (optionally) a pre-fit
  K-Means model from the *models/* folder — all paths are **relative**.
• Accepts per-feature weights and number-of-classrooms.
• Returns a DataFrame with a new `Classroom` column numbered from 1.
"""
#utils/gnn_utils.py
from pathlib import Path
import joblib, torch, pandas as pd, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

# ── artefact paths (project-relative) ────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH  = MODELS_DIR / "gnn_model1.pth"
SCALER_PATH = MODELS_DIR / "scaler1.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans_model1.pkl"

# ── GCN encoder definition ───────────────────────────────────────────
class GCNEncoder(torch.nn.Module):
	def __init__(self, in_dim: int, hid: int = 64, emb: int = 10):
		super().__init__()
		self.conv1 = GCNConv(in_dim, hid)
		self.conv2 = GCNConv(hid, hid)
		self.conv3 = GCNConv(hid, emb)

	def forward(self, data: Data):
		x, ei = data.x, data.edge_index
		x = F.relu(self.conv1(x, ei))
		x = F.relu(self.conv2(x, ei))
		return self.conv3(x, ei)

# ── load artefacts once ──────────────────────────────────────────────
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = GCNEncoder(in_dim=5).to(_DEVICE)
_state = torch.load(MODEL_PATH, map_location=_DEVICE)
_model.load_state_dict(_state, strict=False)   # ignore extra decoder.* keys
_model.eval()

_scaler = joblib.load(SCALER_PATH)
_kmeans_prefit = joblib.load(KMEANS_PATH) if KMEANS_PATH.exists() else None

# ── public constants & API ───────────────────────────────────────────
REQUIRED_COLS = [
	"Total_Score",
	"Study_Hours_per_Week",
	"Stress_Level (1-10)",
	"is_bullied",
	"feels_safe_in_class",
]

def allocate(
	df_raw: pd.DataFrame,
	weights: dict[str, float],
	n_cls: int
) -> pd.DataFrame:
	"""
	Allocate students into `n_cls` classrooms using GNN embeddings + K-Means.

	Parameters
	----------
	df_raw : DataFrame containing REQUIRED_COLS plus Student_ID
	weights: dict of feature weights keyed by REQUIRED_COLS
	n_cls  : number of classrooms/clusters

	Returns
	-------
	DataFrame with new column `Classroom` numbered from 1.
	"""
	df = df_raw.copy()

	# normalise boolean column
	if df["is_bullied"].dtype == object:
		df["is_bullied"] = df["is_bullied"].str.lower().map({"yes": 1, "no": 0})

	# apply user weights & scale
	X = df[REQUIRED_COLS].astype(float)
	for col in REQUIRED_COLS:
		X[col] *= weights[col]
	X_scaled = _scaler.transform(X)

	# trivial self-edge graph (replace with real edges when available)
	n = len(df)
	edge_idx = torch.arange(n, dtype=torch.long).repeat(2, 1).to(_DEVICE)

	data = Data(
		x=torch.tensor(X_scaled, dtype=torch.float32).to(_DEVICE),
		edge_index=edge_idx,
	)

	with torch.no_grad():
		emb = _model(data).cpu().numpy()

	# reuse pre-fit KMeans if it matches; else fit fresh
	if _kmeans_prefit is not None and _kmeans_prefit.n_clusters == n_cls:
		labels = _kmeans_prefit.predict(emb)
	else:
		labels = KMeans(n_clusters=n_cls, random_state=42).fit_predict(emb)

	df["Classroom"] = labels + 1   # start at 1
	return df
