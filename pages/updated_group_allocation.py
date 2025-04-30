# smart_allocation_interactive.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import plotly.express as px

# ────────────────────────────────────────────────────────────
# Model Definition
# ────────────────────────────────────────────────────────────

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, num_features)  # ✅ Add this

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x  # ✅ Still return embeddings (not decoder output)

# ────────────────────────────────────────────────────────────
# Load trained model and tools
# ────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(num_features=5, hidden_dim=64, embedding_dim=10)
model.load_state_dict(torch.load(r'C:\Users\Shoaib\Desktop\Ali Swinburne\classforge-demo\gnn_model1.pth', map_location=device))
model.to(device)
model.eval()

kmeans = joblib.load(r'C:\Users\Shoaib\Desktop\Ali Swinburne\classforge-demo\kmeans_model1.pkl')
scaler = joblib.load(r'C:\Users\Shoaib\Desktop\Ali Swinburne\classforge-demo\scaler1.pkl')

# ────────────────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Student Smart Allocator", layout="wide")
st.title("📚 Interactive Classroom Allocation")

uploaded = st.file_uploader(
    "Upload your full student data CSV (names, emails, etc. allowed)",
    type=["csv"]
)

if not uploaded:
    st.info("⬆️ Upload your CSV file to get started...")
    st.stop()

# Sidebar weights
st.sidebar.header("🔧 Feature Weights")
weights = {
    'Total_Score': st.sidebar.slider('Total Score Weight', 0.0, 100.0, 63.29, 0.1),
    'Study_Hours_per_Week': st.sidebar.slider('Study Hours Weight', 0.0, 40.0, 17.58, 0.1),
    'Stress_Level (1-10)': st.sidebar.slider('Stress Level Weight', 0.0, 10.0, 5.48, 0.1),
    'is_bullied': st.sidebar.slider('Is Bullied Weight', 0.0, 1.0, 0.50, 0.01),
    'feels_safe_in_class': st.sidebar.slider('Feels Safe Weight', 0.0, 5.0, 3.00, 0.1),
}


# Process CSV

df_raw = pd.read_csv(uploaded)

# Define required columns
target_columns = ['Total_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'is_bullied', 'feels_safe_in_class']

# Check for missing required columns
missing_cols = [col for col in target_columns if col not in df_raw.columns]
if missing_cols:
    st.error(f"❌ The following required columns are missing: {missing_cols}")
    st.stop()

target_columns = list(weights.keys())
missing_cols = [col for col in target_columns if col not in df_raw.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

if df_raw['is_bullied'].dtype == object:
    df_raw['is_bullied'] = df_raw['is_bullied'].str.lower().map({'yes': 1, 'no': 0})

# Drop rows with missing values
df_clean = df_raw.dropna(subset=target_columns).copy()
st.success(f"✅ Cleaned data: {len(df_clean)} students retained after dropping missing rows.")

# Apply weights
features = df_clean[target_columns].astype(float)
for col in target_columns:
    features[col] *= weights[col]

# Scale features
features_scaled = scaler.transform(features)

# ────────────────────────────────────────────────────────────
# GNN + KMeans Allocation
# ────────────────────────────────────────────────────────────

from sklearn.cluster import KMeans

with st.spinner("Allocating classrooms with custom weights..."):
    features_tensor = torch.tensor(features_scaled, dtype=torch.float).to(device)
    dummy_edge_index = torch.tensor([[i, i] for i in range(len(df_clean))], dtype=torch.long).t().contiguous().to(device)
    data = Data(x=features_tensor, edge_index=dummy_edge_index)

    with torch.no_grad():
        embeddings = model(data).detach().cpu().numpy()

    # Let user choose number of clusters dynamically
    num_clusters = st.sidebar.slider("Number of Classrooms", min_value=2, max_value=10, value=5, step=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    classrooms = kmeans.fit_predict(embeddings)

    df_clean["Classroom"] = classrooms

# ────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────

st.markdown("## 🏫 Classroom Assignment Results")
#st.dataframe(df_clean)

st.markdown("### 📋 Students per Classroom (Summary Table)")
class_counts = df_clean['Classroom'].value_counts().sort_index().reset_index()
class_counts.columns = ['Classroom', 'Number of Students']
st.table(class_counts)

# Explanation box for user context
st.markdown("### ℹ️ How These Classrooms Were Formed")

# List the weights used
st.markdown("#### 🔩 Feature Weights Applied:")
for feature, weight in weights.items():
    st.markdown(f"- **{feature}**: weight = `{weight}`")

# Explain number of clusters
st.markdown(f"#### 🏷 Number of Classrooms: `{num_clusters}`")

# Interpret clustering
st.info(
   """
    Each student was represented by a GNN-based embedding using the weighted combination
    of the 5 selected features. Then, KMeans clustering grouped these students into distinct
    classroom clusters based on similarity in those weighted traits.

    You can modify feature importance using the sliders in the sidebar to influence how students are grouped —
    for example, increasing 'Stress Level' weight makes clustering more sensitive to mental load,
    while reducing it emphasizes academics or safety more.

    Try changing feature weights or the number of classrooms and observe how group assignments update instantly!
    """
)


st.markdown("### 📊 Students per Classroom (Bar Chart)")
bar_fig = px.bar(class_counts, x='Classroom', y='Number of Students')
st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("### 🥧 Students per Classroom (Pie Chart)")
pie_fig = px.pie(class_counts, names='Classroom', values='Number of Students', title='Classroom Share')
st.plotly_chart(pie_fig, use_container_width=True)

# ────────────────────────────────────────────────────────────
# Download Button
# ────────────────────────────────────────────────────────────

csv = df_clean.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Allocated Students CSV", data=csv, file_name="allocated_students.csv", mime="text/csv")
