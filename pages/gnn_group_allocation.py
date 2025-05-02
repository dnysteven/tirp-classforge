# pages/gnn_group_allocation.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_utils import render_sidebar
from utils.gnn_utils import allocate, REQUIRED_COLS
from utils.cpsat_utils import to_csv_bytes

# ── page chrome ──────────────────────────────────────────────────────
st.set_page_config(page_title="GNN Classroom Allocator", layout="wide")
render_sidebar()
st.title("📚 GNN-based Classroom Allocation")

# ── upload ───────────────────────────────────────────────────────────
upl = st.file_uploader("Upload student CSV (must include required columns)", type="csv")
if not upl:
	st.info("⬆️ Upload your CSV to begin.")
	st.stop()

df_raw = pd.read_csv(upl)

missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing:
	st.error(f"Missing columns: {missing}")
	st.stop()

# ── feature-weight sliders ───────────────────────────────────────────
st.markdown("### 🎛 Feature weights")
w_cols = st.columns(len(REQUIRED_COLS))
default_w = {
	"Total_Score": 63.29,
	"Study_Hours_per_Week": 17.58,
	"Stress_Level (1-10)": 5.48,
	"is_bullied": 0.50,
	"feels_safe_in_class": 3.0,
}
weights = {
	col: w_cols[i].slider(
		col, 0.0, 100.0 if col == "Total_Score" else 10.0,
		value=default_w[col], step=0.1
	)
	for i, col in enumerate(REQUIRED_COLS)
}

num_cls = st.slider("Number of classrooms", 2, 10, 5, 1)

# ── allocation ───────────────────────────────────────────────────────
with st.spinner("Allocating classrooms…"):
    df_alloc = allocate(df_raw, weights, num_cls)

# build Name column if available
if "Name" not in df_alloc.columns and {"First_Name", "Last_Name"} <= set(df_alloc.columns):
	df_alloc["Name"] = (
		df_alloc["First_Name"].astype(str).str.strip()
		+ " "
		+ df_alloc["Last_Name"].astype(str).str.strip()
	)

# reorder columns: Classroom, Student_ID, Name, …
front_cols = ["Classroom", "Student_ID"]
if "Name" in df_alloc.columns:
	front_cols.append("Name")
df_alloc = df_alloc[front_cols + [c for c in df_alloc.columns if c not in front_cols]]

# ── tabs: rosters first, visualisations second ───────────────────────
tab_roster, tab_vis = st.tabs(["Class rosters", "Visualisations"])

# ───────── Class rosters tab ─────────────────────────────────────────
with tab_roster:
	editable = st.checkbox("Enable manual edits", value=False)
	df_edit = st.data_editor(
		df_alloc,
		disabled=not editable,
		num_rows="fixed",
		column_config={"Classroom": st.column_config.NumberColumn(min_value=1)},
		use_container_width=True,
	)
	if editable:
		df_alloc = df_edit

	cols = st.columns(2)                               # two tables per row
	for idx, (cls, sub) in enumerate(df_alloc.groupby("Classroom"), 1):
		with cols[(idx - 1) % 2]:
			st.markdown(f"**Class {cls}**")
			display_cols = ["Student_ID", "Name"] if "Name" in sub.columns else ["Student_ID"]
			display_cols += ["Total_Score", "Study_Hours_per_Week"]
			st.dataframe(sub[display_cols], hide_index=True, use_container_width=True)

	st.download_button(
		"📥 Download allocation (CSV)",
		to_csv_bytes(df_alloc),
		file_name="gnn_group_allocation.csv",
		mime="text/csv",
	)

# ───────── Visualisations tab ────────────────────────────────────────
with tab_vis:
  # Interpret clustering
  st.markdown(f"#### 🏷 Number of Classrooms: `{num_cls}`")
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
  
  st.markdown("### 📊 Students per Classroom – Bar")
  bar_df = df_alloc["Classroom"].value_counts().sort_index().reset_index()
  bar_df.columns = ["Classroom", "Students"]
  st.bar_chart(bar_df, x="Classroom", y="Students")
  
  st.markdown("### 🥧 Classroom distribution – Pie")
  st.plotly_chart(
		px.pie(bar_df, names="Classroom", values="Students", title="Classroom share"),
		use_container_width=True,
	)
