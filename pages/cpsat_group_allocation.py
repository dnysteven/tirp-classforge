# pages/cpsat_group_allocation.py
import streamlit as st
import math, pandas as pd

from utils.ui_utils import render_sidebar
from utils.cpsat_utils import load_csv, compute_fitness, solve_constraints, to_csv_bytes

# ── Page config & sidebar ────────────────────────────────────────────
st.set_page_config(page_title="CP-SAT Classroom Optimiser", layout="wide")
render_sidebar()
st.title("CP-SAT Group Allocation")

# ── File upload ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your dataset (CSV)", type="csv")
if not uploaded:
  st.info("Upload a CSV with academic, well-being, friendship and disrespect columns.")
  st.stop()

try:
  df_raw = load_csv(uploaded)
except Exception as err:
  st.error(f"Failed to load CSV: {err}")
  st.stop()

# ── Class capacity parameters ─────────────────────────────────────────
N = len(df_raw)
col_cap, col_cls = st.columns(2)
capacity = col_cap.number_input("Capacity per class", 1, value=30, step=1)
min_cls  = math.ceil(N / capacity)
num_cls  = col_cls.number_input("Number of classes", min_value=min_cls, value=min_cls, step=1)

# ── CP-SAT Oprimisation ──────────────────────────────────────────────────
df = compute_fitness(df_raw)
if "Name" not in df.columns and {"First_Name", "Last_Name"} <= set(df.columns):
  df["Name"] = df["First_Name"].str.strip() + " " + df["Last_Name"].str.strip()

if st.button("Run allocation"):
  with st.spinner("Optimising allocations…"):
    assignment = solve_constraints(df, num_cls, capacity)

  df["Class"] = df["Student_ID"].map(assignment).astype(int) + 1  # start at 1
  df = df[["Class"] + [c for c in df.columns if c != "Class"]]     # Class first
  st.success("Allocation complete!")

  tab_roster, tab_metrics = st.tabs(["Class rosters", "Average metrics"])

  # ─── Class rosters tab ───────────────────────────────────────────
  with tab_roster:
    editable = st.checkbox("Enable manual edits", value=False)
    df_edit = st.data_editor(
      df,
      disabled=not editable,
      num_rows="fixed",
      column_config={"Class": st.column_config.NumberColumn(min_value=1)},
      use_container_width=True,
    )
    if editable:
      df = df_edit

    st.markdown("### Class rosters")
    cols = st.columns(2)                                             # two tables per row
    for idx, (cls, sub) in enumerate(df.groupby("Class"), 1):
      with cols[(idx - 1) % 2]:
        st.markdown(f"**Class {cls}**")
        st.dataframe(
          sub[["Student_ID", "Name", "fitness", "Total_Score"]],
          hide_index=True,
          use_container_width=True,
        )

    st.download_button(
      "Download allocation (CSV)",
      to_csv_bytes(df),
      file_name="cpsat_group_allocation.csv",
      mime="text/csv",
    )

  # ─── Average metrics tab ─────────────────────────────────────────
  with tab_metrics:
    st.markdown("**Average fitness per class**")
    st.bar_chart(df.groupby("Class")["fitness"].mean())
    st.markdown("**Average total score per class**")
    st.bar_chart(df.groupby("Class")["Total_Score"].mean())
    st.markdown("**Average friends per class**")
    st.bar_chart(df.groupby("Class")["closest_friend_count"].mean())
    st.markdown("**Disrespected % per class**")
    st.bar_chart(
      df.groupby("Class")["disrespected_by_peers"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
