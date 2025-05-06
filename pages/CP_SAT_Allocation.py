import streamlit as st, math, pandas as pd
from utils.cpsat_utils import load_csv, compute_fitness, solve_constraints, to_csv_bytes

st.set_page_config(page_title="CP-SAT Classroom Optimiser", layout="wide")
st.title("CP‑SAT Constraint Optimisation")

# dataframe source ---------------------------------------------------
if "uploaded_df" not in st.session_state:
    st.session_state["redirect_warning"] = True
    if hasattr(st, "switch_page"):
        st.switch_page("Home.py")
    else:
        st.experimental_set_query_params(page="Home.py")
        st.experimental_rerun()
else:
    df_raw = st.session_state.uploaded_df.copy()

# parameters ---------------------------------------------------------
N=len(df_raw)
cap_col, cls_col = st.columns(2)
capacity = cap_col.number_input("Capacity per class",1,50,30,1)
min_cls  = math.ceil(N/capacity)
num_cls  = cls_col.number_input("Number of classes",min_cls,1000,min_cls,1)

df = compute_fitness(df_raw)
if "Name" not in df.columns and {"First_Name", "Last_Name"}.issubset(df.columns):
    df["Name"] = (
        df["First_Name"].astype(str).str.strip()
        + " "
        + df["Last_Name"].astype(str).str.strip()
    )

if st.button("Run allocation"):
    with st.spinner("Optimising…"):
        assign = solve_constraints(df,num_cls,capacity)
    df["Class"]=df["Student_ID"].map(assign).astype(int)+1
    df = df[["Class","Student_ID","Name"]+[c for c in df.columns if c not in ("Class","Student_ID","Name")]]

    tab_roster, tab_metrics = st.tabs(["Class rosters","Average metrics"])
    with tab_roster:
        editable=st.checkbox("Enable manual edits",value=False)
        df_edit=st.data_editor(df,disabled=not editable,num_rows="fixed",
                                column_config={"Class":st.column_config.NumberColumn(min_value=1)},
                                use_container_width=True)
        if editable: df=df_edit
        cols=st.columns(2)
        for i,(cls,sub) in enumerate(df.groupby("Class"),1):
            with cols[(i-1)%2]:
                st.markdown(f"**Class {cls}**")
                st.dataframe(sub[["Student_ID","Name","fitness","Total_Score"]],
                            hide_index=True,use_container_width=True)
        st.download_button("Download CSV",to_csv_bytes(df),"cpsat_alloc.csv","text/csv")
    with tab_metrics:
        st.bar_chart(df["Class"].value_counts().sort_index(),use_container_width=True)
        st.bar_chart(df.groupby("Class")["fitness"].mean(),use_container_width=True)
