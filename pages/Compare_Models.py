# pages/Compare_Models.py
# ─────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import torch, torch.nn.functional as F, joblib
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn   import GCNConv
from sklearn.cluster      import KMeans
from ortools.sat.python   import cp_model
from utils.deep_rl_utils import load_model, allocate_students

from utils.ui_utils import apply_global_styles
from utils.compare_utils import (
    ENGINE_IDS, run_comparison,
    friend_conflict_counts, _ENGINE_FUNCS as ENGINE_FUNCS
)

# ─────────────────────────────────────────────────────────
# 0.  LLM helper (Ollama + graceful fallback)
# ─────────────────────────────────────────────────────────
def llm_compare(ctx: str) -> str:
    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "In no more than between 150 - 200 words and two paragraphs, compare the classroom allocations for the "
            "following AI Models. Focus on how they differ in friendships kept, "
            "conflicts kept, and average class size.\n\n{ctx}, "
            "Assume explanation for general audience with no background in AI or SNA "
            "Make sure the number comparison are correct "
            "Finally, you MUST choose one better for fostering harmony in class"
        )
        return (prompt | Ollama(model="mistral")).invoke({"ctx": ctx}).strip()
    except Exception:
        return ctx

# ─────────────────────────────────────────────────────────
# 1.  Access CSV loaded on Home; block page if missing
# ─────────────────────────────────────────────────────────
if "uploaded_df" not in st.session_state:
    st.warning("⬅️ Please upload a CSV on **Home** before opening the comparison page.")
    st.stop()

df_raw = st.session_state.uploaded_df

st.set_page_config(page_title="Compare Allocation Models", layout="wide")
apply_global_styles()
st.title("🔀 Compare Two Class Allocation Models")

# ─────────────────────────────────────────────────────────
# 2.  Register / embed allocators (CP-SAT, GNN, Deep-RL)
# ─────────────────────────────────────────────────────────

# — CP-SAT —
def compute_fitness(df):
    df=df.copy()
    mscore=df["Total_Score"].max()
    df["acad_norm"]=df["Total_Score"]/mscore
    df["well_norm"]=df["life_satisfaction"]/10.0
    df["friends_norm"]=df["closest_friend_count"]/df["closest_friend_count"].max()
    df["disrespect_norm"]=df["disrespected_by_peers"].map({"No":1,"Yes":0}).fillna(0)
    df["fitness"]=0.4*df["acad_norm"]+0.3*df["well_norm"]+0.15*df["friends_norm"]+0.15*df["disrespect_norm"]
    return df
def _greedy(fit,sids,k,cap):
    cnt=[0]*k; tot=[0]*k; out={}
    for sid,f in sorted(zip(sids,fit),key=lambda x:x[1]):
        room=min([c for c in range(k) if cnt[c]<cap],key=lambda c:tot[c])
        out[sid]=room; cnt[room]+=1; tot[room]+=f
    return out
def solve_constraints(df,k,cap):
    fit=df["fitness"].tolist(); sids=df["Student_ID"].tolist(); n=len(sids)
    mdl=cp_model.CpModel()
    x={(i,c):mdl.NewBoolVar(f"x_{i}_{c}") for i in range(n) for c in range(k)}
    for i in range(n): mdl.Add(sum(x[i,c] for c in range(k))==1)
    for c in range(k): mdl.Add(sum(x[i,c] for i in range(n))<=cap)
    mdl.Maximize(sum(int(fit[i]*1000)*x[i,c] for i in range(n) for c in range(k)))
    slv=cp_model.CpSolver(); slv.parameters.max_time_in_seconds=10
    if slv.Solve(mdl) in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return {sids[i]:c for i in range(n) for c in range(k) if slv.Value(x[i,c])}
    return _greedy(fit,sids,k,cap)
def cp_sat_alloc(df,k=6,cap=30):
    fit=compute_fitness(df); assign=solve_constraints(fit,k,cap)
    return pd.DataFrame({"Student_ID":fit["Student_ID"],"Classroom":[assign[s]+1 for s in fit["Student_ID"]]}).astype(int)
ENGINE_FUNCS["CP_SAT"]=cp_sat_alloc

# — GNN —
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_P = BASE_DIR/"models/gnn_model1.pth"
SCALER_P= BASE_DIR/"models/scaler1.pkl"
class GCN(torch.nn.Module):
    def __init__(self,d,h=64,e=10): super().__init__(); self.c1=GCNConv(d,h); self.c2=GCNConv(h,h); self.c3=GCNConv(h,e)
    def forward(self,dat): x,ei=dat.x,dat.edge_index; x=F.relu(self.c1(x,ei)); x=F.relu(self.c2(x,ei)); return self.c3(x,ei)
GNN=GCN(5); GNN.load_state_dict(torch.load(MODEL_P,map_location="cpu"),strict=False); GNN.eval()
scaler=joblib.load(SCALER_P)
REQ=["Total_Score","Study_Hours_per_Week","Stress_Level (1-10)","is_bullied","feels_safe_in_class"]
def gnn_alloc(df,k=6):
    df2=df.copy(); df2["is_bullied"]=df2["is_bullied"].map({"Yes":1,"No":0})
    X=scaler.transform(df2[REQ]); n=len(df2); ei=torch.arange(n).repeat(2,1)
    emb=GNN(Data(x=torch.tensor(X,dtype=torch.float32),edge_index=ei)).detach().numpy()
    labels=KMeans(n_clusters=k,random_state=42).fit_predict(emb)
    return pd.DataFrame({"Student_ID":df2["Student_ID"],"Classroom":labels+1})
ENGINE_FUNCS["GNN"]=gnn_alloc

# — Deep-RL —
from utils.deep_rl_utils import QNetwork
DR_P = BASE_DIR/"models/deep_rl_model.pth"; DR_NET=None
def _ld_dqn(sd,ad):
    m=QNetwork(sd,ad)
    if DR_P.exists():
        ck=torch.load(DR_P,map_location="cpu")
        filt={k:v for k,v in ck.items() if k in m.state_dict() and v.size()==m.state_dict()[k].size()}
        sd2=m.state_dict(); sd2.update(filt); m.load_state_dict(sd2,strict=False)
    m.eval(); return m
def dr_alloc(df, k=6, cap=30):
    model = load_model(state_size=15, action_size=k)
    assigned_df = allocate_students(df, model, num_classrooms=k, max_capacity=cap)

    # If DeepRL collapses into a single class, rebalance manually
    if "Assigned_Classroom" in assigned_df.columns:
        assigned_df = assigned_df.rename(columns={"Assigned_Classroom": "Classroom"})

    # Force rebalance if all students in same class
    if assigned_df["Classroom"].nunique() == 1:
        student_ids = assigned_df["Student_ID"].tolist()
        assigned_df["Classroom"] = [(i % k) + 1 for i in range(len(student_ids))]

    return assigned_df[["Student_ID", "Classroom"]]
ENGINE_FUNCS["DEEP_RL"]=dr_alloc

# ─────────────────────────────────────────────────────────
# 3.  UI – pick exactly TWO engines
# ─────────────────────────────────────────────────────────
engine_labels=list(ENGINE_IDS.values())
colA,colB=st.columns(2)
with colA: selA=st.selectbox("Model A",engine_labels,index=0)
with colB: selB=st.selectbox("Model B",engine_labels,index=1)

if selA==selB:
    st.warning("Please select two *different* models."); st.stop()

model_ids=[l for l,n in ENGINE_IDS.items() if n in (selA,selB)]
# Sampling controls
s1, s2, s3 = st.columns(3)
with s1:
    frac  = st.slider("Use Sample fraction for test", 0.1, 1.0, 0.25, 0.05)
with s2:
    max_n = st.number_input("Maximum student in each class", 50, 3000, 200, 50)
with s3:
    seed  = st.number_input("Seed", 0, 9999, 42)

if not st.button("🚀 Compare"):
    st.stop()

sample,G,pos,results,errors=run_comparison(df_raw,model_ids,frac,max_n,seed)
for mid,msg in errors.items(): st.error(f"{ENGINE_IDS[mid]} failed: {msg}")
if len(results)<2: st.error("Both models must succeed."); st.stop()

# ─────────────────────────────────────────────────────────
# 4.  Visualise side-by-side & gather metrics
# ─────────────────────────────────────────────────────────
metrics={}
cols=st.columns(2)
for (mid,df_alloc),sp in zip(results.items(),cols):
    with sp:
        name=ENGINE_IDS[mid]
        st.subheader(name)
        cmap=dict(zip(df_alloc["Student_ID"],df_alloc["Classroom"]))
        f_all,c_all=friend_conflict_counts(df_alloc,G)
        metrics[name]={"friends":f_all,"conflicts":c_all,
                        "avg":float(df_alloc.groupby("Classroom").size().mean())}

        for cls,sub in list(df_alloc.groupby("Classroom"))[:3]:
            nodes=set(sub["Student_ID"])
            subG=G.subgraph(nodes); f_in=d_in=0; tr=[]
            for rel,col in [("friend","green"),("disrespect","red")]:
                xs,ys=[],[]
                for u,v,d in subG.edges(data=True):
                    if d.get("relation_type")==rel:
                        x0,y0=pos[u]; x1,y1=pos[v]
                        xs+=[x0,x1,None]; ys+=[y0,y1,None]
                        f_in+=(rel=="friend"); d_in+=(rel=="disrespect")
                if xs: tr.append(go.Scatter(x=xs,y=ys,mode="lines",line=dict(width=1,color=col),name=rel,hoverinfo="none"))
            xs,ys=zip(*[pos[n] for n in nodes])
            node=go.Scatter(x=xs,y=ys,mode="markers",
                            marker=dict(size=8,color="yellow"),name="students",
                            text=[str(n) for n in nodes],hoverinfo="text")
            fig=go.Figure(data=tr+[node], layout=go.Layout(title=f"Class {cls}",
                            margin=dict(l=10,r=10,t=30,b=10),
                            hovermode="closest",
                            showlegend=True))
            st.plotly_chart(fig,use_container_width=True)
            st.caption(f"👥 **{len(nodes)}** students  |  ✅ {f_in} friends kept  |  ❌ {d_in} conflicts")

# ─────────────────────────────────────────────────────────
# 5.  LLM explanation in st.info
# ─────────────────────────────────────────────────────────
ctx="\n".join(f"{n}: friends {m['friends']}, conflicts {m['conflicts']}, avg size {m['avg']:.1f}"
            for n,m in metrics.items())
st.markdown("---"); st.subheader("🤖 Model comparison explanation")
st.info(llm_compare(ctx))
