# Compare_Test.py  – standalone CSV-based model comparer
# ─────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
import torch, torch.nn.functional as F, joblib
from pathlib import Path
from torch_geometric.data   import Data
from torch_geometric.nn     import GCNConv
from sklearn.cluster        import KMeans
from ortools.sat.python     import cp_model

from utils.ui_utils         import apply_global_styles
from utils.compare_utils    import (
    ENGINE_IDS,               # id ➜ label
    friend_conflict_counts,   # metric helper
    run_comparison, _ENGINE_FUNCS as ENGINE_FUNCS
)

# ─────────────────────────────────────────────────────────
# 0.  LLM helper (Ollama + graceful fallback)
# ─────────────────────────────────────────────────────────
def llm_compare(context: str) -> str:
    """Return more than 100 and less than 200 word comparison from Ollama, or raw context if unavailable."""
    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            "In no more than between 100 - 200 words, compare the classroom allocations for the "
            "following AI engines. Focus on how they differ in friendships kept, "
            "conflicts kept, and average class size.\n\n{context}"
        )
        llm = Ollama(model="mistral")
        return (prompt | llm).invoke({"context": context}).strip()
    except Exception:
        return context

# ─────────────────────────────────────────────────────────
# 1.  Embedded CP-SAT allocator  (same as before)
# ─────────────────────────────────────────────────────────
def compute_fitness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acad_norm"]       = df["Total_Score"]          / df["Total_Score"].max()
    df["well_norm"]       = df["life_satisfaction"]    / 10.0
    df["friends_norm"]    = df["closest_friend_count"] / df["closest_friend_count"].max()
    df["disrespect_norm"] = df["disrespected_by_peers"].map({"No": 1, "Yes": 0}).fillna(0)
    df["fitness"] = 0.4*df["acad_norm"] + 0.3*df["well_norm"] + 0.15*df["friends_norm"] + 0.15*df["disrespect_norm"]
    return df

def _greedy(fit,sids,k,cap):
    cnt=[0]*k; tot=[0]*k; out={}
    for sid,f in sorted(zip(sids,fit), key=lambda x:x[1]):
        room=min([c for c in range(k) if cnt[c]<cap], key=lambda c:tot[c])
        out[sid]=room; cnt[room]+=1; tot[room]+=f
    return out

def solve_constraints(df: pd.DataFrame, k:int, cap:int):
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

def cp_sat_allocate(df: pd.DataFrame, k=6, cap=30):
    fit_df=compute_fitness(df); assign=solve_constraints(fit_df,k,cap)
    res=pd.DataFrame({"Student_ID":fit_df["Student_ID"],
                      "Classroom":[assign.get(s,-1)+1 for s in fit_df["Student_ID"]]})
    return res.astype(int)

ENGINE_FUNCS["CP_SAT"] = cp_sat_allocate

# ─────────────────────────────────────────────────────────
# 2.  Embedded GNN allocator (unchanged, uses detach().numpy())
# ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH  = MODELS_DIR / "gnn_model1.pth"
SCALER_PATH = MODELS_DIR / "scaler1.pkl"

class GCNEncoder(torch.nn.Module):
    def __init__(self,in_dim,hid=64,emb=10):
        super().__init__(); self.conv1=GCNConv(in_dim,hid)
        self.conv2=GCNConv(hid,hid); self.conv3=GCNConv(hid,emb)
    def forward(self,data:Data):
        x,ei=data.x,data.edge_index
        x=F.relu(self.conv1(x,ei)); x=F.relu(self.conv2(x,ei)); return self.conv3(x,ei)

_DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_gnn_model=GCNEncoder(5).to(_DEVICE); _gnn_model.load_state_dict(torch.load(MODEL_PATH,map_location=_DEVICE),strict=False); _gnn_model.eval()
_scaler   = joblib.load(SCALER_PATH)
REQ_COLS  = ["Total_Score","Study_Hours_per_Week","Stress_Level (1-10)","is_bullied","feels_safe_in_class"]

def gnn_allocate(df_raw:pd.DataFrame,k=6):
    df=df_raw.copy()
    df["is_bullied"]=df["is_bullied"].map({"Yes":1,"No":0})
    X_scaled=_scaler.transform(df[REQ_COLS])
    n=len(df); ei=torch.arange(n).repeat(2,1).to(_DEVICE)
    emb=(_gnn_model(Data(x=torch.tensor(X_scaled,dtype=torch.float32).to(_DEVICE),edge_index=ei))
         .detach().cpu().numpy())
    labels=KMeans(n_clusters=k,random_state=42).fit_predict(emb)
    return pd.DataFrame({"Student_ID":df["Student_ID"],"Classroom":labels+1})

ENGINE_FUNCS["GNN"]=gnn_allocate

# ─────────────────────────────────────────────────────────
# 3.  Embedded Deep-RL allocator with loose loading
# ─────────────────────────────────────────────────────────
from utils.deep_rl_utils import QNetwork
DEEPRL_PATH = MODELS_DIR / "deep_rl_model.pth"
_DEEPRL_MODEL=None
def _load_dqn(state_dim:int, action_dim:int):
    mdl=QNetwork(state_dim,action_dim)
    if not DEEPRL_PATH.exists(): return mdl
    ckpt=torch.load(DEEPRL_PATH,map_location="cpu")
    good={k:v for k,v in ckpt.items() if k in mdl.state_dict() and v.size()==mdl.state_dict()[k].size()}
    sd=mdl.state_dict(); sd.update(good); mdl.load_state_dict(sd,strict=False); mdl.eval(); return mdl

def deep_rl_allocate(df_raw:pd.DataFrame,k=6,cap=30):
    global _DEEPRL_MODEL
    num_cols=df_raw.select_dtypes("number").columns.tolist()
    if "Student_ID" in num_cols: num_cols.remove("Student_ID")
    if not num_cols: raise RuntimeError("Deep-RL: no numeric features")
    state_dim=len(num_cols)
    if _DEEPRL_MODEL is None or _DEEPRL_MODEL.fc3.out_features!=k:
        _DEEPRL_MODEL=_load_dqn(state_dim,k)
    states=torch.tensor(df_raw[num_cols].values,dtype=torch.float32)
    with torch.no_grad(): actions=_DEEPRL_MODEL(states).argmax(1).cpu().numpy()
    return pd.DataFrame({"Student_ID":df_raw["Student_ID"],"Classroom":actions+1})

ENGINE_FUNCS["DEEP_RL"]=deep_rl_allocate

# ─────────────────────────────────────────────────────────
# 4.  Streamlit UI
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Compare Test", layout="wide")
apply_global_styles()
st.title("🧪 Stand-Alone Model Comparison")

upl=st.file_uploader("Upload CSV with social links & features",type="csv")
if not upl: st.stop()
df_raw=pd.read_csv(upl)

label2id={lbl:mid for mid,lbl in ENGINE_IDS.items()}
labels=st.multiselect("Select ≥ 2 models",list(label2id.keys()))
model_ids=[label2id[l] for l in labels]

a,b,c=st.columns(3)
with a: frac=st.slider("Fraction",0.1,1.0,0.25,0.05)
with b: max_n=st.number_input("Max rows",50,3000,200,50)
with c: seed =st.number_input("Seed",0,9999,42)

if len(model_ids)<2 or not st.button("🚀 Run comparison"): st.stop()

sample,G_base,pos,results,errors=run_comparison(df_raw,model_ids,frac,max_n,seed)
for mid,msg in errors.items(): st.error(f"{ENGINE_IDS.get(mid,mid)} failed: {msg}")
if len(results)<2: st.error("Need ≥ 2 successful engines."); st.stop()
st.success(f"Sample size: {len(sample)} students")

# ——— Visualisation + collect metrics ————————————————
metrics={}        # store per-engine summary
CLASS_COLORS=["red","green","blue"]

for pair in [results.items()][0::2]:
    cols=st.columns(len(pair))
    for (mid,df_alloc),col in zip(pair,cols):
        with col:
            name=ENGINE_IDS[mid]
            st.markdown(f"### {name}")
            cmap=dict(zip(df_alloc["Student_ID"],df_alloc["Classroom"]))
            friends_all,conf_all=friend_conflict_counts(df_alloc,G_base)
            avg_size=float(df_alloc.groupby("Classroom").size().mean())
            metrics[name]={"friends":friends_all,"conflicts":conf_all,"avg":avg_size}

            for cls,sub_df in list(df_alloc.groupby("Classroom"))[:3]:
                sub_nodes=set(sub_df["Student_ID"]); subgraph=G_base.subgraph(sub_nodes)
                f_in=d_in=0; traces=[]
                for rel,color in [("friend","green"),("disrespect","red")]:
                    xs,ys=[],[]
                    for u,v,d in subgraph.edges(data=True):
                        if d.get("relation_type")==rel:
                            x0,y0=pos[u]; x1,y1=pos[v]
                            xs+=[x0,x1,None]; ys+=[y0,y1,None]
                            f_in+=(rel=="friend"); d_in+=(rel=="disrespect")
                    if xs:
                        traces.append(go.Scatter(x=xs,y=ys,mode="lines",
                                                 line=dict(width=1,color=color),
                                                 hoverinfo="none",name=rel))
                xs,ys=zip(*[pos[n] for n in sub_nodes])
                node=go.Scatter(x=xs,y=ys,mode="markers",
                                marker=dict(size=8,color="yellow"),name="students",
                                hoverinfo="text",text=[str(n) for n in sub_nodes])
                fig=go.Figure(data=traces+[node],
                              layout=go.Layout(title=f"Class {cls}",
                                               hovermode="closest",
                                               margin=dict(l=10,r=10,t=30,b=10),
                                               showlegend=True))
                st.plotly_chart(fig,use_container_width=True)
                st.caption(f"👥 Students: **{len(sub_nodes)}** | "
                           f"✅ Friends kept: **{f_in}** | "
                           f"❌ Conflicts kept: **{d_in}**")
                
# ——— LLM explanation box ————————————————————————————————
ctx="\n".join(
    f"{n}: friends kept {m['friends']}, conflicts kept {m['conflicts']}, "
    f"avg size {m['avg']:.0f}"
    for n,m in metrics.items()
)
st.markdown("---"); st.subheader("🤖 Model comparison")
st.info(llm_compare(ctx))