# utils/cpsat_utils.py
import io, math, pandas as pd
from ortools.sat.python import cp_model

# ---------------- Data Helpers --------------------------------------
def load_csv(file) -> pd.DataFrame:
  df = pd.read_csv(file)
  df["Student_ID"] = df["Student_ID"].astype(str)
  return df

def compute_fitness(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["acad_norm"]   = df["Total_Score"] / df["Total_Score"].max()
  df["well_norm"]   = df["life_satisfaction"] / 10.0
  df["friends_norm"]= df["closest_friend_count"] / df["closest_friend_count"].max()
  df["disrespect_norm"] = df["disrespected_by_peers"].map({"No": 1, "Yes": 0}).fillna(0)
  w_acad, w_well, w_friend, w_disrespect = 0.4, 0.3, 0.15, 0.15
  df["fitness"] = (
    w_acad  * df["acad_norm"] +
    w_well  * df["well_norm"] +
    w_friend* df["friends_norm"] +
    w_disrespect * df["disrespect_norm"]
  )
  return df

# ---------------- CSV download helper -------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
  buf = io.BytesIO()
  df.to_csv(buf, index=False)
  return buf.getvalue()

# ---------------- Greedy Fallback -----------------------------------
def _greedy_assign(fitness, sids, n_cls, cap):
  counts   = [0]*n_cls
  totals   = [0.0]*n_cls
  assign   = {}
  for sid, fit in sorted(zip(sids, fitness), key=lambda x:x[1]):
    room = min(
      [c for c in range(n_cls) if counts[c] < cap],
      key=lambda c: totals[c]
    )
    assign[sid] = room
    counts[room] += 1
    totals[room] += fit
  return assign

# ---------------- CP-SAT Optimiser ----------------------------------
def solve_constraints(df: pd.DataFrame, n_cls: int, cap: int):
  fitness = df["fitness"].tolist()
  sids    = df["Student_ID"].tolist()
  N       = len(sids)
  
  # safety cap
  if N * n_cls > 200_000:
    return _greedy_assign(fitness, sids, n_cls, cap)

  mdl = cp_model.CpModel()
  x = {(i,c): mdl.NewBoolVar(f"x_{i}_{c}") for i in range(N) for c in range(n_cls)}
  # each student in exactly one class
  for i in range(N):
    mdl.Add(sum(x[i,c] for c in range(n_cls)) == 1)
  # capacity
  for c in range(n_cls):
    mdl.Add(sum(x[i,c] for i in range(N)) <= cap)
  # maximise fitness
  terms = [int(fitness[i]*1000) * x[i,c] for i in range(N) for c in range(n_cls)]
  mdl.Maximize(sum(terms))

  solver = cp_model.CpSolver()
  solver.parameters.max_time_in_seconds = 10
  solver.parameters.num_search_workers = 8
  if solver.Solve(mdl) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    return {sids[i]: c for i in range(N) for c in range(n_cls) if solver.Value(x[i,c])}
  # fall back
  return _greedy_assign(fitness, sids, n_cls, cap)
