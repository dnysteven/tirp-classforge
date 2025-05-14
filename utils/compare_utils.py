# utils/compare_utils.py  – central hub for model comparison
# -----------------------------------------------------------
from __future__ import annotations
import pandas as pd, numpy as np, networkx as nx
from pathlib import Path

# ---- public mapping used by Home & Compare pages -------------
ENGINE_IDS = {
    "CP_SAT":   "CP-SAT",
    "GNN":      "GNN + K-Means",
    "GA":       "Genetic Algorithm",
    "DEEP_RL":  "Deep-RL (DQN)",
    "SCENARIO": "Scenario-Based AI",
}

# ===== 1. Sampling helpers =====================================
def sample_df(df: pd.DataFrame, frac: float, max_n: int, seed: int) -> pd.DataFrame:
    n_rows = min(int(len(df) * frac), max_n)
    return (
        df.sample(frac=1, random_state=seed)   # shuffle
          .iloc[:n_rows]
          .reset_index(drop=True)
    )

def build_graph(df_sample: pd.DataFrame, seed: int):
    from utils.ga_utils import simulate_graph
    G = simulate_graph(df_sample)
    pos = nx.spring_layout(G, seed=seed)
    return G, pos

# ===== 2. Per-engine default runners ===========================
from utils.cpsat_utils   import compute_fitness, solve_constraints
from utils.gnn_utils     import allocate as gnn_allocate
from utils.ga_utils      import load_and_scale, setup_deap, run_ga, simulate_graph
from utils.deep_rl_utils import load_model, allocate_students
from utils.scenario_utils import run_scenario_clustering

def _cp_sat(df: pd.DataFrame) -> pd.DataFrame:
    from utils.cpsat_utils import compute_fitness, solve_constraints

    try:
        fit_df = compute_fitness(df.copy())
    except Exception:
        fit_df = df.copy()
        if "Student_ID" not in fit_df.columns:
            fit_df["Student_ID"] = list(range(len(fit_df)))
        fit_df["fitness"] = 1.0

    try:
        fitness_vals = fit_df["fitness"].tolist()
        student_ids = fit_df["Student_ID"].tolist()
        # solve_constraints likely returns a list — convert it
        assigned = solve_constraints(fitness_vals, student_ids, 30)
        assigned_map = dict(zip(student_ids, assigned))  # safe dict
    except Exception as e:
        raise RuntimeError(f"CP-SAT solver failed: {e}")

    return pd.DataFrame({
        "Student_ID": student_ids,
        "Classroom": [assigned_map.get(sid, -1) + 1 for sid in student_ids],
    })


def _gnn(df: pd.DataFrame) -> pd.DataFrame:
    from utils.gnn_utils import allocate
    weights = [0.2] * 5

    try:
        res = allocate(df.copy(), weights=weights, n_cls=6)
    except Exception as e:
        raise RuntimeError(f"GNN allocator failed internally: {e}")

    # Handle multiple return formats
    if isinstance(res, pd.DataFrame):
        df_alloc = res
    elif isinstance(res, (list, tuple)) and isinstance(res[0], pd.DataFrame):
        df_alloc = res[0]
    else:
        raise RuntimeError("GNN allocator returned unexpected type.")

    if "group" not in df_alloc.columns or "student_id" not in df_alloc.columns:
        raise RuntimeError("Missing expected columns 'group' or 'student_id'.")

    return df_alloc.rename(columns={
        "group": "Classroom",
        "student_id": "Student_ID"
    })[["Student_ID", "Classroom"]]

def _ga(df: pd.DataFrame) -> pd.DataFrame:
    scaled,_ = load_and_scale(df)
    tb = setup_deap(scaled, simulate_graph(scaled), 6, (0.33,0.33,0.34))
    best,_ = run_ga(tb, pop_size=40, gens=10)
    return pd.DataFrame({"Student_ID": scaled["Student_ID"],
                         "Classroom": np.array(best)+1})

def _deep_rl(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model(Path("models") / "deep_rl_model.pth", state_size=15)
    roster,_ = allocate_students(df, model, num_classrooms=6, max_capacity=30)
    return roster[["Student_ID","Classroom"]]

def _scenario(df: pd.DataFrame) -> pd.DataFrame:
    out,*_ = run_scenario_clustering(df, "High Academic Imbalance", 6)
    return out.rename(columns={"Scenario_Group":"Classroom"})[["Student_ID","Classroom"]]

_ENGINE_FUNCS = {
    "CP_SAT":   _cp_sat,
    "GNN":      _gnn,
    "GA":       _ga,
    "DEEP_RL":  _deep_rl,
    "SCENARIO": _scenario,
}

# ===== 3. Master helper for Compare page =======================
def run_comparison(
    full_df: pd.DataFrame,
    model_ids: list[str],
    frac: float,
    max_n: int,
    seed: int,
):
    """
    Returns:
      sample_df, G_base, pos, results_dict, errors_dict
    """
    sample = sample_df(full_df, frac, max_n, seed)
    G, pos = build_graph(sample, seed)

    results, errors = {}, {}
    for mid in model_ids:
        fn = _ENGINE_FUNCS.get(mid)
        if fn is None:
            errors[mid] = "engine not implemented"
            continue
        try:
            results[mid] = fn(sample)
        except Exception as e:
            errors[mid] = str(e)

    return sample, G, pos, results, errors

# ===== 4. Friend / Conflict edge counts ==========================
def friend_conflict_counts(df_alloc: pd.DataFrame, G: nx.Graph) -> tuple[int, int]:
    """
    Returns:
      (number of friend edges inside same class,
      number of disrespect edges inside same class)
    """
    cmap = dict(zip(df_alloc["Student_ID"], df_alloc["Classroom"]))
    f_in = d_in = 0
    for u, v, d in G.edges(data=True):
        if cmap.get(u) != cmap.get(v): continue
        if d["relation_type"] == "friend":
            f_in += 1
        elif d["relation_type"] == "disrespect":
            d_in += 1
    return f_in, d_in