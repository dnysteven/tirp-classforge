# utils/compare_utils.py  – central hub for model comparison
# -----------------------------------------------------------
from __future__ import annotations
import pandas as pd, numpy as np, networkx as nx
import itertools, ast
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
        df.sample(frac=1, random_state=seed)
        .iloc[:n_rows]
        .reset_index(drop=True)
    )
    
# ===== 2. SNA Graph builders ===================================
def build_social_graph(
    df: pd.DataFrame,
    *,
    use_synthetic: bool = True,
    friend_hours_tol: int = 2,
    top_friend_q: float = 0.75,
    hi_satisf: int = 7,
    hi_stress: int = 8,
) -> nx.Graph:
    """
    Returns an undirected graph with edge attribute 'relation_type'
    (= 'friend' or 'disrespect').

    • If `use_synthetic` and the columns Friends / Disrespected_By exist, those lists are used directly.
    • Otherwise, edges are inferred heuristically from existing columns.
    """
    G = nx.Graph()
    ids = df["Student_ID"].tolist()
    G.add_nodes_from(ids)

    # --- 1) use synthetic lists if present ---
    if use_synthetic and {"Friends", "Disrespected_By"} <= set(df.columns):
        for _, row in df.iterrows():
            sid = row["Student_ID"]

            # Friends list
            friends = row["Friends"]
            if isinstance(friends, str):
                friends = ast.literal_eval(friends)
            for fid in friends:
                if fid in ids:
                    G.add_edge(sid, fid, relation_type="friend")

            # Disrespect list
            enemies = row["Disrespected_By"]
            if isinstance(enemies, str):
                enemies = ast.literal_eval(enemies)
            for eid in enemies:
                if eid in ids:
                    G.add_edge(sid, eid, relation_type="disrespect")
        return G

    # --- 2) heuristic inference from other columns ---
    top_cut = df["closest_friend_count"].quantile(top_friend_q)

    # convenient Series look-ups
    fcnt   = df.set_index("Student_ID")["closest_friend_count"]
    hours  = df.set_index("Student_ID")["Study_Hours_per_Week"]
    groupw = df.set_index("Student_ID")["prefers_group_work"]                # Y/N
    advice = df.set_index("Student_ID")["gets_schoolwork_advice_from_friends"]
    feedbk = df.set_index("Student_ID")["receives_learning_feedback_from_peers"]
    satisf = df.set_index("Student_ID")["life_satisfaction"].fillna(0)
    safe   = df.set_index("Student_ID")["feels_safe_in_class"].fillna(0)
    comfort= df.set_index("Student_ID")["feels_comfortable_at_school"].fillna(0)
    iso    = df.set_index("Student_ID")["feels_isolated_due_to_opinion"].fillna(0)
    stress = df.set_index("Student_ID")["Stress_Level (1-10)"].fillna(0)
    bully  = df.set_index("Student_ID")["is_bullied"].map({"Yes": 1, "No": 0}).fillna(0)
    disrp  = df.set_index("Student_ID")["disrespected_by_peers"].map({"Yes": 1, "No": 0}).fillna(0)

    for u, v in itertools.combinations(ids, 2):
        score_f = score_d = 0

        # -------  A.  Collaboration / study-buddy  ---------------------------
        # both like group work
        if groupw[u] == "Y" and groupw[v] == "Y":
            score_f += 2

        # similar study hours
        if abs(hours[u] - hours[v]) <= friend_hours_tol:
            score_f += 2

        # someone gives OR receives academic advice/feedback
        if (
            advice[u] == "Y" or advice[v] == "Y" or
            feedbk[u] == "Y" or feedbk[v] == "Y"
        ):
            score_f += 2

        # -------  B.  Well-being support  ------------------------------------
        # one student isolated/bullied  +  the other high comfort & safety
        lonely_u = iso[u] >= 6 or bully[u] or disrp[u]
        lonely_v = iso[v] >= 6 or bully[v] or disrp[v]
        supporter_u = comfort[u] >= hi_satisf and safe[u] >= 4
        supporter_v = comfort[v] >= hi_satisf and safe[v] >= 4

        if (lonely_u and supporter_v) or (lonely_v and supporter_u):
            score_f += 5

        # -------  C.  Conflict heuristics ------------------------------------
        if disrp[u] or disrp[v]:                    score_d += 1
        if bully[u] or bully[v]:                    score_d += 1
        if max(stress[u], stress[v]) >= hi_stress:  score_d += 1

        # -------  D.  Decide edge type ---------------------------------------
        if score_f > score_d and score_f >= 1:
            G.add_edge(u, v, relation_type="friend")
        elif score_d >= 3 and score_d >= score_f:
            G.add_edge(u, v, relation_type="disrespect")

    return G

# Return social graph + spring-layout positions.
def build_graph(df_sample: pd.DataFrame, seed: int, *, use_synthetic=False):
    G = build_social_graph(df_sample, use_synthetic=use_synthetic)
    pos = nx.spring_layout(G, seed=seed)
    return G, pos

# ===== 3. Per-engine default runners ===========================
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

# ===== 4. Master helper for Compare page =======================
def run_comparison(
    full_df: pd.DataFrame,
    model_ids: list[str],
    frac: float,
    max_n: int,
    seed: int,
):
    """
    Return sample_df, G_base, pos, results_dict, errors_dict
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
    cmap = dict(zip(df_alloc["Student_ID"], df_alloc["Classroom"]))
    f_in = d_in = 0
    for u, v, d in G.edges(data=True):
        if cmap.get(u) != cmap.get(v):
            continue
        if d["relation_type"] == "friend":
            f_in += 1
        elif d["relation_type"] == "disrespect":
            d_in += 1
    return f_in, d_in