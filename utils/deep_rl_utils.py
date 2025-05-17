"""
utils/deep_rl_utils.py  ·  Deep-RL (DQN) utilities
---------------------------------------------------
This module is extracted from *Classroom Allocation Dqn.ipynb* so that
`pages/Deep_RL_Allocation.py` can import it directly.

Functions
---------
load_model(state_size: int, action_size: int, ckpt_path: str = "models/deep_rl_model.pth") -> QNetwork
    Load a pretrained DQN with exactly the notebook's architecture / hyper-params.

allocate_students(df: DataFrame, model: QNetwork, *, num_classrooms=10, max_capacity=30) -> DataFrame
    Run greedy ε-greedy inference and return a DataFrame with Assigned_Classroom + Reason.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import random


# ===== 0. Fixed notebook hyper-parameters ====================================
GAMMA          = 0.99     # Discount factor
LR             = 1e-3     # Adam learning-rate
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.995
REPLAY_SIZE    = 20_000
BATCH_SIZE     = 64
HIDDEN_SIZE    = 128
SEED           = 42

# Feature list from notebook (order matters!)
_FEATURES = [
    "Total_Score",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Stress_Level (1-10)",
    "life_satisfaction",
    "closest_friend_count",
    "feels_safe_in_class",
    "feels_comfortable_at_school",
    "feels_isolated_due_to_opinion",
    "is_bullied",                 # Y/N (converted to 0/1)
    "disrespected_by_peers",      # Y/N (converted to 0/1)
]
STATE_DIM  = len(_FEATURES)        # = 15
ACTION_DIM = 10                    # Max classrooms supported by notebook


# ===== 1.  Deep-Q Network (unchanged) ========================================
class QNetwork(nn.Module):
    """Simple 3-layer fully-connected DQN."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        torch.manual_seed(SEED)
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No softmax (Q-values)


# ===== 2.  Model loader  =====================================================
def load_model(
    state_size: int = STATE_DIM,
    action_size: int = ACTION_DIM,
    ckpt_path: str | Path = "models/deep_rl_model.pth",
) -> QNetwork:
    """
    Load notebook-trained DQN.
    • If layer shapes mismatch, only compatible weights are loaded and
      mismatching layers are re-initialised (to allow k-class flexibility).
    """
    model = QNetwork(state_size, action_size)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print("[deep_rl_utils] Checkpoint not found – using random-init weights.")
        model.eval()
        return model

    chkpt = torch.load(ckpt_path, map_location="cpu")
    # Copy only matching shapes
    new_sd = model.state_dict()
    matched = {
        k: v for k, v in chkpt.items()
        if k in new_sd and v.shape == new_sd[k].shape
    }
    new_sd.update(matched)
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model


# ===== 3.  Feature pre-processing  ==========================================
def _prepare_state_matrix(df_raw: pd.DataFrame) -> torch.Tensor:
    """
    Return a [N, STATE_DIM] float32 tensor with notebook feature order.
    Non-numeric flags are converted to 0/1.
    NaNs are filled with column means.
    """
    df = df_raw.copy()

    # Map Y/N to 1/0 for bullying & disrespect flags
    for col in ("is_bullied", "disrespected_by_peers"):
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    # Ensure required columns exist
    missing = [c for c in _FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Deep-RL allocator – missing columns: {missing}")

    # Fill NaN with column mean
    df[_FEATURES] = df[_FEATURES].fillna(df[_FEATURES].mean())

    state = torch.tensor(df[_FEATURES].values, dtype=torch.float32)
    return state


# ===== 4.  Allocation function  =============================================
def allocate_students(
    df_raw: pd.DataFrame,
    model: QNetwork,
    *,
    num_classrooms: int = ACTION_DIM,
    max_capacity: int = 30,
    epsilon: float = EPSILON_END,        # greedy by default
) -> pd.DataFrame:
    """
    Greedy ε-greedy allocation (no replay – inference only).

    Returns df with columns:
        Student_ID | Assigned_Classroom | Reason (why that class)
    """
    if num_classrooms > ACTION_DIM:
        raise ValueError(
            f"Deep-RL model supports up to {ACTION_DIM} classrooms (got {num_classrooms})."
        )

    df = df_raw.copy()
    states = _prepare_state_matrix(df)        # [N, 15]
    with torch.no_grad():
        q_values = model(states)              # [N, ACTION_DIM]

    # Track per-class counts
    counts = {c: 0 for c in range(num_classrooms)}

    assigned = []
    for idx, sid in enumerate(df["Student_ID"]):
        # ε-greedy: random or greedy action within capacity
        if random.random() < epsilon:
            legal = [c for c in range(num_classrooms) if counts[c] < max_capacity]
            if legal:
                action = random.choice(legal)
            else:
                action = min(counts, key=counts.get)
        else:
            qs = q_values[idx, :num_classrooms].clone().numpy()
            order = np.argsort(qs)[::-1]  # descending
            for a in order:
                if counts[a] < max_capacity:
                    action = int(a)
                    break
            else:
                action = min(counts, key=counts.get)

        counts[action] += 1

        # Build student-based reason explanation
        student_row = df_raw.loc[df_raw["Student_ID"] == sid].squeeze()
        reasons = []
        if student_row.get("closest_friend_count", 0) >= 3:
            reasons.append("has close friends")
        if student_row.get("is_bullied", "No") == "Yes":
            reasons.append("is bullied")
        if student_row.get("disrespected_by_peers", "No") == "Yes":
            reasons.append("faced disrespect")
        if student_row.get("Total_Score", 0) >= 85:
            reasons.append("high performer")
        if student_row.get("Stress_Level (1-10)", 0) >= 8:
            reasons.append("high stress")
        if student_row.get("life_satisfaction", 10) <= 3:
            reasons.append("low satisfaction")

        explanation = ", ".join(reasons) if reasons else "balanced profile"
        assigned.append((sid, action + 1, explanation))   # 1-indexed

    out = pd.DataFrame(assigned, columns=["Student_ID", "Assigned_Classroom", "Reason"])
    return out


# ===== 5.  __all__ ===========================================================
__all__ = [
    "QNetwork",
    "load_model",
    "allocate_students",
    "_FEATURES",
]
