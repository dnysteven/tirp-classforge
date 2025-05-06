# utils/deep_rl_utils.py
"""
Deep‑RL (DQN) classroom allocator.

Loads:
    • models/deep_rl_model.pth   ← pre‑trained Q‑network
Exposes:
    • load_model()
    • allocate_students()        ← used by the Streamlit page
"""

from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

# ───────────────────────── network ────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ───────────────────────── paths & loader ─────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "deep_rl_model.pth"

_BINARY_COLS = [
    "has_close_friends",
    "is_bullied",
    "disrespected_by_peers",
    "participates_in_activities",
]

_FEATURES = [
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Total_Score",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night",
    "life_satisfaction",
    *_BINARY_COLS,
]

def load_model(state_size: int, action_size: int) -> QNetwork:
    """Load the trained DQN (CPU only)."""
    net = QNetwork(state_size, action_size)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net

# ───────────────────────── allocation core ────────────────────
def allocate_students(
    df_raw: pd.DataFrame,
    model: QNetwork,
    num_classrooms: int = 10,
    max_capacity: int = 20,
) -> pd.DataFrame:
    """Return a DataFrame with classroom assignments and explanation."""
    df = df_raw.copy()

    # binary map Yes/No → 1/0
    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    df.fillna(0, inplace=True)

    incl_cols = ["Student_ID"]
    if "Student_Name" in df.columns:
        incl_cols.append("Student_Name")
    incl_cols += [c for c in _FEATURES if c in df.columns]
    df = df[incl_cols]

    classrooms = {i: [] for i in range(num_classrooms)}
    results = []

    for _, row in df.iterrows():
        feats = (
            row.drop(["Student_ID", "Student_Name"], errors="ignore")
            .values.astype(np.float32)
        )
        state = torch.tensor(feats).unsqueeze(0)
        with torch.no_grad():
            q = model(state).squeeze()

            # mask full classes
            for c in range(num_classrooms):
                if len(classrooms[c]) >= max_capacity:
                    q[c] = -float("inf")

            choice = torch.argmax(q).item()

        classrooms[choice].append(row["Student_ID"])

        # explanation (same rules as original script)
        reasons = []
        if "has_close_friends" in row and row["has_close_friends"] == 1:
            reasons.append("has close friends")
        if "is_bullied" in row and row["is_bullied"] == 1:
            reasons.append("is bullied")
        if "disrespected_by_peers" in row and row["disrespected_by_peers"] == 1:
            reasons.append("faced disrespect")
        if "Total_Score" in row and row["Total_Score"] >= 85:
            reasons.append("high performer")
        if "Stress_Level (1-10)" in row and row["Stress_Level (1-10)"] >= 8:
            reasons.append("high stress")
        if "life_satisfaction" in row and row["life_satisfaction"] <= 3:
            reasons.append("low satisfaction")

        results.append(
            {
                "Student_ID": row["Student_ID"],
                "Student_Name": row.get("Student_Name", ""),
                "Assigned_Classroom": choice + 1,  # start at 1 for users
                "Reason": ", ".join(reasons) or "balanced profile",
                "Performance_Group": (
                    "High"
                    if row.get("Total_Score", 0) >= 85
                    else "Mid"
                    if row.get("Total_Score", 0) >= 70
                    else "Low"
                ),
            }
        )

    return pd.DataFrame(results)