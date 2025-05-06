"""
Scenario‑Based AI model utilities
⚠️ All comments below come from the original prototype file
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

# ----------------- CHECK & FIX MISSING COLUMNS -----------------
def ensure_belonging_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'sense_of_belonging_score' missing — generate synthetic values (1.0 to 5.0)
    """
    if 'sense_of_belonging_score' not in df.columns:
        np.random.seed(42)
        df = df.copy()
        df['sense_of_belonging_score'] = np.random.uniform(1.0, 5.0, len(df))
    return df


# Scenario-wise feature selection
SCENARIO_FEATURES = {
    "High Academic Imbalance": ['Midterm_Score', 'Final_Score', 'Total_Score', 'Assignments_Avg'],
    "Friendship-Based Allocation": ['sense_of_belonging_score', 'feels_nervous_frequency', 'manbox_attitudes_score'],
    "Conflict-Based Separation": ['bullying_not_tolerated', 'feels_depressed_frequency', 'worthless_feeling_frequency'],
    "Mixed Performance Clusters": [
        'Midterm_Score', 'Participation_Score', 'feels_nervous_frequency',
        'sports_nonparticipation_shame', 'sense_of_belonging_score'
    ]
}

SCENARIO_EXPLANATIONS = {
    "High Academic Imbalance": {
        0: "🧠 Group A: High performers grouped together",
        1: "📘 Group B: Struggling students needing support",
        2: "📊 Group C: Moderate performers",
        3: "🔄 Group D: Mixed academic abilities",
        4: "🧪 Group E: Varying score consistency"
    },
    "Friendship-Based Allocation": {
        0: "👭 Group A: Strong friendship ties",
        1: "👥 Group B: Socially isolated",
        2: "🤝 Group C: Peer influencers",
        3: "🎯 Group D: Neutral groupings",
        4: "🫂 Group E: Mixed connections"
    },
    "Conflict-Based Separation": {
        0: "❌ Group A: High behavioral conflict",
        1: "🧘 Group B: Calm/disciplined group",
        2: "⚖️ Group C: Balanced behavior",
        3: "🔔 Group D: Potential disruptors",
        4: "🧩 Group E: Mixed risk students"
    },
    "Mixed Performance Clusters": {
        0: "💬 Group A: Well-rounded students",
        1: "🎓 Group B: Academically strong, socially moderate",
        2: "📉 Group C: Low behavior score but high academics",
        3: "📚 Group D: Balanced across metrics",
        4: "💡 Group E: Socially strong, academically average"
    }
}


def run_scenario_clustering(df: pd.DataFrame, scenario_type: str, n_clusters: int):
    """
    Executes the feature preparation + k‑means clustering for the chosen scenario.
    Adds columns: 'Scenario_Group', 'Group_Name'
    Returns the modified dataframe and the explanation dict for groups.
    """
    df = ensure_belonging_score(df)

    selected_features = SCENARIO_FEATURES.get(scenario_type, [])
    feature_data = df[selected_features].copy()

    # Convert Yes/No strings to 1/0
    for col in feature_data.columns:
        if feature_data[col].dtype == 'object':
            feature_data[col] = feature_data[col].map({'Yes': 1, 'No': 0})

    feature_data = feature_data.fillna(feature_data.mean(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df = df.copy()
    df['Scenario_Group'] = kmeans.fit_predict(scaled)

    # Map group numbers to letters
    group_labels = {i: f"Group {chr(65 + i)}" for i in range(n_clusters)}
    df['Group_Name'] = df['Scenario_Group'].map(group_labels)

    return df, group_labels, SCENARIO_EXPLANATIONS[scenario_type]


def allocate_to_classrooms(df: pd.DataFrame, capacity: int):
    """
    Simple sequential allocation: shuffles students then chunks by capacity.
    Returns the dataframe with a new 'Classroom' column and the #classrooms needed.
    """
    total = len(df)
    required_classrooms = math.ceil(total / capacity)

    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled['Classroom'] = (shuffled.index // capacity) + 1
    return shuffled, required_classrooms