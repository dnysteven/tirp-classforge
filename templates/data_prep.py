# Responsible for loading, scaling data and building social graph

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import networkx as nx
import itertools, random


def load_and_scale(path: str = r"C:\Users\ghufr\OneDrive\Desktop\Classforge-Project\data\Cleaned_Expanded_Students_Grading_Dataset.csv"):
    """
    Load CSV into DataFrame, select & scale features.
    Returns scaled DataFrame and list of feature columns.
    """
    df = pd.read_csv(path)

    
    df_1000 = df.sample(n=1000, random_state=42)
    subset_path = r"C:\Users\ghufr\OneDrive\Desktop\Classforge-Project\data/Students_1000.csv"
    df_1000.to_csv(subset_path, index=False)
    
    
    # 1) Academic metrics
    academic_cols = [
        'Midterm_Score', 'Final_Score', 'Assignments_Avg',
        'Quizzes_Avg', 'Participation_Score', 'Projects_Score'
    ]

    # 2) Wellbeing metrics
    wellbeing_cols = [
        'Stress_Level (1-10)', 'Sleep_Hours_per_Night',
        'life_satisfaction', 'feels_nervous_frequency',
        'feels_depressed_frequency', 'effort_exhaustion_level'
    ]

    # 3) Socioeconomic metrics (categorical)
    socioeconomic_cols = [
        'Parent_Education_Level', 'Family_Income_Level',
        'Internet_Access_at_Home'
    ]

    feature_cols = academic_cols + wellbeing_cols + socioeconomic_cols

    # Make a copy for transformation


    df_scaled = df_1000.copy()

    # Encode categorical socioeconomic columns
    for col in socioeconomic_cols:
        if df_scaled[col].dtype == 'object':
            df_scaled[col] = LabelEncoder().fit_transform(df_scaled[col])

    # Scale all feature columns to [0,1]
    scaler = MinMaxScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    # Composite academic score
    df_scaled['Academic_Composite'] = df_scaled[academic_cols].sum(axis=1)

    return df_scaled, feature_cols

def simulate_graph(df, p_friend: float = 0.05, p_disrespect: float = 0.02):
    """
    Simulate a social network graph with 'friend' and 'disrespect' edges.
    """
    student_ids = df['Student_ID'].tolist()
    G = nx.Graph()
    G.add_nodes_from(student_ids)

    for u, v in itertools.combinations(student_ids, 2):
        r = random.random()
        if r < p_friend:
            G.add_edge(u, v, relation_type='friend')
        elif r < p_friend + p_disrespect:
            G.add_edge(u, v, relation_type='disrespect')

    return G
