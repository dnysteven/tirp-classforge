# ==============================
# scenario_model.py
# Scenario-Based Grouping & Dataset Preparation
# ==============================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load Cleaned Dataset
dataset_path = 'Cleaned_Expanded_Students_Grading_Dataset.csv'
data = pd.read_csv(dataset_path)
print("✅ Dataset Loaded Successfully!")

# Step 2: Feature Selection (Academic, Behavior, Social Factors)
selected_features = [
    'wants_more_peer_interaction', 'bullying_not_tolerated', 'feels_nervous_frequency',
    'feels_depressed_frequency', 'effort_exhaustion_level', 'worthless_feeling_frequency',
    'fixed_intelligence_belief', 'covid_impact_concern', 'manbox_attitudes_score',
    'sports_nonparticipation_shame', 'gender_bias_in_stem', 'sense_of_belonging_score',
    'perceived_teacher_support', 'academic_self_efficacy'
]

feature_data = data[selected_features].copy()

# Step 3: Encode Yes/No
for col in feature_data.columns:
    if feature_data[col].dtype == 'object':
        feature_data[col] = feature_data[col].map({'Yes': 1, 'No': 0})

# Step 4: Handle Missing
feature_data = feature_data.fillna(feature_data.mean(numeric_only=True))
print("✅ Missing values handled.")

# Step 5: Scaling
scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)
print("✅ Data scaled.")

# Step 6: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Scenario_Group'] = kmeans.fit_predict(feature_data_scaled)
print("✅ Scenario groups assigned.")

# Step 7: Export
data.to_csv('Scenario_Assigned_Students.csv', index=False)
print("✅ File saved as 'Scenario_Assigned_Students.csv'")
