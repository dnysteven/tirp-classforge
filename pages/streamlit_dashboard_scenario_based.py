# streamlit_dashboard_scenario_based.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- PAGE SETUP --------------------------
st.set_page_config(page_title="ClassForge Scenario Dashboard", layout="wide")
st.title("🎯 ClassForge: Scenario-Based AI Modelling Dashboard")

# ----------------------- FILE UPLOAD ----------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Drag and drop or browse CSV file", type=["csv"])

if not uploaded_file:
    st.warning("⚠️ Please upload a CSV file to begin.")
    st.stop()

# -------------------------- LOAD DATA --------------------------
data = pd.read_csv(uploaded_file)
st.success("✅ Dataset Loaded Successfully!")

# ----------------- CHECK & FIX MISSING COLUMNS -----------------
if 'sense_of_belonging_score' not in data.columns:
    st.warning("⚠️ 'sense_of_belonging_score' missing — generating synthetic values (1.0 to 5.0).")
    np.random.seed(42)
    data['sense_of_belonging_score'] = np.random.uniform(1.0, 5.0, len(data))

# --------------------- SCENARIO LOGIC --------------------------
st.sidebar.header("🧠 Scenario Settings")
scenario_type = st.sidebar.selectbox(
    "Select Scenario Type",
    ["High Academic Imbalance", "Friendship-Based Allocation", "Conflict-Based Separation", "Mixed Performance Clusters"]
)

# Cluster selector
num_clusters = st.sidebar.slider("Select number of clusters", 2, 5, 3)

# ----------------- CLASSROOM ALLOCATION PLANNER (in sidebar) -----------------
st.sidebar.markdown("### 🏢 Classroom Allocation Planner")
class_capacity = st.sidebar.number_input("Enter classroom capacity (students per class)", min_value=1, value=100)
num_classes = st.sidebar.number_input("Enter number of classrooms available", min_value=1, value=10)

# Scenario-wise feature selection
scenario_features = {
    "High Academic Imbalance": ['Midterm_Score', 'Final_Score', 'Total_Score', 'Assignments_Avg'],
    "Friendship-Based Allocation": ['sense_of_belonging_score', 'feels_nervous_frequency', 'manbox_attitudes_score'],
    "Conflict-Based Separation": ['bullying_not_tolerated', 'feels_depressed_frequency', 'worthless_feeling_frequency'],
    "Mixed Performance Clusters": [
        'Midterm_Score', 'Participation_Score', 'feels_nervous_frequency',
        'sports_nonparticipation_shame', 'sense_of_belonging_score'
    ]
}
selected_features = scenario_features.get(scenario_type, [])

# Extract only selected features
feature_data = data[selected_features].copy()

# Convert Yes/No to 1/0
for col in feature_data.columns:
    if feature_data[col].dtype == 'object':
        feature_data[col] = feature_data[col].map({'Yes': 1, 'No': 0})

# Fill missing values
feature_data = feature_data.fillna(feature_data.mean(numeric_only=True))

# ------------------------ CLUSTERING --------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Scenario_Group'] = kmeans.fit_predict(scaled_features)

# ----------------- Scenario Labels & Meanings -----------------
scenario_explanations = {
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

# Label mapping
group_labels = {i: f"Group {chr(65+i)}" for i in range(num_clusters)}
group_meanings = scenario_explanations[scenario_type]
data['Group_Name'] = data['Scenario_Group'].map(group_labels)

# ---------------------- SUMMARY TABLE --------------------------
st.markdown(f"### 📌 {scenario_type} - Scenario Group Summary")
group_counts = data['Scenario_Group'].value_counts().sort_index()
summary_table = pd.DataFrame({
    "Group Name": [group_labels[i] for i in range(num_clusters)],
    "Number of Students": [group_counts.get(i, 0) for i in range(num_clusters)],
    "Meaning": [group_meanings.get(i, "(no description)") for i in range(num_clusters)]
})
st.dataframe(summary_table)

# ------------------ HEATMAP ----------------------
st.markdown("### 📊 Feature Averages by Scenario Group")
try:
    heatmap_data = data.groupby("Group_Name")[selected_features].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
except:
    st.warning("⚠️ Heatmap could not be displayed due to incompatible data.")

# -------------------- TREND COMPARISON ---------------------
st.markdown("### 📈 Feature Trend Comparison")
features_to_compare = st.multiselect("Select Features to Compare", options=selected_features)
if features_to_compare:
    for feature in features_to_compare:
        avg_data = data.groupby("Group_Name")[[feature]].mean().reset_index()
        avg_data["Meaning"] = avg_data["Group_Name"].map({group_labels[i]: group_meanings[i] for i in range(num_clusters)})
        fig = px.bar(avg_data, x="Group_Name", y=feature, color="Group_Name",
                     title=f"Average {feature} by Scenario Group", hover_data=["Meaning"],
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- SCENARIO DESCRIPTIONS ------------------
st.markdown("### 📜 Scenario Group Descriptions")
for idx in range(num_clusters):
    label = chr(65 + idx)
    desc = group_meanings.get(idx, f"Group {label}: No description defined")
    st.markdown(f"- **{desc}**")

# -------------------- CLASSROOM DISTRIBUTION ------------------
st.markdown("### 🏫 Classroom Distribution")
total_students = len(data)
required_classrooms = math.ceil(total_students / class_capacity)

# Mix all students to break group bubbles
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_data['Classroom'] = (shuffled_data.index // class_capacity) + 1

# Show allocation
st.write(f"Total Students: {total_students}")
st.write(f"Required Classrooms: {required_classrooms}")
st.write(f"Available Classrooms: {num_classes}")
if required_classrooms > num_classes:
    st.error(f"❗ {required_classrooms - num_classes} more classroom(s) needed!")
else:
    st.success("✅ Classrooms are sufficient.")

# Show classroom-wise student IDs
if 'Student_ID' in shuffled_data.columns:
    st.subheader("Classroom-wise Student Allocation")
    for room in range(1, required_classrooms + 1):
        class_df = shuffled_data[shuffled_data['Classroom'] == room]
        st.markdown(f"**Classroom {room}** ({len(class_df)} students)")
        st.dataframe(class_df[['Student_ID', 'Group_Name']])
else:
    st.warning("⚠️ 'Student_ID' column missing. Cannot show student-wise allocation.")

# -------------------- EXPORT ---------------------
csv = shuffled_data.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered + Classroom Data", csv, "Final_Classroom_Assignment.csv", "text/csv")

st.markdown("---")
st.caption("💡 ClassForge - Scenario-Based Allocation Tool | AI-Powered 2025")
