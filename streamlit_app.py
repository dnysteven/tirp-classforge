# Streamlit App for DQN-Based Classroom Allocation

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# ==== Model Setup ====
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

@st.cache_resource
def load_model(path, state_size, action_size):
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ==== Allocation Logic ====
def allocate_students(df, model, num_classrooms=10, max_capacity=20):
    df = df.copy()
    binary_columns = ['has_close_friends', 'is_bullied', 'disrespected_by_peers', 'participates_in_activities']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df.fillna(0, inplace=True)

    required_features = [
        'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
        'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
        'Stress_Level (1-10)', 'Sleep_Hours_per_Night', 'life_satisfaction',
        'has_close_friends', 'is_bullied', 'disrespected_by_peers', 'participates_in_activities'
    ]

    included_columns = ['Student_ID'] + [col for col in required_features if col in df.columns]
    if 'Student_Name' in df.columns:
        included_columns.insert(1, 'Student_Name')
    df = df[included_columns]

    classrooms = {i: [] for i in range(num_classrooms)}
    student_ids = df['Student_ID'].tolist()
    student_names = df['Student_Name'].tolist() if 'Student_Name' in df.columns else [''] * len(df)
    results = []

    for i in range(len(df)):
        try:
            student_row = df.iloc[i]
            features = student_row.drop(['Student_ID', 'Student_Name'], errors='ignore').values.astype(np.float32)
            state = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state).squeeze()
                for j in range(num_classrooms):
                    if len(classrooms[j]) >= max_capacity:
                        q_values[j] = -float('inf')
                action = torch.argmax(q_values).item()

            classrooms[action].append(student_ids[i])

            reasons = []
            if student_row['has_close_friends'] == 1:
                reasons.append("has close friends")
            if student_row['is_bullied'] == 1:
                reasons.append("is bullied")
            if student_row['disrespected_by_peers'] == 1:
                reasons.append("faced disrespect")
            if student_row['Total_Score'] >= 85:
                reasons.append("high performer")
            if student_row['Stress_Level (1-10)'] >= 8:
                reasons.append("high stress")
            if student_row['life_satisfaction'] <= 3:
                reasons.append("low satisfaction")

            explanation = ", ".join(reasons) if reasons else "balanced profile"

            results.append({
                "Student_ID": student_ids[i],
                "Student_Name": student_names[i],
                "Assigned_Classroom": action,
                "Reason": explanation,
                "Performance_Group": (
                    "High" if student_row['Total_Score'] >= 85 else
                    "Mid" if student_row['Total_Score'] >= 70 else
                    "Low"
                )
            })
        except Exception:
            results.append({
                "Student_ID": student_ids[i],
                "Student_Name": student_names[i],
                "Assigned_Classroom": "Error",
                "Reason": "N/A",
                "Performance_Group": "N/A"
            })

    return pd.DataFrame(results), classrooms

# ==== Streamlit UI ====
st.set_page_config(page_title="Classroom Allocation", layout="centered")
st.title("AI-Powered Classroom Allocation")
st.write("Upload a CSV file with student data to generate classroom assignments using a trained DQN model.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)

    if 'Student_ID' not in df_input.columns:
        st.error("Missing 'Student_ID' column.")
    else:
        model = load_model("classroom_allocation_dqn_model_final.pth", state_size=15, action_size=10)
        st.success("Model loaded. Processing students...")

        assigned_df, classrooms = allocate_students(df_input, model)

        # Sidebar filters with default = all selected
        with st.sidebar:
            st.header("Filters")
            classroom_options = sorted(assigned_df['Assigned_Classroom'].unique())
            performance_options = sorted(assigned_df['Performance_Group'].unique())

            selected_classrooms = st.selectbox("Select Classroom:", options=["All"] + classroom_options, index=0)
            selected_performance = st.selectbox("Select Performance Group:", options=["All"] + performance_options, index=0)

        filtered_df = assigned_df.copy()
        if selected_classrooms != "All":
            filtered_df = filtered_df[filtered_df['Assigned_Classroom'] == selected_classrooms]
        if selected_performance != "All":
            filtered_df = filtered_df[filtered_df['Performance_Group'] == selected_performance]

        # Display full filtered data
        st.subheader("📋 Allocation Results")
        st.dataframe(filtered_df)

        # Grouped tables per classroom
        st.subheader("🧑‍🏫 Allocation by Classroom")
        for class_id in sorted(filtered_df["Assigned_Classroom"].unique()):
            st.markdown(f"#### Classroom {class_id}")
            st.dataframe(filtered_df[filtered_df["Assigned_Classroom"] == class_id])

        # Merged output with original features for export
        if 'Total_Score' in df_input.columns:
            merged_df = pd.merge(
                assigned_df,
                df_input,
                on=['Student_ID', 'Student_Name'],
                how='left',
                suffixes=('', '_input')
            )
        else:
            merged_df = assigned_df

        csv = merged_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Assignments as CSV",
            data=csv,
            file_name='classroom_assignments.csv',
            mime='text/csv'
        )

        # ==== Visual Summary Statistics ====
        if 'Total_Score' in merged_df.columns and 'Stress_Level (1-10)' in merged_df.columns and 'life_satisfaction' in merged_df.columns:
            st.subheader("📊 Average Metrics by Classroom")
            summary_stats = merged_df.groupby("Assigned_Classroom").agg({
                "Total_Score": "mean",
                "Stress_Level (1-10)": "mean",
                "life_satisfaction": "mean"
            }).round(2).rename(columns={
                "Total_Score": "Avg Total Score",
                "Stress_Level (1-10)": "Avg Stress",
                "life_satisfaction": "Avg Satisfaction"
            })
            st.dataframe(summary_stats)
