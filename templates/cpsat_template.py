# cpsat_template.py

import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import math

# ----- Data Loading -----
@st.cache_data
def load_data(uploaded_file):
	"""Load the uploaded CSV file into a DataFrame. Assume data is already cleaned."""
	df = pd.read_csv(uploaded_file)
	# Student_ID should be string
	df['Student_ID'] = df['Student_ID'].astype(str)
	return df

# ----- Fitness Computation -----
def compute_fitness(df):
	"""Compute fitness from academic, well-being, friendship, and disrespect metrics."""
	# Academic normalization
	df['acad_norm'] = df['Total_Score'] / df['Total_Score'].max()
	# Well-being normalization (0–10 scale)
	df['well_norm'] = df['life_satisfaction'] / 10.0
	# Friends normalization
	df['friends_norm'] = df['closest_friend_count'] / df['closest_friend_count'].max()
	# Disrespect normalization (1 if not disrespected, 0 if disrespected)
	df['disrespect_flag'] = df['disrespected_by_peers'].map({'Yes': 1, 'No': 0}).fillna(0)
	df['disrespect_norm'] = df['disrespect_flag']
	# Weighted sum of components
	w_acad, w_well, w_friend, w_disrespect = 0.4, 0.3, 0.15, 0.15
	df['fitness'] = (
		w_acad * df['acad_norm'] +
		w_well * df['well_norm'] +
		w_friend * df['friends_norm'] +
		w_disrespect * df['disrespect_norm']
	)
	return df

# ----- Greedy & CP-SAT Solver -----
def greedy_assign(fitness, student_ids, num_classes, capacity):
	"""Greedy assignment by filling lowest-fitness classes first."""
	class_counts = [0] * num_classes
	class_fitness = [0.0] * num_classes
	assignment = {}
	for sid, fval in sorted(zip(student_ids, fitness), key=lambda x: x[1]):
		# assign to class with lowest total fitness under capacity
		avail = [c for c in range(num_classes) if class_counts[c] < capacity]
		cls = min(avail, key=lambda c: class_fitness[c]) if avail else 0
		assignment[sid] = cls
		class_counts[cls] += 1
		class_fitness[cls] += fval
	return assignment


def solve_constraints(fitness, student_ids, num_classes, capacity):
	"""Use CP-SAT for moderate sizes, fallback to greedy for large or infeasible."""
	N = len(student_ids)
	total_vars = N * num_classes
	# CP-SAT if small enough
	if total_vars <= 200_000:
		model = cp_model.CpModel()
		x = {(i, c): model.NewBoolVar(f'x_{i}_{c}')
			for i in range(N) for c in range(num_classes)}
		# each student exactly one
		for i in range(N):
			model.Add(sum(x[i, c] for c in range(num_classes)) == 1)
		# capacity constraint
		for c in range(num_classes):
			model.Add(sum(x[i, c] for i in range(N)) <= capacity)
		# objective: maximize total fitness
		terms = []
		for i in range(N):
			f_val = int(fitness.iloc[i] * 1000)
			for c in range(num_classes):
				terms.append(f_val * x[i, c])
		model.Maximize(sum(terms))
		solver = cp_model.CpSolver()
		solver.parameters.max_time_in_seconds = 10
		solver.parameters.num_search_workers = 8
		status = solver.Solve(model)
		if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
			assignment = {}
			for i in range(N):
				for c in range(num_classes):
					if solver.Value(x[i, c]) == 1:
						assignment[student_ids[i]] = c
						break
			return assignment
	# fallback
	return greedy_assign(fitness, student_ids, num_classes, capacity)

# ----- Streamlit App -----
def main():
	st.set_page_config(page_title="AI Constraint Optimization", layout="wide")
	st.title("AI-Powered Constraint Optimization")

	st.sidebar.markdown("## Upload Data")
	uploaded = st.sidebar.file_uploader(
		"Upload integrated student dataset (CSV)", type="csv"
	)
	if not uploaded:
		st.warning("Please upload your cleaned dataset with social columns.")
		return

	try:
		df = load_data(uploaded)
	except Exception as e:
		st.error(f"Failed to load data: {e}")
		return

	N = len(df)
	capacity = st.sidebar.number_input(
		"Capacity per class", min_value=1, value=30, step=1
	)
	min_classes = math.ceil(N / capacity)
	num_classes = st.sidebar.number_input(
		"Number of classes", min_value=min_classes, value=min_classes, step=1
	)

	df = compute_fitness(df)

	st.subheader("Sample of Computed Fitness")
	st.dataframe(
		df[['Student_ID','Total_Score','life_satisfaction','closest_friend_count','disrespected_by_peers','fitness']].head(10)
	)

	if st.sidebar.button("Run Allocation"):
		with st.spinner("Optimizing allocations..."):
			assignment = solve_constraints(
				df['fitness'], df['Student_ID'].tolist(),
				num_classes, capacity
			)
		df['Assigned_Class'] = df['Student_ID'].map(assignment)
		st.success("Allocation complete!")

		st.subheader("Class Size Distribution")
		st.bar_chart(df['Assigned_Class'].value_counts().sort_index())

		st.subheader("Average Fitness per Class")
		st.bar_chart(df.groupby('Assigned_Class')['fitness'].mean())

		st.subheader("Average Total Score per Class")
		st.bar_chart(df.groupby('Assigned_Class')['Total_Score'].mean())

		st.subheader("Average Number of Friends per Class")
		st.bar_chart(df.groupby('Assigned_Class')['closest_friend_count'].mean())

		st.subheader("Disrespected % per Class")
		st.bar_chart(
			df.groupby('Assigned_Class')['disrespected_by_peers'].apply(lambda x: (x=='Yes').mean()*100)
		)

		st.subheader("Detailed Assignments")
		st.dataframe(
			df[['Student_ID','Assigned_Class','fitness','Total_Score','life_satisfaction','closest_friend_count','disrespected_by_peers']]
		)

if __name__ == "__main__":
	main()
