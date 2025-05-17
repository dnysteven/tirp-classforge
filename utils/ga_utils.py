import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

def load_and_scale(df):
    df = df.copy()

    # Convert Yes/No to 1/0 for binary columns
    binary_columns = ["is_bullied", "disrespected_by_peers"]
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    scaler = MinMaxScaler()
    feature_cols = [
        "Total_Score", "life_satisfaction", "Stress_Level (1-10)",
        "is_bullied", "disrespected_by_peers", "closest_friend_count"
    ]
    scaled_values = scaler.fit_transform(df[feature_cols])
    df[feature_cols] = scaled_values
    return df, feature_cols

# UPDATED: align with GA engine’s expected edge logic
def simulate_graph(df):
    G = nx.Graph()
    ids = df["Student_ID"].tolist()
    for u in ids:
        G.add_node(u)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            u = ids[i]
            v = ids[j]
            row_u = df.iloc[i]
            row_v = df.iloc[j]

            score_f = 0
            score_d = 0

            # Friendship indicators
            if row_u.get("closest_friend_count", 0) >= 0.6 and row_v.get("closest_friend_count", 0) >= 0.6:
                score_f += 2
            if abs(row_u.get("Stress_Level (1-10)", 0) - row_v.get("Stress_Level (1-10)", 0)) <= 0.3:
                score_f += 2
            if row_u.get("life_satisfaction", 0) >= 0.6 and row_v.get("life_satisfaction", 0) >= 0.6:
                score_f += 1

            # Conflict indicators
            if row_u.get("disrespected_by_peers", 0) >= 0.5 or row_v.get("disrespected_by_peers", 0) >= 0.5:
                score_d += 1
            if row_u.get("is_bullied", 0) >= 0.5 or row_v.get("is_bullied", 0) >= 0.5:
                score_d += 1
            if max(row_u.get("Stress_Level (1-10)", 0), row_v.get("Stress_Level (1-10)", 0)) >= 0.8:
                score_d += 1

            if score_f > score_d and score_f >= 2:
                G.add_edge(u, v, relation_type="friend")
            elif score_d >= 2 and score_d >= score_f:
                G.add_edge(u, v, relation_type="disrespect")

    return G

def setup_deap(df, sim_matrix, num_classes, weights=(0.4, 0.3, 0.3)):
    n = len(df)
    student_ids = df["Student_ID"].tolist()

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    def evaluate(individual):
        try:
            groups = {i: [] for i in range(num_classes)}
            for sid, cls in zip(student_ids, individual):
                groups[cls].append(sid)
            sizes = [len(g) for g in groups.values()]
            variance = np.var(sizes)
            friends = 0
            conflict = 0
            for i in range(n):
                for j in range(i+1, n):
                    if sim_matrix.has_edge(student_ids[i], student_ids[j]):
                        rel = sim_matrix[student_ids[i]][student_ids[j]].get("relation_type")
                        same_class = individual[i] == individual[j]
                        if rel == "friend" and same_class:
                            friends += 1
                        elif rel == "disrespect" and same_class:
                            conflict += 1
            return -variance, friends, -conflict
        except Exception as e:
            print(f"[evaluate ERROR] {e}")
            return -9999, 0, 0

    toolbox = base.Toolbox()
    toolbox.register("attr_class", random.randint, 0, num_classes - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_class, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.2)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_classes - 1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def run_ga(toolbox, pop_size=100, gens=50, cxpb=0.7, mutpb=0.2):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gens, 
                                   stats=stats, halloffame=hof, verbose=False)
    best = hof[0]
    return best, log
