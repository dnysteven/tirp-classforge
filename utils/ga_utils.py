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

def simulate_graph(df):
    G = nx.Graph()
    ids = df["Student_ID"].tolist()

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            u = ids[i]
            v = ids[j]
            add_friend = (
                df.iloc[i]["has_close_friends"] == "Yes" and
                df.iloc[j]["has_close_friends"] == "Yes"
            )
            add_disrespect = (
                df.iloc[i]["disrespected_by_peers"] == "Yes" or
                df.iloc[j]["disrespected_by_peers"] == "Yes"
            )

            if add_friend:
                G.add_edge(u, v, relation_type="friend")
            elif add_disrespect:
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
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_classes - 1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def run_ga(toolbox, pop_size=100, gens=50, cxpb=0.7, mutpb=0.2):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("best", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    try:
        pop, log = algorithms.eaMuPlusLambda(
            pop, toolbox,
            mu=pop_size, lambda_=pop_size * 2,
            cxpb=cxpb, mutpb=mutpb,
            ngen=gens,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
    except Exception as e:
        print(f"[GA ERROR] {e}")
        return None, []

    if len(hof) == 0:
        print("[⚠] Hall of Fame is empty — using best from population instead.")
        best = tools.selBest(pop, 1)[0]
    else:
        best = hof[0]

    return best, log

def build_solution_similarity_graph(pop, threshold=0.8):
    n = len(pop)
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            sim = np.mean(np.array(pop[i]) == np.array(pop[j]))
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)
    return G
