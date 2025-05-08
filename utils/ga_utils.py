# utils/ga_utils.py
# ------------------------------------------------------------------
# Responsible for loading, scaling data and building social graph  │
# +  Defines the DEAP‑based Genetic Algorithm for classroom alloc. │
# ------------------------------------------------------------------

# ╭──────────────────── data_prep original imports ──────────────────╮
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import networkx as nx
import itertools, random
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────── ga_engine original imports ──────────────────╮
import numpy as np
from collections import defaultdict
from deap import base, creator, tools, algorithms
# ╰──────────────────────────────────────────────────────────────────╯


# ╭───────────────────────── DATA PREP CODE ─────────────────────────╮
def load_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Load a DataFrame, select & scale features.
    Returns scaled DataFrame and list of feature columns.
    Adapted from original file‑based version to accept a DataFrame.
    """
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

    # Copy for transformation
    df_scaled = df.copy()

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
# ╰──────────────────────────────────────────────────────────────────╯


# ╭────────────────────── GENETIC ALGORITHM CODE ────────────────────╮
def setup_deap(df, graph, num_classes: int, weights: tuple):
    """
    Configure DEAP toolbox and fitness for GA.

    df           : DataFrame from load_and_scale()
    graph        : NetworkX graph from simulate_graph()
    num_classes  : number of classrooms (K)
    weights      : (w_acad, w_friend, w_neg)
    """
    w_acad, w_friend, w_neg = weights

    # Create DEAP Fitness (maximize where positive, minimize where negative)
    creator.create(
        'FitnessMulti',
        base.Fitness,
        weights=(-w_acad, w_friend, -w_neg)
    )
    creator.create('Individual', list, fitness=creator.FitnessMulti)

    student_ids = df['Student_ID'].tolist()
    acad_scores = dict(zip(student_ids, df['Academic_Composite']))

    toolbox = base.Toolbox()
    toolbox.register('attr_class', np.random.randint, 0, num_classes)
    toolbox.register(
        'individual',
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_class,
        n=len(student_ids)
    )
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # Map student to class
        assignment = {student_ids[i]: individual[i] for i in range(len(student_ids))}

        # 1) Academic variance
        classes = defaultdict(list)
        for sid, cls in assignment.items():
            classes[cls].append(acad_scores[sid])
        means = [np.mean(v) for v in classes.values()]
        acad_var = np.var(means)

        # 2) Friend retention
        friend_count = sum(
            1 for u, v, d in graph.edges(data=True)
            if d['relation_type']=='friend' and assignment[u]==assignment[v]
        )

        # 3) Disrespect penalty
        neg_count = sum(
            1 for u, v, d in graph.edges(data=True)
            if d['relation_type']=='disrespect' and assignment[u]==assignment[v]
        )

        return acad_var, friend_count, neg_count

    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxUniform, indpb=0.2)
    toolbox.register(
        'mutate',
        tools.mutUniformInt,
        low=0,
        up=num_classes-1,
        indpb=0.1
    )
    toolbox.register('select', tools.selNSGA2)
    return toolbox


def run_ga(toolbox, pop_size: int = 100, gens: int = 50, cxpb: float = 0.7, mutpb: float = 0.2):
    """
    Execute the GA and return the best individual + evolution log
    """
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('best', np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=pop_size, lambda_=pop_size*2,
        cxpb=cxpb, mutpb=mutpb,
        ngen=gens,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    return hof[0], log
# ╰──────────────────────────────────────────────────────────────────╯
