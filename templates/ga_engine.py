# Defines the DEAP-based Genetic Algorithm for classroom allocation

import numpy as np
from collections import defaultdict
from deap import base, creator, tools, algorithms


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
