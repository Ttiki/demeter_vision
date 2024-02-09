import itertools

def create_scenarios():
    # Define temperature and rainfall classes
    CT = {'T1': (-float('inf'), 21.2), 'T2': (21.2, 22), 'T3': (22, float('inf'))}
    CR = {'R1': (-float('inf'), 1.8), 'R2': (1.8, 2.2), 'R3': (2.2, float('inf'))}

    # Compute Cartesian product of temperature and rainfall classes
    scenarios = list(itertools.product(CR.keys(), CT.keys()))

    # Create a dictionary to store the scenarios and their corresponding labels
    scenario_labels = {f"x{i + 1}": x for i, x in enumerate(scenarios)}

    return scenario_labels


