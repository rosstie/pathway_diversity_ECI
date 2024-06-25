import itertools
import csv

def read_proximities_and_commodity_codes(file_path,threshold):
    proximity_dict = {}
    commodity_codes = set()
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            commoditycode_1, commoditycode_2, proximity = row
            proximity = float(proximity)
            if proximity >= threshold:
                proximity_dict[(commoditycode_1, commoditycode_2)] = proximity
                commodity_codes.update([commoditycode_1, commoditycode_2])
    return proximity_dict, sorted(commodity_codes)

def generate_states(commodity_codes,max_codes,memo={}):
    # if tuple(commodity_codes) in memo:
        # return memo[tuple(commodity_codes)]
    
    if (tuple(commodity_codes), max_codes) in memo:
        return memo[(tuple(commodity_codes), max_codes)]
    
    states = []
    n = len(commodity_codes)
    for i in range(1, n + 1):
        for comb in itertools.combinations(commodity_codes, i):
            states.append(comb)
    
    memo[tuple(commodity_codes)] = states
    return states

'''
# states generator no memoisation
def generate_states(commodity_codes):
    states = []
    n = len(commodity_codes)
    for i in range(1, n + 1):
        for comb in itertools.combinations(commodity_codes, i):
            states.append(comb)
    print('states',states)
    return states
'''

def calculate_transition_probability(state1, state2, proximity_dict, memo):
    key = (state1, state2)
    if key in memo:
        return memo[key]
    
    if not state1 or not state2:
        return 0
    
    prob = 0
    for elem1 in state1:
        for elem2 in state2:
            if elem1 != elem2:
                prob += proximity_dict.get((elem1, elem2), 0)
    
    memo[key] = prob
    return prob

def create_transition_matrix(states, proximity_dict):
    memo = {}
    n = len(states)
    transition_matrix = [[0]*n for _ in range(n)]
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if i != j:
                transition_matrix[i][j] = calculate_transition_probability(state1, state2, proximity_dict, memo)
    
    # Normalize each row
    for i in range(n):
        row_sum = sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] = [x / row_sum for x in transition_matrix[i]]
    
    
    return transition_matrix, memo

def create_state_mapping(states):
    state_mapping = {}
    for i, state in enumerate(states):
        state_mapping[i] = state
    return state_mapping

import numpy as np

def simulate_markov_process(transition_matrix, initial_vector, max_additions, steps):
    current_vector = np.array(initial_vector)
    for _ in range(steps):
        next_vector = np.dot(current_vector, transition_matrix)
        
        # Apply constraints: limit the number of products added
        additions = next_vector - current_vector
        num_additions = np.sum(additions > 0)
        
        if num_additions > max_additions:
            addition_indices = np.argsort(additions)[-max_additions:]
            constrained_next_vector = np.zeros_like(next_vector)
            constrained_next_vector[addition_indices] = next_vector[addition_indices]
            next_vector = constrained_next_vector
        
        current_vector = next_vector / np.sum(next_vector)  # Ensure normalization
        print('vector at step',_,current_vector)
    
    return current_vector


# Example usage
# Pruning params to limit state space and transition matrix size 
threshold = 0.1 # minimum proximity threshold to consider - Historical Data
max_codes = 3 # maximum number of commodity codes to consider in a given state - Historcal Data  

# Variables for calculating transition matrix
file_path = '../data/eci_test.csv'
proximity_dict, commodity_codes = read_proximities_and_commodity_codes(file_path,threshold)
states = generate_states(commodity_codes)
transition_matrix, memo = create_transition_matrix(states, proximity_dict)
state_mapping = create_state_mapping(states)

# Print the transition matrix
for row in transition_matrix:
    print(row)

# print the dimensions of the transition matrix to make sure it matches anticipated dimensions
print('dimensions of transition matrix:',len(transition_matrix),len(transition_matrix[0])) 
print(len(commodity_codes)**2) # to check if the dimensions of the transition matrix match the number of possible states

# Print the state mapping
print(state_mapping)



################# Example usage of simulate_markov_process ################# 
# Calculation params 
steps = 10
max_additions = len(commodity_codes)

# Example initial vector
initial_vector = [1 if state == ('0101','0103','0104') else 0 for state in states]
print('initial vector:',initial_vector)

# Simulate the Markov process
final_vector = simulate_markov_process(transition_matrix, initial_vector, max_additions=3, steps=10)

# Print the final state distribution
print(final_vector)

# convert final vector states to commodity codes
final_states = []
for i, prob in enumerate(final_vector):
    if prob > 0:
        final_states.append(state_mapping[i])

