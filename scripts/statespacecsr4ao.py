import itertools
import csv
import numpy as np
from scipy.sparse import lil_matrix

import cProfile
import pstats

import scipy.sparse as sp
import logging
import time


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_proximities_and_commodity_codes(file_path, threshold):
    """
    Read proximities and commodity codes from a file. Only consider proximities above a certain threshold 
    when building the proximity dictionary to limit transitions considered.

    Args:
        file_path (str): The path to the file containing the data.
        threshold (float): The threshold value for proximity.

    Returns:
        tuple: A tuple containing the proximity dictionary and a sorted list of commodity codes.

    """

    start_time = time.time()
    logger.info("Reading proximities and commodity codes from file...")

    proximity_dict = {}
    commodity_codes = set()
    
    try:

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header of CSV file
            for row in reader:
                commoditycode_1, commoditycode_2, proximity = row
                proximity = float(proximity)
                if proximity >= threshold:
                    proximity_dict[(commoditycode_1, commoditycode_2)] = proximity
                    commodity_codes.update([commoditycode_1, commoditycode_2])
            elapsed_time = time.time() - start_time
            logger.info(f"Finished reading file. Time taken: {elapsed_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    return proximity_dict, sorted(commodity_codes)

def generate_states(commodity_codes, max_codes, memo={}):
    """
    Generate all possible states given a list of commodity codes and a maximum number of codes. 
    Maximum number of codes is used to limit the state space in line with historical data of the amount of products 
    (defined by commodity codes) that can be exported competitively by a country or region.

    Args:
        commodity_codes (list): A list of commodity codes.
        max_codes (int): The maximum number of codes allowed in a state.
        memo (dict, optional): A dictionary to store previously computed states. Defaults to {}.

    Returns:
        list: A list of all possible states.

    """
    start_time = time.time()
    logger.info("Generating states...")

    if (tuple(commodity_codes), max_codes) in memo:
        return memo[(tuple(commodity_codes), max_codes)]
    
    states = []
    n = len(commodity_codes)
    for i in range(1, min(max_codes, n) + 1):
        for comb in itertools.combinations(commodity_codes, i):
            states.append(comb)
    
    memo[(tuple(commodity_codes), max_codes)] = states

    elapsed_time = time.time() - start_time
    logger.info(f"Finished generating states. Time taken: {elapsed_time:.2f} seconds.")

    return states

def calculate_transition_probability(state1, state2, proximity_dict, memo):
    """
    Calculates the transition probability between two states based on the sum of proximities between
    elements in state1 and state2. Assumes that state2 requires all elements to be present.

    Args:
        state1 (list): The first state.
        state2 (list): The second state.
        proximity_dict (dict): A dictionary containing the proximity values between elements.
        memo (dict): A memoization dictionary to store previously calculated probabilities.

    Returns:
        float: The transition probability between state1 and state2.
    """
    key = (state1, state2)
    if key in memo:
        return memo[key]
    
    if not state1 or not state2:
        return 0
    
    prob = 0
    for elem1 in state1:
        for elem2 in state2:
            prob += proximity_dict.get((elem1, elem2), 0)
    
    memo[key] = prob
    return prob

def create_transition_matrix(states, proximity_dict,max_code_additions):
    """
    Creates a sparse transition matrix based on the given states and proximity dictionary.

    Args:
        states (list): A list of states.
        proximity_dict (dict): A dictionary representing the proximity between states.

    Returns:
        tuple: A tuple containing the sparse transition matrix and a memoization dictionary.
    """
    # timing & logging
    start_time = time.time()
    logger.info("Creating transition matrix...")
    
    # main function
    memo = {}
    n = len(states)
    
    # Create a sparse matrix to store transition probabilities 
    # Use lil_matrix for efficient row-wise construction
    transition_matrix = lil_matrix((n, n))

    try:
        # iterate through all pairs of states and calculate transition probabilities
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                # limit considered transitions to those that add at most max_code_additions
                if i != j and len(set(state2) - set(state1)) <= max_code_additions:
                    # Calculate transition probability
                    transition_matrix[i, j] = calculate_transition_probability(state1, state2, proximity_dict, memo)
        
        # Convert to CSR format for efficient arithmetic operations
        # TODO: consider wether coverting between formats is necessary or if we can use lil_matrix directly
        # as this is only done once it might be better to convert to CSR for large matrices 
        transition_matrix = transition_matrix.tocsr()  
        
        # Normalize each row
        row_sums = transition_matrix.sum(axis=1).A.ravel()
        zero_rows = row_sums == 0
        row_sums[zero_rows] = 1  # Avoid division by zero
        normalized_matrix = transition_matrix / row_sums[:, np.newaxis]

        # log time taken and exceptions
        elapsed_time = time.time() - start_time
        logger.info(f"Finished creating transition matrix. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error creating transition matrix: {e}")
        raise

    # Return the normalized matrix and memoization dictionary
    return normalized_matrix, memo

def create_state_mapping(states):
    """
    Create a mapping of states to their corresponding indices.

    Args:
        states (list): A list of states.

    Returns:
        dict: A dictionary mapping indices to states.
    """
    # timing & logging
    start_time = time.time()
    logger.info("Creating state mapping...")

    # main function
    state_mapping = {}
    try:
        for i, state in enumerate(states):
            state_mapping[i] = state
    
    # log time taken and exceptions
        elapsed_time = time.time() - start_time
        logger.info(f"Finished creating state mapping. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error creating state mapping: {e}")
        raise
    
    return state_mapping

def simulate_markov_process(transition_matrix, initial_vector, additions_constraint, steps):
    # timing & logging
    start_time = time.time()
    logger.info("Simulating Markov process...")


    current_vector = np.array(initial_vector)
    transition_matrix = transition_matrix.toarray()  # Convert sparse matrix to dense array 

    try:

        # main function
        for _ in range(steps):
            next_vector = np.dot(current_vector, transition_matrix)
            
            # Apply constraints: limit the number of products added
            additions = next_vector - current_vector
            num_additions = np.sum(additions > 0)
            
            if num_additions > additions_constraint:
                addition_indices = np.argsort(additions)[-additions_constraint:]
                constrained_next_vector = np.zeros_like(next_vector)
                constrained_next_vector[addition_indices] = next_vector[addition_indices]
                next_vector = constrained_next_vector
            
            current_vector = next_vector / np.sum(next_vector)  # Ensure normalization
            # print('vector at step', _, current_vector)

        elapsed_time = time.time() - start_time
        logger.info(f"Finished Markov process simulation. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error simulating Markov process: {e}")
        raise
    
    return current_vector

'''
# function to simulate markov process as CSR (potentially more efficient)
def simulate_markov_process(transition_matrix, initial_vector, max_additions, steps):
    start_time = time.time()
    logger.info("Simulating Markov process...")
    current_vector = sp.csr_matrix(initial_vector).T  # Convert initial vector to sparse column vector
    try:
        for step in range(steps):
            next_vector = transition_matrix.dot(current_vector)
            
            # Apply constraints: limit the number of products added
            additions = next_vector - current_vector
            addition_indices = additions.nonzero()[0]
            
            if len(addition_indices) > max_additions:
                top_additions = np.argsort(additions.data)[-max_additions:]
                constrained_next_vector = sp.lil_matrix(next_vector.shape)
                constrained_next_vector[addition_indices[top_additions]] = next_vector[addition_indices[top_additions]]
                next_vector = constrained_next_vector.tocsr()
            
            current_vector = next_vector / next_vector.sum()  # Ensure normalization
            logger.info(f"Vector at step {step}: {current_vector.toarray().flatten()}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Finished Markov process simulation. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error simulating Markov process: {e}")
        raise
    return current_vector.toarray().flatten()
    
    return current_vector

'''

# Example usage
# Pruning params to limit state space and transition matrix size 
threshold = 0.1 # minimum proximity threshold to consider - Historical Data
max_codes = 4 # maximum number of commodity codes to consider in a given state - Historical Data  
max_code_additions = 2 # the mamximum number of products that can be added over a time step - Historical Data

# Variables for calculating transition matrix
file_path = '../data/eci_test.csv'
proximity_dict, commodity_codes = read_proximities_and_commodity_codes(file_path, threshold)
states = generate_states(commodity_codes, max_codes)
transition_matrix, memo = create_transition_matrix(states, proximity_dict,max_code_additions)
state_mapping = create_state_mapping(states)

# Print the transition matrix
print('Transition Matrix:')
print(transition_matrix.toarray())  # Convert sparse matrix to dense array

# Print the state mapping
# print('\nState Mapping:')
# print(state_mapping)

# Print the dimensions of the transition matrix to make sure it matches anticipated dimensions
logger.info(f"Dimensions of transition matrix: {transition_matrix.shape}")
logger.info(f"Expected number of states: {len(commodity_codes)**2}")  # To check if the dimensions of the transition matrix match the number of possible states

# Print the state mapping
logger.info(f"State mapping: {state_mapping}")



# Simulation parameters
steps = 10 #time steps considered
additions_constraint = 3 # additions considered within developed transitions matrix  <= max_code_additions

# Example initial vector (arbitrary for demonstration)
initial_vector = [1 if state == ('0101','0103','0104') else 0 for state in states]

# Simulate the Markov process
final_vector = simulate_markov_process(transition_matrix, initial_vector, additions_constraint, steps)

# Print the final state distribution
print('\nFinal State Distribution:')
# as an array
print(final_vector)

# Converted to a list of 
for state_index, state_prob in enumerate(final_vector):
    if state_prob > 0:
        print(f'State: {state_mapping[state_index]}, Probability: {state_prob}')