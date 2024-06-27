import itertools
import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
import logging
import time
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_proximities_and_commodity_codes(file_path, threshold):
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

def generate_states(commodity_codes, max_codes, save_interval=1000, save_path="states.pkl", memo={}):
    start_time = time.time()
    logger.info("Generating states...")
    if (tuple(commodity_codes), max_codes) in memo:
        return memo[(tuple(commodity_codes), max_codes)]
    
    states = []
    n = len(commodity_codes)
    try:
        for i in range(1, max_codes + 1):
            for comb in itertools.combinations(commodity_codes, i):
                states.append(comb)
                if len(states) % save_interval == 0:
                    with open(save_path, 'wb') as f:
                        pickle.dump(states, f)
                    logger.info(f"Saved {len(states)} states to {save_path}.")
    
        memo[(tuple(commodity_codes), max_codes)] = states
        elapsed_time = time.time() - start_time
        logger.info(f"Finished generating states. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error generating states: {e}")
        raise
    return states

def calculate_transition_probability(state1, state2, proximity_dict, memo):
    key = (state1, state2)
    if key in memo:
        return memo[key]
    
    if not state1 or not state2:
        return 0
    
    prob = 1
    for elem2 in state2:
        if elem2 not in state1:
            transition_prob = 0
            for elem1 in state1:
                transition_prob += proximity_dict.get((elem1, elem2), 0)
            prob *= transition_prob
    
    memo[key] = prob
    return prob

def create_transition_matrix(states, proximity_dict, max_code_additions, save_interval=1000, 
                             transition_matrix_path="transition_matrix.npz", memo_path="memo.pkl", batch_size=1000):
    start_time = time.time()
    logger.info("Creating transition matrix...")
    memo = load_memo(memo_path)  # Load existing memo if any
    n = len(states)
    transition_matrix = lil_matrix((n, n))  # Using LIL format for construction

    try:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            for i in range(batch_start, batch_end):
                state1 = states[i]
                for j in range(n):
                    state2 = states[j]

                    logger.info(f"Calculating transition probability from {state1} to {state2} in batch {n}") # Log progress
                    if i != j and len(set(state2) - set(state1)) <= max_code_additions:
                        transition_matrix[i, j] = calculate_transition_probability(state1, state2, proximity_dict, memo)
                if i % save_interval == 0:
                    #needs to be converted to CSR for to be able to save
                    csr_transition_matrix = csr_matrix(transition_matrix)
                    save_npz(transition_matrix_path, csr_transition_matrix)
                    save_memo(memo, memo_path)
                    logger.info(f"Saved transition matrix and memo at step {i}.")
        
        # Normalize each row
        for i in range(n):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_matrix[i] / row_sum
        
        elapsed_time = time.time() - start_time
        logger.info(f"Finished creating transition matrix. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error creating transition matrix: {e}")
        raise
    csr_transition_matrix = csr_matrix(transition_matrix)
    save_npz(transition_matrix_path, csr_transition_matrix)
    save_memo(memo, memo_path)
    return csr_transition_matrix, memo  # Convert to CSR format for efficient arithmetic operations

def create_state_mapping(states, save_path="state_mapping.pkl"):
    start_time = time.time()
    logger.info("Creating state mapping...")
    state_mapping = {}
    try:
        for i, state in enumerate(states):
            state_mapping[i] = state
        with open(save_path, 'wb') as f:
            pickle.dump(state_mapping, f)
        elapsed_time = time.time() - start_time
        logger.info(f"Finished creating state mapping. Time taken: {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error creating state mapping: {e}")
        raise
    return state_mapping

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

def load_states(states_path):
    logger.info(f"Loading states from {states_path}...")
    with open(states_path, 'rb') as f:
        states = pickle.load(f)
    logger.info(f"Loaded {len(states)} states.")
    return states

def load_transition_matrix(transition_matrix_path):
    logger.info(f"Loading transition matrix from {transition_matrix_path}...")
    transition_matrix = load_npz(transition_matrix_path)
    logger.info(f"Loaded transition matrix with shape {transition_matrix.shape}.")
    return transition_matrix

def load_state_mapping(state_mapping_path):
    logger.info(f"Loading state mapping from {state_mapping_path}...")
    with open(state_mapping_path, 'rb') as f:
        state_mapping = pickle.load(f)
    logger.info(f"Loaded state mapping with {len(state_mapping)} entries.")
    return state_mapping

def save_memo(memo, memo_path):
    with open(memo_path, 'wb') as f:
        pickle.dump(memo, f)
    logger.info(f"Memo saved to {memo_path}.")

def load_memo(memo_path):
    if os.path.exists(memo_path):
        with open(memo_path, 'rb') as f:
            memo = pickle.load(f)
        logger.info(f"Memo loaded from {memo_path}.")
        return memo
    else:
        return {}

# Example usage
# Pruning params to limit state space and transition matrix size 
threshold = 0.3 # minimum proximity threshold to consider - Historical Data? todo
max_codes = 390 # maximum number of commodity codes to consider in a given state - Historical Data - China
max_code_additions = 40 # maximum number of commodity codes that can be added in a transition

# File paths for saving/loading data
states_file_path = '../results/states_100.pkl'
transition_matrix_file_path = '../results/transition_matrix_100.npz'
state_mapping_file_path = '../results/state_mapping_100.pkl'
memo_path = '../results/memo_100.pkl'

# Variables for calculating transition matrix
file_path = '../data/hs92_proximities_100.csv'
proximity_dict, commodity_codes = read_proximities_and_commodity_codes(file_path, threshold)
states = generate_states(commodity_codes, max_codes, save_path=states_file_path)
transition_matrix, memo = create_transition_matrix(states, proximity_dict, max_code_additions, memo_path=memo_path)
state_mapping = create_state_mapping(states, save_path=state_mapping_file_path)

# Print the dimensions of the transition matrix to make sure it matches anticipated dimensions
logger.info(f"Dimensions of transition matrix: {transition_matrix.shape}")

# Example initial vector
initial_vector = [1 if state == ('0101', '0103', '0104') else 0 for state in states]
logger.info(f"Initial vector: {initial_vector}")




# Simulate the Markov process
final_vector = simulate_markov_process(transition_matrix, initial_vector, max_additions=3, steps=10)

# Print the final state distribution
logger.info(f"Final state distribution: {final_vector}")

# Convert final vector states to commodity codes
final_states = []
for i, prob in enumerate(final_vector):
    if prob > 0:
        final_states.append(state_mapping[i])
logger.info(f"Final states: {final_states}")