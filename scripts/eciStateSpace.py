import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from itertools import combinations
import numpy as np

def load_data_to_sparse_matrix(data_path, threshold):
    """
    Loads data from a CSV file and creates the complete state space and transition matrix
    with normalized probabilities, considering a threshold for proximity measures.

    Args:
        data_path (str): Path to the CSV file containing commodity code data.
        threshold (float, optional): Minimum proximity value to consider (default: 0).

    Returns:
        tuple: A tuple containing the sparse transition matrix, a dictionary mapping
                state tuples (commodity code combinations) to their indices in
                the sparse matrix, and a list of unique commodity codes.
    """

    # Read CSV data into a Pandas DataFrame
    df = pd.read_csv(data_path)

    # Extract relevant columns
    commodity_code_1 = df["commoditycode_1"].tolist()
    commodity_code_2 = df["commoditycode_2"].tolist()
    proximity = df["proximity"].tolist()

    # Create a dictionary to map unique commodity codes to their indices
    code_index_map = {}
    current_index = 0

    for code in set(commodity_code_1 + commodity_code_2):
        if code not in code_index_map:
            code_index_map[code] = current_index
            current_index += 1

    # Generate all possible state combinations (tuples)
    unique_codes = list(code_index_map.keys())
    all_state_combos = [combo for combo in combinations(unique_codes, 2)]
  
    # Create a dictionary to map state tuples to their indices in the transition matrix (if needed)
    # Check if there are any unique state combinations after filtering
    filtered_combos = [combo for combo in all_state_combos if any(proximity[i] > threshold for i in range(len(commodity_code_1)) 
                                if (combo[0] == commodity_code_1[i] and combo[1] == commodity_code_2[i]))]
    
    rows,cols = zip(*filtered_combos)
    values = np.array([proximity[i] for i in range(len(commodity_code_1)) if (commodity_code_1[i], commodity_code_2[i]) in filtered_combos])

    if filtered_combos:
        state_map = {combo: i for i, combo in enumerate(filtered_combos)}
        #print(type((len(filtered_combos), len(filtered_combos))))
        # original version which didn't work because of shape errors and zeros? 
        # transition_matrix = csr_matrix((np.zeros(shape=(len(filtered_combos), len(filtered_combos))), ([], [])))
        # transition_matrix = csr_matrix((values,(rows,cols)), shape= (max(rows)+1,max(cols)+1), dtype = np.float32)                                 
        # transition_matrix = lil_matrix((values,(rows,cols)), shape= (max(rows)+1,max(cols)+1), dtype = np.float32) 
        # Iterate through each data point
        for i in range(len(commodity_code_1)):
            code1 = commodity_code_1[i]
            code2 = commodity_code_2[i]
            proximity_value = proximity[i]

            # Consider only proximity values above the threshold
            if proximity_value > threshold:
                source_state = (code1, code2)

                # Update the transition matrix based on the source state
                source_index = state_map.get(source_state)
                if source_index is not None:
                    for j in range(len(commodity_code_1)):
                        dest_code1 = commodity_code_1[j]
                        dest_code2 = commodity_code_2[j]
                        dest_state = (dest_code1, dest_code2)
                        dest_index = state_map.get(dest_state)
                        if dest_index is not None:
                            transition_matrix[source_index, dest_index] += proximity_value
        if transition_matrix is not None:
            # Normalize transition matrix (optional, consider row-wise normalization)
            transition_matrix = normalize_sparse_matrix(transition_matrix)
            #print(transition_matrix)

    # Return results
    return transition_matrix, state_map, unique_codes

def normalize_sparse_matrix(sparse_matrix):
  """
  Normalizes a sparse matrix row-wise (each row sums to 1).

  Args:
      sparse_matrix (scipy.sparse.csr_matrix): The sparse matrix to normalize.

  Returns:
      scipy.sparse.csr_matrix: The normalized sparse matrix.
  """

  # Extract row and column sums (consider using csr_matrix methods)
  row_sums = np.array(sparse_matrix.sum(axis=1))

  # Normalize by dividing each row by its sum (avoid division by zero)
  row_sums[row_sums == 0] = 1  # Handle rows with zero sum
  normalized_matrix = sparse_matrix / row_sums[:, np.newaxis]

  return normalized_matrix


# Example usage
data_path = '../data/eci_test.csv'

# only proximites above threshold will be considered
threshold = 0.1  # Consider proximities above 0.1 only

# data_path = '../data/hs92_proximities.csv'
transition_matrix, code_1, code_2 = load_data_to_sparse_matrix(data_path, threshold)

# dimensions of the sparse matrix
print("Sparse matrix dimensions:", transition_matrix.shape)
