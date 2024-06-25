#this piece of code contains functions to generate the state space of a given ECI data set
import pandas as pd
from scipy.sparse import csr_matrix



class Commodity: 
  """
  Class to represent a commodity code.
  """

  def __init__(self, code):
    self.code = code # Commodity code (e.g., HS92 code) 
  
  # (Optional) Add methods for additional attributes or functionality


class CommodityCodePair:
  """
  class to represent a pair of commodity codes.

  """
  def __init__(self, code1, code2):
    self.code1 = code1
    self.code2 = code2
  
  def __eq__(self, other):
    # Define equality check for code pairs
    return self.code1 == other.code1 and self.code2 == other.code2
  
  def __hash__(self):
    """
    hash function for code pairs (used in dictionaries)
    this function returns a hash value for the code pair
    whi
    """
    # Define hash function for code pairs (used in dictionaries)
    return hash((self.code1, self.code2))

class State:
  def __init__(self, code_pair):
    self.code_pair = code_pair




class SparseEconomicMatrix:
  """
  Class to represent a sparse economic matrix and generate the state space.
  """

   
  def __init__(self, data_path):
    """
    Initialize the SparseEconomicMatrix object with the path to the data file.
    """
    self.data_path = data_path
    self.commodity_codes = set()  # Set to store unique commodity codes
    self.state_map = {}  # Dictionary to map code pairs to states
    self.sparse_matrix = None  # Sparse matrix to store proximities
  
  def generate_all_states(self):
    # Generate all unique combinations of commodity codes (replace with your preferred method)

    all_codes = list(self.commodity_codes) # List of all unique commodity codes
    state_combinations = [] # List to store all possible state combinations
    for i in range(len(all_codes)): # Loop through all codes
      for j in range(i + 1, len(all_codes)): # Loop through remaining codes
        state_combinations.append(CommodityCodePair(all_codes[i], all_codes[j]))
    return state_combinations

  def load_data(self, threshold=0.0):  # Add threshold argument
    # ... (Code for loading data from CSV)
    """
    Loads data from a CSV file into a sparse matrix.

    Args:
        data_path (str): Path to the CSV file containing commodity code data.
        use_dates (bool, optional): Whether to process the date column (default: False).

    Returns:
        tuple: A tuple containing the sparse matrix, a list of commodity codes 1,
                a list of commodity codes 2, and optionally a list of processed dates (if used).
    """

    # Read CSV data into a Pandas DataFrame
    df = pd.read_csv(data_path)

    # Extract relevant columns
    commodity_code_1 = df["commoditycode_1"].tolist()
    commodity_code_2 = df["commoditycode_2"].tolist()
    proximity = df["proximity"].tolist()

    # Create sparse matrix (consider using csr_matrix format for row-wise summation)
    self.sparse_matrix = csr_matrix((proximity, (commodity_code_1, code_2_indices)))
    
    # Initialize an empty dictionary to store transition probabilities
    self.transition_matrix = {}
    
    # Iterate through each row (source state) in the sparse matrix
    for i, row in enumerate(self.sparse_matrix):
      source_code1 = commodity_code_1[i]
      source_code2 = commodity_code_2[i]
      source_state_pair = CommodityCodePair(source_code1, source_code2)
      
      # Extract non-zero elements (destination states and proximities)
      destinations, proximity_values = row.nonzero()
      
      # Filter proximities above the threshold
      filtered_indices = proximity_values > threshold
      filtered_destinations = destinations[filtered_indices]
      filtered_proximities = proximity_values[filtered_indices]

         # Normalize proximity values to sum to 1 (transition probabilities)
      total_proximity = proximity_values.sum()
      normalized_proximities = proximity_values / total_proximity  # Element-wise division
      
      # Create a dictionary entry for the source state with destination states and probabilities
      self.transition_matrix[source_state_pair] = dict(zip(
          [CommodityCodePair(code1, code2) for code1, code2 in zip(commodity_code_1[destinations], commodity_code_2[destinations])],
          normalized_proximities))
      
      # Update state map (unchanged from previous example)
      self.state_map[source_state_pair] = State(source_state_pair)
      self.commodity_codes.add(source_code1)
      self.commodity_codes.add(source_code2)


# Example usage
data_path = '../data/hs92_proximities.csv'
threshold = 0.2  # Consider proximities above 0.2 only
sparse_matrix_obj = SparseEconomicMatrix(data_path)
sparse_matrix_obj.load_data(threshold)

# The transition matrix will only reflect transitions with proximities exceeding the threshold