import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from itertools import combinations
import numpy as np

def parse_csv_data(filename):
    """
    Parses CSV data containing commodity codes and proximities.

    Args:
        filename: Path to the CSV file.

    Returns:
        List of entries (tuples of commodity code pairs and proximity).
    """
    entries = []
    with open(filename, 'r') as file:
        for line in file.readlines()[1:]:  # Skip header
            code1, code2, proximity = line.strip().split(',')
            entries.append((code1, code2, float(proximity)))
    print(entries)
    return entries

def generate_states(entries, n, memo):
    """
    Generates all possible states with up to n elements.

    Args:
        entries: List containing elements and conditional probabilities.
        n: Maximum number of elements in a state.
        memo: Dictionary for memoization (empty initially).

    Returns:
        List of all possible states.
    """
    if n == 0:
        return []  # Empty state

    states = []
    for i in range(len(entries)): # Iterate over all elements
        # Check if single element state or sub-state already processed
        state_key = tuple([entries[i]] if n == 1 else entries[:i] + entries[i+1:])  # Note: Use tuple for dict key
        
        # Generate sub-states if not already processed 
        if state_key not in memo:
            memo[state_key] = generate_states(entries[:i] + entries[i+1:], n-1, memo.copy())
            states.extend(memo[state_key])
        else:
        # Only append if state hasn't been added already
            if state_key not in states:
                states.append(memo[state_key])  # Note: Append the state itself
        
    # Print memo and states for demonstration purposes (can be removed)
    print('memo', memo)
    print('states', states)

    return states

def calculate_transition_probability(state1, state2, entries):
  """
  Calculates transition probability between two states.

  Args:
      state1: List representing the first state.
      state2: List representing the second state.
      entries: List containing elements (commodity codes) and conditional probabilities (proximities).

  Returns:
      Transition probability between state1 and state2 (0 if no transition possible).
  """
  probability = 0
  for element in state2:
    found = False
    for entry in entries:
      if entry[0] in state1 and entry[1] == element:
        probability += entry[2]
        found = True
        break
    if not found:
      return 0

  return probability

def build_transition_matrix(states, entries):
  """
  Builds the transition matrix between all states.

  Args:
      states: List of all possible states.
      entries: List containing elements (commodity codes) and conditional probabilities (proximities).

  Returns:
      Transition matrix with dimensions (len(states), len(states)).
  """
  transition_matrix = [[0 for _ in range(len(states))] for _ in range(len(states))]
  for i in range(len(states)):
    for j in range(len(states)):
      if i == j:
        continue  # No transition for same state
      transition_matrix[i][j] = calculate_transition_probability(states[i], states[j], entries)

  return transition_matrix

def build_state_code_map(states):
  """
  Builds a dictionary mapping states to their constituent commodity codes.

  Args:
      states: List of all possible states.

  Returns:
      Dictionary mapping states (lists) to their corresponding commodity codes (tuples).
  """
  state_code_map = {}
  for state in states:
    state_code_map[tuple(state)] = state
  return state_code_map

# Example usage
filename = '../data/eci_test.csv'  # Replace with your CSV file path
entries = parse_csv_data(filename)
n = 4  # Maximum elements per state (adjust as needed)

all_states = generate_states(entries, n, {})
transition_matrix = build_transition_matrix(all_states, entries)
state_code_map = build_state_code_map(all_states)

print("All possible states:", all_states)
print("transition_matrix:")
for row in transition_matrix:
  print(row)

print("Example state code lookup:")
example_state = ["0101", "0102"]
if example_state in state_code_map:
  print(f"State codes for {example_state}:", state_code_map[example_state])
else:
    print(f"State '{example_state}' not found in possible states.")

# only proximites above threshold will be considered
threshold = 0.1  # Consider proximities above 0.1 only


# The transition matrix will only reflect transitions with proximities exceeding the threshold