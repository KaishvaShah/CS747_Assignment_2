import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import sys

def extract_data_game_positions(game_positions_path):
    # Define MDP data structures
    R_mdp_data = {
        'state': [],
    }

    # Open and read the MDP file
    with open(game_positions_path, 'r') as file:
        for line in file:
            tokens = line.split()
            if tokens[0] == 'state':
                continue
            R_mdp_data['state'].append(tokens[0])
        return R_mdp_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-policy", type=str, required=True, dest="value-policy")
    parser.add_argument("--opponent", type=str, required=True)

    args = parser.parse_args()
    # Specify the path to your MDP file
    game_positions_path = args.opponent

    R_mdp_data = extract_data_game_positions(game_positions_path)
    all_states = []
    with open("encoder_mapping.txt", "r") as file:
        for line in file:
            all_states.append(line.split()[0])
    # all_states = all_states[:-1]
    # print(all_states)
    # Initialize a list to store the indices
    indices = []

    # Iterate through elements in all_states and find their indices in R_mdp_data['state']
    for state in R_mdp_data['state']:
        if state in all_states:
            index = all_states.index(state)
            indices.append(index)

    # print(indices)
        
    # Specify the path to your file
    # Specify the path to your file
    file_path = getattr(args, "value-policy")
    values_list = []
    # Open and read the file, splitting lines based on spaces
    try:
        with open(file_path, "r", encoding='utf-16') as file:
            for line in file:
                # Strip any leading/trailing whitespace
                line = line.strip()
                
                # Check if the line is empty or just whitespace
                if not line:
                    continue  # Skip empty lines
                
                # Split each line into a list of values using spaces as the delimiter
                values = line.split()
                
                # Convert the values to the appropriate data type (float and int in this case)
                value_function = float(values[0])
                policy = int(float(values[1]))
                
                # Create a tuple or any suitable data structure to store the values together
                value_tuple = (value_function, policy)
                
                # Append the tuple to the list
                values_list.append(value_tuple)
    except UnicodeError:
        with open(file_path, "r", encoding='utf-8') as file:
            for line in file:
                # Strip any leading/trailing whitespace
                line = line.strip()
                
                # Check if the line is empty or just whitespace
                if not line:
                    continue  # Skip empty lines
                
                # Split each line into a list of values using spaces as the delimiter
                values = line.split()
                
                # Convert the values to the appropriate data type (float and int in this case)
                value_function = float(values[0])
                policy = int(float(values[1]))
                
                # Create a tuple or any suitable data structure to store the values together
                value_tuple = (value_function, policy)
                
                # Append the tuple to the list
                values_list.append(value_tuple)        
    # Now, values_list contains the data from the file as tuples, skipping empty lines
    # For example, you can access the value function as values_list[0][0] and the policy as values_list[0][1]
    # You can iterate through the list to access all the values
    for index in indices:
        print(f"{all_states[index]} {values_list[index][1]} {values_list[index][0]}")
        # print(f"{values_list[index][1]}")