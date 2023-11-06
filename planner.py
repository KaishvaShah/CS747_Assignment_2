import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import pulp
import time

# Record the start time
start_time = time.time()

def extract_data_mdp(mdp):
    # Define MDP data structures
    mdp_data = {
        'metadata': {},
        'transitions': None,
        'rewards': None
    }

    # Specify the path to your MDP file
    mdp_file_path = mdp

    try:
        # Try to open the file with utf-16 encoding
        with open(mdp_file_path, 'rt', encoding='utf-16') as file:
            for line in file:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                keyword = tokens[0]
                if keyword == 'numStates':
                    mdp_data['metadata']['numStates'] = int(tokens[1])
                elif keyword == 'numActions':
                    mdp_data['metadata']['numActions'] = int(tokens[1])
                elif keyword == 'mdptype':
                    mdp_data['metadata']['mdptype'] = tokens[1]
                elif keyword == 'discount':
                    mdp_data['metadata']['discount'] = float(tokens[1])
                elif keyword == 'transition':
                    s1 = int(tokens[1])
                    ac = int(tokens[2])
                    s2 = int(tokens[3])
                    r = float(tokens[4])
                    p = float(tokens[5])

                    # Initialize transitions and rewards arrays if None
                    if mdp_data['transitions'] is None:
                        num_states = mdp_data['metadata']['numStates']
                        num_actions = mdp_data['metadata']['numActions']
                        mdp_data['transitions'] = np.zeros((num_states, num_actions, num_states))
                        mdp_data['rewards'] = np.zeros((num_states, num_actions, num_states))

                    # Update the transition probability and reward
                    mdp_data['transitions'][s1, ac, s2] = p
                    mdp_data['rewards'][s1, ac, s2] = r
    except UnicodeError:
        # If utf-16 encoding fails, fall back to utf-8
        with open(mdp_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                keyword = tokens[0]
                if keyword == 'numStates':
                    mdp_data['metadata']['numStates'] = int(tokens[1])
                elif keyword == 'numActions':
                    mdp_data['metadata']['numActions'] = int(tokens[1])
                elif keyword == 'mdptype':
                    mdp_data['metadata']['mdptype'] = tokens[1]
                elif keyword == 'discount':
                    mdp_data['metadata']['discount'] = float(tokens[1])
                elif keyword == 'transition':
                    s1 = int(tokens[1])
                    ac = int(tokens[2])
                    s2 = int(tokens[3])
                    r = float(tokens[4])
                    p = float(tokens[5])

                    # Initialize transitions and rewards arrays if None
                    if mdp_data['transitions'] is None:
                        num_states = mdp_data['metadata']['numStates']
                        num_actions = mdp_data['metadata']['numActions']
                        mdp_data['transitions'] = np.zeros((num_states, num_actions, num_states))
                        mdp_data['rewards'] = np.zeros((num_states, num_actions, num_states))

                    # Update the transition probability and reward
                    mdp_data['transitions'][s1, ac, s2] = p
                    mdp_data['rewards'][s1, ac, s2] = r

    return mdp_data


def extract_policy(policy_file):
    policy = np.array([])
    with open(policy_file, 'r') as file:
        for line in file:
            policy = np.append(policy, int(line.strip()[0]))
    policy = policy.astype(int)
    return policy            

def vi(mdp_data, convergence_threshold=1e-15):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']
    discount = mdp_data['metadata']['discount']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']

    # Initialize value function and policy
    value_t = np.zeros(numStates, dtype=np.float32)
    policy = np.zeros(numStates, dtype=int)  # Assuming integer policy

    while True:
        value_t1 = np.zeros(numStates)

        # Compute the Q-values for all states and actions in a vectorized manner
        q_values = np.sum(transitions * (rewards + discount * value_t), axis=2)

        # Find the optimal policy and update the value function
        policy = np.argmax(q_values, axis=1)
        value_t1 = np.max(q_values, axis=1)

        # Check for convergence using the max absolute difference
        max_diff = np.max(np.abs(value_t1 - value_t))
        if max_diff < convergence_threshold:
            break

        value_t = value_t1

    return value_t, policy


def lp(mdp_data):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']
    discount = mdp_data['metadata']['discount']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']

    # Create a linear programming problem
    lp_problem = pulp.LpProblem("Finding_Optimal_Value_Function", pulp.LpMaximize)
    
    # Define LP variables for state values
    V = [pulp.LpVariable(f"V_{s}") for s in range(numStates)]     

    # Define the objective function (maximize the sum of state values)
    lp_problem += -1*pulp.lpSum(V)

    # Define constraints
    for s in range(numStates):
        for a in range(numActions):
            rhs = pulp.lpSum(transitions[s, a, s1] * (rewards[s, a, s1] + discount * V[s1]) for s1 in range(numStates))
            lp_problem += rhs - V[s] <= 0

    # Suppress solver's output
    # solver = pulp.COIN_CMD(msg=0)
    # Solve the problem
    lp_problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Get the optimal values
    optimal_values = np.array([V[s].varValue for s in range(numStates)])

    q_value = np.zeros((numStates, numActions))

    for s in range(numStates):
        for a in range(numActions):
            for s1 in range(numStates):
                q_value[s, a] += transitions[s, a, s1]*(rewards[s, a, s1] + discount*(V[s].varValue))

    optimal_policy = np.argmax(q_value, axis = 1)

    return optimal_values, optimal_policy

def q(pi, mdp_data):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']
    discount = mdp_data['metadata']['discount']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']
    value = linear_eq_solver(mdp_data, pi)
    q_value = np.zeros((numStates, numActions))
    for s_loop in range(numStates):
        for a_loop in range(numActions):
            for s1 in range(numStates):
                q_value[s_loop, a_loop] += transitions[s_loop, a_loop, s1] * (rewards[s_loop, a_loop, s1] + discount * value[s1])
    return q_value

def howards_policy_iteration(mdp_data):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']
    discount = mdp_data['metadata']['discount']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']

    # Initialize a random policy
    policy = np.random.randint(numActions, size=numStates)
    # print("Initial policy ", policy)
    # Convergence criterion
    max_iterations = 100000

    for iteration in range(max_iterations):
        # Policy Evaluation: Solve the linear system for the current policy
        value_function = linear_eq_solver(mdp_data, policy)

        # Policy Improvement: Update the policy based on Information Sets (IAs)
        new_policy = policy_improvement(mdp_data, value_function, discount, policy)

        # Check for convergence (if the policy remains unchanged)
        if np.all(new_policy == policy):
            # print(f"new policy {new_policy} old policy {policy}")
            break

        policy = new_policy  # Update the policy for the next iteration

    optimal_value = value_function
    optimal_policy = policy
    return optimal_value, optimal_policy

def policy_improvement(mdp_data, value_function, discount, policy):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']

    new_policy = policy.copy()
    q_function = q(policy, mdp_data)
    
    # Precompute Q-values for all state-action pairs
    q_values = np.zeros((numStates, numActions))
    for s in range(numStates):
        for a in range(numActions):
            q_values[s, a] = q_function[s, a]
    
    for s in range(numStates):
        best_actions = np.where(q_values[s, :] > value_function[s] + 1e-10)[0]
        
        if best_actions.size > 0:
            new_policy[s] = np.random.choice(best_actions)
    
    return new_policy

def linear_eq_solver(mdp_data, policy):
    numStates = mdp_data['metadata']['numStates']
    numActions = mdp_data['metadata']['numActions']
    discount = mdp_data['metadata']['discount']
    transitions = mdp_data['transitions']
    rewards = mdp_data['rewards']

    # Initialize A and b
    A = np.zeros((numStates, numStates))
    b = np.zeros(numStates)

    for s in range(numStates):
        # Compute the diagonal element
        A[s, s] = 1 - discount * transitions[s, policy[s], s]

        for s1 in range(numStates):
            if s != s1:
                A[s, s1] = -discount * transitions[s, policy[s], s1]
        
        b[s] = np.sum(transitions[s, policy[s], :] * rewards[s, policy[s], :])

    # Solve the linear system
    output = np.linalg.solve(A, b)
    
    return output

if __name__ == "__main__":
    parser.add_argument("--mdp",type=str, required=True)
    parser.add_argument("--algorithm",type=str,default="vi", required=False)
    parser.add_argument("--policy",type=str, required=False)
    args = parser.parse_args()

    # if(args.algorithm == "hpi"):
    #     pass
    if(args.policy != None):
        mdp_data = extract_data_mdp(args.mdp)
        policies = extract_policy(args.policy)
        values = linear_eq_solver(mdp_data, policies)
        for value, policy in zip(values, policies):
            # Use f-strings to format the output with at least 6 decimal places
            output = f"{value:.6f}\t{policy:.6f}"
            # Print the formatted output
            print(output)

    elif(args.algorithm == "vi"):
        mdp_data = extract_data_mdp(args.mdp)
        values, policies = vi(mdp_data)
        values = np.round(values, 6)
        policies = np.round(policies, 6)
        for value, policy in zip(values, policies):
            # Use f-strings to format the output with at least 6 decimal places
            output = f"{value:.6f}\t{policy:.6f}"
            # Print the formatted output
            print(output)

    elif(args.algorithm == "lp"):
        mdp_data = extract_data_mdp(args.mdp)
        values, policies = lp(mdp_data)
        for value, policy in zip(values, policies):
            # Use f-strings to format the output with at least 6 decimal places
            output = f"{value:.6f}\t{policy:.6f}"
            # Print the formatted output
            print(output)

    elif(args.algorithm == "hpi"):
        mdp_data = extract_data_mdp(args.mdp)
        values, policies = howards_policy_iteration(mdp_data)
        for value, policy in zip(values, policies):
            # Use f-strings to format the output with at least 6 decimal places
            output = f"{value:.6f}\t{policy:.6f}"
            # Print the formatted output
            print(output)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # print(f"Time taken: {elapsed_time} seconds")


    
