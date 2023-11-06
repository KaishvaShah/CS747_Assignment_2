import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import sys
# Define MDP data structures
mdp_data = {
    'metadata': {'numStates' : None, 'numActions' : None, 'mdptype' : None, 'discount' : None},
    'transitions': None,
    'rewards': None
}

def update_r_position(r, state, R_mdp_data):
    # Provide possible new r positions with probabilities
    rl = [max(r[0] - 1, 0), r[1]]
    rr = [min(r[0] + 1, 3), r[1]]
    ru = [r[0], max(r[1] - 1, 0)]
    rd = [r[0], min(r[1] + 1, 3)]
    all_new_r_positions = [rl, rr, ru, rd]
    possible_r_positions = [all_new_r_positions[i] for i in range(len(R_mdp_data['R_info'][state])) if (R_mdp_data['R_info'][state][i] != 0)]
    positive_prob_r = [R_mdp_data['R_info'][state][i] for i in range(len(R_mdp_data['R_info'][state])) if (R_mdp_data['R_info'][state][i] != 0)]
    return possible_r_positions, positive_prob_r
    # return [rl, rr, ru, rd], R_mdp_data['R_info'][state]

def extract_data_game_positions(game_positions_path):
    # Define MDP data structures
    R_mdp_data = {
        'state': [],
        'R_info': {},
    }

    # Open and read the MDP file
    with open(game_positions_path, 'r') as file:
        for line in file:
            tokens = line.split()
            if tokens[0] == 'state':
                continue
            state = tokens[0]
            R_mdp_data['state'].append(state)
            state = str(state)
            left = float(tokens[1])
            R_mdp_data['R_info'][state] = [left]
            right = float(tokens[2])
            R_mdp_data['R_info'][state].append(right)
            up = float(tokens[3])
            R_mdp_data['R_info'][state].append(up)
            down = float(tokens[4])
            R_mdp_data['R_info'][state].append(down)
        return R_mdp_data

def formulating_state(b1, b2, r, poss):
    b1_pos = b1[0] + 1 + b1[1]*4
    b2_pos = b2[0] + 1 + b2[1]*4
    r_pos = r[0] + 1 + r[1]*4
    # print(f"r_pos here is {r_pos} {r[0]} {r[1]}")
    poss = str(poss)

    if(b1_pos <= 9):
        b1_pos = "0" + str(b1_pos)
    else:
        b1_pos = str(b1_pos)
    if(b2_pos <= 9):
        b2_pos = "0" + str(b2_pos)
    else:
        b2_pos = str(b2_pos)
    if(r_pos <= 9):
        r_pos = "0" + str(r_pos)
    else:
        r_pos = str(r_pos)
    
    new_state = b1_pos + b2_pos + r_pos + poss

    return new_state

def movement_in_possession(b1, b2, r, poss, p, state, action, transitions, rewards, end_state, r_prob=1, multiplication_factor = 1):
    # Successful transition to another viable state
    new_state = formulating_state(b1, b2, r, poss)
    state_action_state = state + str(action) + new_state

    if state_action_state in transitions:
        transitions[state_action_state] += (1 - 2 * p) * multiplication_factor * r_prob
    else:
        transitions[state_action_state] = (1 - 2 * p) * multiplication_factor *r_prob

    if state_action_state in rewards:
        rewards[state_action_state] += 0 
    else:
        rewards[state_action_state] = 0

    # Going to end state with prob 2*p
    end_episode(b1, b2, r, poss, p, state, action, transitions, rewards, end_state, with_prob=(1- (1 - 2*p)*multiplication_factor) * r_prob)


def movement_not_in_possession(b1, b2, r, poss, p, state, action, transitions, rewards, end_state, r_prob = 1, multiplication_factor = 1):
    # Successful transition to another viable state
    new_state = formulating_state(b1, b2, r, poss)
    state_action_state = state + str(action) + new_state
    
    if state_action_state in transitions:
        transitions[state_action_state] += r_prob *multiplication_factor*(1 - p)
    else:
        transitions[state_action_state] = r_prob * multiplication_factor * (1 - p)

    if state_action_state in rewards:
        rewards[state_action_state] += 0  # You might want to modify this if rewards need to accumulate.
    else:
        rewards[state_action_state] = 0
    # Going to end state with prob p
    end_episode(b1, b2, r, poss, p, state, action, transitions, rewards, end_state, with_prob = (1 - multiplication_factor*(1 - p)) * r_prob)

def end_episode(b1, b2, r, poss, p, state, action, transitions, rewards, end_state, with_prob=0, reward_awarded=0):
    # if(reward_awarded == 0):
    #     reward_awarded = -1
    
    new_state = end_state
    state_action_state = state + str(action) + new_state
    
    if state_action_state in transitions:
        transitions[state_action_state] += with_prob
    else:
        transitions[state_action_state] = with_prob
    
    if state_action_state in rewards:
        rewards[state_action_state] += reward_awarded  # You might want to modify this if rewards need to accumulate.

    else:
        rewards[state_action_state] = reward_awarded
    # rewards[state_action_state] = rewards[state_action_state]

    if(end_state == "1111110"):
        rewards[state_action_state] = 1

def movement(state, action, transitions, rewards, end_state, p, q, R_mdp_data):
    b1 = [(int(states[:2]) + 3)%4, (int(states[:2]) - 1)//4]
    # print(b1)
    b2 = [(int(states[2:4]) + 3)%4, (int(states[2:4]) - 1)//4]
    # print(b2)
    r = [(int(states[4:6]) + 3)%4, (int(states[4:6]) - 1)//4]
    # print(r)
    poss = int(states[-1])
    # print(poss)

    # R position update has not happened yet
    possible_r, possible_r_prob =   update_r_position(r, state, R_mdp_data)
    # print(f"Possible positions of r {possible_r}")
    # print(f"Possible probabilities of r {possible_r_prob}")
    # print(state)
    # print(action)

    # R position update has happened
    old_b1 = b1.copy()
    old_b2 = b2.copy()
    for r_new, r_prob in zip(possible_r, possible_r_prob):
        b1 = old_b1.copy()
        b2 = old_b2.copy()
        if(action == 0): #Left
            if(b1[0] == 0):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                # b1 position is updated
                b1[0] = b1[0] - 1
                if(poss == 1): #If player 1 has possession
                    if((old_b1 == r_new and r == b1) or (b1 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 2 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 1): #Right
            if(b1[0] == 3):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob = r_prob)
            else:
                b1[0] = b1[0] + 1
                if(poss == 1): #If player 1 has possession
                    if((old_b1 == r_new and r == b1) or (b1 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 2 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 2): #Up
            if(b1[1] == 0):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b1[1] = b1[1] - 1
                if(poss == 1): #If player 1 has possession
                    if((old_b1 == r_new and r == b1) or (b1 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 2 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 3): #Down
            if(b1[1] == 3):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b1[1] = b1[1] + 1
                if(poss == 1): #If player 1 has possession
                    if((old_b1 == r_new and r == b1) or (b1 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)           
                else: #If player 2 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 4): #Left
            if(b2[0] == 0):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b2[0] = b2[0] - 1
                if(poss == 2): #If player 2 has possession
                    if((old_b2 == r_new and r == b2) or (b2 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 1 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 5): #Right
            if(b2[0] == 3):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b2[0] = b2[0] + 1
                if(poss == 2): #If player 2 has possession
                    if((old_b2 == r_new and r == b2) or (b2 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 1 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 6): #Up
            if(b2[1] == 0):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b2[1] = b2[1] - 1
                if(poss == 2): #If player 2 has possession
                    if((old_b2 == r_new and r == b2) or (b2 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)
                else: #If player 1 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

        elif(action == 7): #Down
            if(b2[1] == 3):
                # Going to end state - out of bounds
                end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob=r_prob)
            else:
                b2[1] = b2[1] + 1
                if(poss == 2): #If player 2 has possession
                    if((old_b2 == r_new and r == b2) or (b2 == r_new)): # if player has been tackled by swap of positions or same grid position then go to end state with prob r_prob
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob, multiplication_factor= 0.5)
                    else:
                        movement_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob) 
                else: #If player 1 has possession
                    movement_not_in_possession(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, r_prob)

    return


def on_same_vertical_line(p1, p2, p3):
    return p1[0] == p2[0] == p3[0]

def on_same_horizontal_line(p1, p2, p3):
    return p1[1] == p2[1] == p3[1]

def on_x_plus_y_line(p1, p2, p3):
    return p1[0] + p1[1] == p2[0] + p2[1] == p3[0] + p3[1]

def on_x_minus_y_line(p1, p2, p3):
    return p1[0] - p1[1] == p2[0] - p2[1] == p3[0] - p3[1]

def is_point_within_box(point, box):
    return (
        box[0][0] <= point[0] <= box[1][0]
        and box[0][1] <= point[1] <= box[1][1]
    )

def passing_intercept_check(b1, b2, r):
    if b1 == b2:
        return r == b1  # Return True only if r is coincident with b1 (and b2 if they are the same point)

    if (
        on_same_vertical_line(b1, b2, r)
        or on_same_horizontal_line(b1, b2, r)
        or on_x_plus_y_line(b1, b2, r)
        or on_x_minus_y_line(b1, b2, r)
    ):
        box = [(min(b1[0], b2[0]), min(b1[1], b2[1])), (max(b1[0], b2[0]), max(b1[1], b2[1]))]
        if is_point_within_box(r, box):
            return True

    return False

def passing(state, action, transitions, rewards, end_state, p, q, R_mdp_data):
    b1 = [(int(states[:2]) + 3)%4, (int(states[:2]) - 1)//4]
    # print(b1)
    b2 = [(int(states[2:4]) + 3)%4, (int(states[2:4]) - 1)//4]
    # print(b2)
    r = [(int(states[4:6]) + 3)%4, (int(states[4:6]) - 1)//4]
    # print(r)
    poss = int(states[-1])
    # print(poss)
    # Transition to change the poss
    if(poss == 1):
        poss = 2
    else:
        poss = 1
    # R position update has not happened yet
    possible_r, possible_r_prob = update_r_position(r, state, R_mdp_data)
    # R position update has happened
    for r_new, r_prob in zip(possible_r, possible_r_prob):
        probability_pass_succ = q - 0.1*max(abs(b1[0] - b2[0]), abs(b1[1] - b2[1]))

        if(passing_intercept_check(b1, b2, r_new)):
            # print("Intercept detected")
            probability_pass_succ *= 0.5

        # Transition to change the poss
        new_state = formulating_state(b1, b2, r_new, poss)

        state_action_state = state + str(action) + new_state

        if state_action_state in transitions:
            transitions[state_action_state] += probability_pass_succ*r_prob
        else:
            transitions[state_action_state] = probability_pass_succ*r_prob

        if state_action_state in rewards:
            rewards[state_action_state] += 0  # You might want to modify this if rewards need to accumulate.

        else:
            rewards[state_action_state] = 0

        probability_pass_fail = 1 - probability_pass_succ
        # Transition to lose possession to R and episode end
        end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob = probability_pass_fail*r_prob)

def shooting_intercept_check(r):
    r_pos = r[0] + 1 + r[1]*4
    if r_pos == 8 or r_pos == 12:
        return True
    return False
    
def shooting(state, action, transitions, rewards, end_state, p, q, R_mdp_data, score_state):
    b1 = [(int(states[:2]) + 3)%4, (int(states[:2]) - 1)//4]
    b2 = [(int(states[2:4]) + 3)%4, (int(states[2:4]) - 1)//4]
    r = [(int(states[4:6]) + 3)%4, (int(states[4:6]) - 1)//4]
    poss = int(states[-1]) 

    # R position update has not happened yet
    possible_r, possible_r_prob = update_r_position(r, state, R_mdp_data)
    # R position update has happened
    for r_new, r_prob in zip(possible_r, possible_r_prob):
        if(poss == 1):
            probability_shoot_succ = (q - 0.2*(3 - b1[0]))  
        elif(poss == 2):
            probability_shoot_succ = (q - 0.2*(3 - b2[0]))

        if(shooting_intercept_check(r_new)):
            probability_shoot_succ *= 0.5
        # Transition to change the state to end_state with reward 1
        end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, score_state, with_prob = probability_shoot_succ*r_prob, reward_awarded=r_prob)

        probability_shoot_fail = 1 - probability_shoot_succ
        # Transition to lose possession to R and episode end
        end_episode(b1, b2, r_new, poss, p, state, action, transitions, rewards, end_state, with_prob = probability_shoot_fail*r_prob)

if __name__ == "__main__":
    parser.add_argument("--opponent",type=str, required=True)
    parser.add_argument("--p",type=float, required=True)
    parser.add_argument("--q",type=float, required=True)
    args = parser.parse_args()
    # Specify the path to your MDP file
    game_positions_path = args.opponent
    p = args.p
    q = args.q

    R_mdp_data = extract_data_game_positions(game_positions_path)
    # print(R_mdp_data['R_info']['0101011'])
    # quit()
    transitions = {}
    rewards = {}

    '''
    transition is a dictionary with key concat(s1, a, s2) value is prob
    rewards is a dictionary with key concat(s1, a, s2) value is prob
    '''
    
    total_states = 16*16*16*2 + 1
    end_state = "0000000"
    score_state = "1111110"
    for states in R_mdp_data['state']:

        b1 = [(int(states[:2]) + 3)%4, (int(states[:2]) - 1)//4]
        b2 = [(int(states[2:4]) + 3)%4, (int(states[2:4]) - 1)//4]
        r = [(int(states[4:6]) + 3)%4, (int(states[4:6]) - 1)//4]
        poss = int(states[-1])

        for action in range(10):
            if(action != 8 or action != 9):
                movement(states, action, transitions, rewards, end_state, p, q, R_mdp_data)
            if(action == 8):
                passing(states, action, transitions, rewards, end_state, p, q, R_mdp_data)  
            elif(action == 9):
                shooting(states, action, transitions, rewards, end_state, p, q, R_mdp_data, score_state)
        # print(transitions)
        # quit()  
    all_states = []
    for (state_action_state,transition_val), (state_action_state,reward_val) in zip(transitions.items(), rewards.items()):
        initial_state = state_action_state[:7]
        final_state = state_action_state[8:]
        action = state_action_state[7]
        all_states.append(initial_state)

    # Set the desired encoding for standard output (stdout)
    desired_encoding = "utf-16"  # Use "utf-8-sig" for UTF-8 without BOM

    # Check if the current encoding is different from the desired encoding
    if sys.stdout.encoding != desired_encoding:
    # Reopen stdout with the desired encoding
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding=desired_encoding, buffering=1)


    all_list_set = set(all_states)
    # convert the set to the list
    unique_list = (list(all_list_set))
    unique_list.append(score_state)
    unique_list.append(end_state)
    print(f"numStates {len(unique_list)}")
    print(f"numActions 10")
    print(f"end {unique_list.index(end_state)} {unique_list.index(score_state)}")

    # quit()
    for (state_action_state,transition_val), (state_action_state,reward_val) in zip(transitions.items(), rewards.items()):
        initial_state = state_action_state[:7]
        final_state = state_action_state[8:]
        action = state_action_state[7]
        index_initial_state = unique_list.index(initial_state)
        index_final_state = unique_list.index(final_state)
        print(f"transition {index_initial_state} {action} {index_final_state} {reward_val} {transition_val}")

    print("mdptype episodic")
    print("discount  1")
    with open("encoder_mapping.txt", "w") as output_file:
        for item in unique_list:
            output_file.write(f"{item}\n")    

        
        
    

            



