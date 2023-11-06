import subprocess
import re
import matplotlib.pyplot as plt

# Define the parameter ranges for p and q
p_value = 0.3
q_values = [0.6, 0.7, 0.8, 0.9, 1]  # Varying q values for [Graph 2]

# Initialize lists to store q values and corresponding value functions
q_list = []
value_function_list = []

# Run the commands and generate policyfile.txt for different q values
for q in q_values:
    # Run encoder.py to create football_mdp
    subprocess.run([
        "python",
        "encoder.py",
        "--opponent",
        "data/football/test-1.txt",
        "--p",
        str(p_value),
        "--q",
        str(q)
    ], stdout=open("football_mdp", "w"))

    # Run planner.py to create value
    subprocess.run([
        "python",
        "planner.py",
        "--mdp",
        "football_mdp"
    ], stdout=open("value", "w"))

    # Run decoder.py to create policyfile.txt
    subprocess.run([
        "python",
        "decoder.py",
        "--value-policy",
        "value",
        "--opponent",
        "data/football/test-1.txt"
    ], stdout=open("policyfile.txt", "w"))

    # Extract the value function for state 0509081 from policyfile.txt
    value_function = None
    with open("policyfile.txt", "r") as file:
        for line in file:
            match = re.match(r'0509081 \d+ ([\d.]+)', line)
            if match:
                value_function = float(match.group(1))
                break

    # Append q and corresponding value function to the lists
    q_list.append(q)
    value_function_list.append(value_function)

# Create [Graph 2] for q values
plt.plot(q_list, value_function_list, marker='o', linestyle='-')
plt.title("Probability of winning vs. q for p=0.3")
plt.xlabel("q")
plt.ylabel("Probability of winning")
plt.grid()

# Save the plot as an image (e.g., PNG)
plt.savefig("graph2.png")

# Show the plot
plt.show()