import subprocess
import re
import matplotlib.pyplot as plt

# Define the parameter ranges for p and q
p_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
q_value = 0.7  # Constant q value for [Graph 1]

# Initialize lists to store p values and corresponding value functions
p_list = []
value_function_list = []

# Run the commands and generate policyfile.txt for different p values
for p in p_values:
    # Run encoder.py to create football_mdp
    subprocess.run([
        "python",
        "encoder.py",
        "--opponent",
        "data/football/test-1.txt",
        "--p",
        str(p),
        "--q",
        str(q_value)
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

    # Append p and corresponding value function to the lists
    p_list.append(p)
    value_function_list.append(value_function)

# Create [Graph 1] for p values
plt.plot(p_list, value_function_list, marker='o', linestyle='-')
plt.title("Probability of winning vs. p for q=0.7")
plt.xlabel("p")
plt.ylabel("Probability of winning")
plt.grid()

# Save the plot as an image (e.g., PNG)
plt.savefig("graph1.png")

# Show the plot
plt.show()
