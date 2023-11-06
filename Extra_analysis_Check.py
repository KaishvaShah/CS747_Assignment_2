import random,argparse,sys
parser = argparse.ArgumentParser()

parser.add_argument("--sol",default="data/football/sol-1.txt", type=str, required=True)

args = parser.parse_args()

# Paths to input and output files
input_file1_path = args.sol
input_file2_path = "policyfile.txt"
output_file_path = "differing_lines.txt"

# Read the contents of the first file (sol-1.txt)
with open(input_file1_path, "r") as file1:
    lines1 = file1.readlines()

# Read the contents of the second file (policyfile.txt)
with open(input_file2_path, "r", encoding="utf-16") as file2:
    lines2 = file2.readlines()

# Create a function to extract the third column as a float
def extract_third_column(line):
    columns = line.split()
    if len(columns) >= 3:
        return float(columns[2])
    return None

# Compare the third column values and count differences
num_differences = 0

# Initialize a list to store differing lines
differing_lines = []

expected_number_goals = 0

for line1, line2 in zip(lines1, lines2):
    value1 = extract_third_column(line1)
    value2 = extract_third_column(line2)
    expected_number_goals += value2
    if value1 is not None and value2 is not None and abs(value1 - value2) > 1e-5:
        num_differences += 1
        differing_lines.append(f"File 1: {line1.strip()}\nFile 2: {line2.strip()}\n")

print(f"Number of different values in the third column: {num_differences}")
print(f"Expected number of goals scored: {expected_number_goals}")
# Save differing lines to a separate file
with open(output_file_path, "w") as output_file:
    output_file.writelines(differing_lines)

print(f"Differing lines saved to {output_file_path}")
