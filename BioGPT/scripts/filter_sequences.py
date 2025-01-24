 # Define the valid amino acids
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Function to check if the sequence contains only valid amino acids
def contains_only_valid_aa(seq):
    return all(aa in valid_amino_acids for aa in seq)

# Function to check the length of heavy chain (VH) and light chain (VL)
def is_valid_chain_length(heavy_chain, light_chain):
    return 110 <= len(heavy_chain) <= 140 and 105 <= len(light_chain) <= 120

def process_sequences(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        raw_number = 1
        for line in infile:
            #print("Raw input line:", repr(line))
            # Remove spaces
            line = line.replace(" ", "").strip()
            
            # Strip outer brackets
            if line.startswith('<') and line.endswith('>'):
                line = line[1:-1]
                print(line)
            
            # Skip lines without hyphen
            if '-' not in line:
                continue

            # Split into parts
            parts = line.split('-')
            if len(parts) != 2:
                continue

            # Assign heavy and light chain
            vh = parts[0]
            vl = parts[1]

            # Validate and write sequences
            if contains_only_valid_aa(vh) and contains_only_valid_aa(vl) and is_valid_chain_length(vh, vl):
                outfile.write(f">{raw_number}_H\n{vh}\n")
                outfile.write(f">{raw_number}_L\n{vl}\n")
            
            raw_number += 1

# Process the sequences
input_file = "../data/generated_sequences.txt"
output_file = "../data/filtered_sequences.fasta"
process_sequences(input_file, output_file)


