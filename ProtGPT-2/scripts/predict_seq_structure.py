from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta
from Bio import SeqIO
import os 

os.chdir('/fs/ess/project/PAA0031/hemantn/OAS_database/Paired_seq/ProtGPT-2')

# Initialize PyRosetta
init_pyrosetta()

# This function is used to prepare the input format for IgFold from the FASTA file
# Igfold format is a dictionary with antibody name as key and chain type as value
# For example: {"Antibody name": {"H": "VHSEQ", "L": "VLSEQ"}}
def parse_fasta_to_sequences(fasta_file):
    antibodies = {}  # Dictionary to hold antibody sequences
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract antibody name and chain type
        antibody_name = record.id.split("_")[0]  # Extract name (e.g., "1")
        chain_type = record.id.split("_")[1].upper()  # Extract chain type (e.g., "H" or "L")
        
        # Initialize if not present
        if antibody_name not in antibodies:
            antibodies[antibody_name] = {}
        
        # Store the sequence for H or L
        antibodies[antibody_name][chain_type] = str(record.seq)
    
    return antibodies

# This function is used to write sequences to a fasts that were skipped during the structure prediction process
# Skipped sequences are those that are missing some parts of the chain or have other issues
def write_skipped_sequences(skipped_sequences, output_file):
    # Write skipped sequences to a FASTA file
    with open(output_file, "w") as fasta_file:
        for name, seq in skipped_sequences:
            if isinstance(seq, dict):  
                if "L" in seq:  # Write light chain if present
                    fasta_file.write(f">{name}_L\n{seq['L']}\n")
                if "H" in seq:  # Write heavy chain if present
                    fasta_file.write(f">{name}_H\n{seq['H']}\n")

# Input fasta file
fasta_file = './data/filtered_sequences.fasta'
output_file = "./data/skipped_sequences.fasta" # Folder where predicted structures will be saved

# Define the output folder
output_folder = "./data/predicted_structures"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist


# Parse the FASTA file
antibodies = parse_fasta_to_sequences(fasta_file)

skipped_sequences = []  # List to store skipped sequences

for antibody_name, sequences in antibodies.items():
    if "H" in sequences and "L" in sequences:  # Ensure both chains are present
        #print(antibody_name)
        #print(sequences)
        pred_pdb = os.path.join(output_folder, f"{antibody_name}.pdb")  # Output file path
        # Run IgFold to predict the structure
        igfold = IgFoldRunner()
        try:
            igfold.fold(
                pred_pdb,  # Output PDB file
                sequences=sequences,  # Antibody sequences
                do_refine=True,  # Refine with PyRosetta
                do_renum=True  # Renumber using Chothia scheme
            )
            print(f"Predicted structure saved: {pred_pdb}")
        except Exception as e:
            print(f"Error processing {antibody_name}: {e}")
            print("Skipping to the next sequence...")
            skipped_sequences.append((antibody_name, sequences))
    else:
        print(f"Skipping {antibody_name}: Having issues in the generated sequence")
        skipped_sequences.append((antibody_name, sequences))
    
# Write skipped sequences to FASTA file
write_skipped_sequences(skipped_sequences, output_file)

