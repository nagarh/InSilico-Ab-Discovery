# This script is used to generate Human Antibody paired sequences (IGHV3-IGKV1 germline pairs) ... 
# using the trained model. 
# The generated sequences are saved to a file for further analysis.
# Change 'number_of_sequences' to the number of sequences you want to generate.

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import torch

# Set the working directory
os.chdir('/fs/project/PAA0031/hemantn/OAS_database/Paired_seq/ProtGPT-2')

# Step 1: Load the trained model and tokenizer
model_path = "./models/protgpt2_antibody_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Step 2: Set up the pipeline with the loaded model and tokenizer
protgpt2 = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Parameters for sequence generation
generation_params = {
    "max_length": 300,
    "do_sample": True,
    "top_k": 950,
    "repetition_penalty": 1.2,
    "eos_token_id": 30,  # End of sequence token ID
    "truncation": True
}

starting_token = "<"  # Starting token
num_sequences = 20000  # Total number of sequences to generate
batch_size = 200  # Number of sequences per batch

# File to save generated sequences
output_file = "./data/generated_sequences.txt"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Clear CUDA cache
torch.cuda.empty_cache()

# Generate sequences in batches and save them directly to a file
with open(output_file, "w") as f:
    for batch_num in range(0, num_sequences, batch_size):
        current_batch_size = min(batch_size, num_sequences - batch_num)
        
        # Generate sequences
        sequences = protgpt2(
            starting_token,
            num_return_sequences=current_batch_size,
            **generation_params
        )
        
        # Write generated sequences to file
        for seq in sequences:
            f.write(seq['generated_text'] + "\n")
        
        # Optionally, print progress
        print(f"Generated {batch_num + current_batch_size}/{num_sequences} sequences.")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()

print(f"Finished generating {num_sequences} sequences. Saved to '{output_file}'.")
