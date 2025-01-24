from transformers import  AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
import torch
import os 


def generate_sequences_on_gpus(model, tokenizer, num_sequences, max_length=300):
    
    model = model.cuda()

    # Prepare input sequences
    input_sequence = "<"
    input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids.cuda()

    # Generate sequences
    generated_sequences = []
    for _ in range(num_sequences):
        outputs = model.generate(
            input_ids,
            do_sample=True,
            top_k=25,
            repetition_penalty=1.2,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        final_sequence = tokenizer.decode(outputs[0])
        print(f"Final sequence: {final_sequence}")
        generated_sequences.append(final_sequence)

    # Save to file
    with open("../data/generated_sequences_test.txt", "w") as f:
        for seq in generated_sequences:
            f.write(seq + "\n")

    print(f"Generated {len(generated_sequences)} sequences saved to 'generated_sequences.txt'.")


   
if __name__ == "__main__":

    # Load the trained model and tokenizer
    model_path = "../models/BioGPT_antibody_model"  # Path to saved model
    tokenizer = PreTrainedTokenizerFast.from_pretrained('../custom_biogpt_tokenizer')
    model = AutoModelForCausalLM.from_pretrained(model_path)

    generate_sequences_on_gpus(model, tokenizer, num_sequences=2000)



