from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models


# Define the vocabulary
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

special_tokens = ['<', '>', '<pad>', '<unk>']
vocab = special_tokens + amino_acids

# Create the base tokenizer
base_tokenizer = Tokenizer(models.Unigram())

# Add special tokens to the tokenizer
special_tokens_dict = {
    "bos_token": "<",
    "eos_token": ">",
    "pad_token": "<pad>",
    "unk_token": "<unk>"
}

# Create the wrapper tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=base_tokenizer,
    bos_token="<",
    eos_token=">",
    pad_token="<pad>",
    unk_token="<unk>"
)

# Add the vocabulary
tokenizer.add_tokens(amino_acids)


# Test the tokenizer
sequence = "<ACDEFGHIK-LMNPQRSTVWY>"
encoded = tokenizer(sequence, return_tensors="pt")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Encoded sequence: {encoded['input_ids']}")
print(f"Decoded sequence: {tokenizer.decode(encoded['input_ids'][0])}")

# Save the tokenizer 
tokenizer.save_pretrained('../custom_biogpt_tokenizer')