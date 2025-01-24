import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys 
from transformers import PreTrainedTokenizerFast

# Step 1: Dataset class to generate input tensors from the processed sequences 

class AntibodyDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }
    

# Step 2: Load Tokenizer and from the model
model_name = "microsoft/biogpt"

# Use your custom tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained('../custom_biogpt_tokenizer')

pad_token_id = tokenizer.pad_token_id 
bos_token_id = tokenizer.bos_token_id

# Load the model configuration and update the vocab size
config = AutoConfig.from_pretrained(model_name)
config.bos_token_id = bos_token_id  # Set it to 1
config.pad_token_id = pad_token_id  # Set it to 3
config.vocab_size = len(tokenizer)  # Update vocab size for your custom tokenizer
model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))

# Load training sequences
sequences_train = pd.read_csv('../data/seq_train_data.csv', header=None)
sequences_train = '<' + sequences_train + '>'
print(sequences_train[0][0])
sequences_train.columns = ['sequences']
sequences_train = sequences_train['sequences'].tolist()

# Load validation sequences
sequences_valid = pd.read_csv('../data/seq_val_data.csv', header=None)
sequences_valid = '<' + sequences_valid + '>'
sequences_valid.columns = ['sequences']
sequences_valid = sequences_valid['sequences'].tolist()

# Create Dataset and DataLoader
dataset_train = AntibodyDataset(sequences_train, tokenizer)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

dataset_valid = AntibodyDataset(sequences_valid, tokenizer)
dataloader_valid = DataLoader(dataset_valid, batch_size=8, shuffle=True)

# Step 4: Set Up Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4) # AdamW optimizer
num_epochs = 2 # Number of epochs to train model

# Step 5: Initialize TensorBoard writer
writer = SummaryWriter(log_dir="../logs")

# Create models directory to save the trained model, updated tokenizer and checkpoints in 'checkpoints' directory
os.makedirs("../models/checkpoints", exist_ok=True)

# Model evaluation function

def validate(model, val_dataloader, device, writer, epoch, step):
    model.eval()  
    val_loss = 0

    with torch.no_grad():  
        for step_val, batch in enumerate(tqdm(val_dataloader, desc="Validation Progress", file=sys.stdout.flush())):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            
            # Log step loss to TensorBoard
            writer.add_scalar('Validation Loss/Step', loss.item(), epoch * len(val_dataloader) + step_val)

    avg_val_loss = val_loss / len(val_dataloader)
    writer.add_scalar('Validation Loss/epoch', avg_val_loss, epoch * len(val_dataloader) + step)


    print(f"Validation Loss: {avg_val_loss}")


# Training loop for the model 

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(tqdm(dataloader_train)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Log step loss to TensorBoard
        writer.add_scalar('Training Loss/Step', loss.item(), epoch * len(dataloader_train) + step)
        #print(f"Step Loss: {loss.item()}")

        # Save checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            checkpoint_dir = f"../models/checkpoints/checkpoint_epoch_{epoch+1}_step_{step+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")


        # Validate after every 500 steps
        if (step + 1) % 500 == 0:
            print(f"Running validation at step {step + 1}...")
            validate(model, dataloader_valid, device, writer, epoch, step)


    # Log average epoch loss to TensorBoard
    avg_epoch_loss = epoch_loss / len(dataloader_train)
    writer.add_scalar('Training Loss/Epoch', avg_epoch_loss, epoch)

    print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss}")


# Create models directory to save the trained model and updated tokenizer
os.makedirs("../models/BioGPT_antibody_model", exist_ok=True)

# Step 7: Save the Trained Model and Tokenizer
model.save_pretrained("../models//BioGPT_antibody_model")
tokenizer.save_pretrained("../models/BioGPT_antibody_model")

# Step 8: Close the TensorBoard writer
writer.close()


