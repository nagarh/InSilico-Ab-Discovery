import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys
from typing import List, Dict


# Change working directory
os.chdir('/data/hn533621/InSilico-Ab-Discovery/ProtGPT-2/')


# Model validation function
def validate(model: torch.nn.Module, val_dataloader: DataLoader, device: torch.device, writer: SummaryWriter, epoch: int, step: int) -> None:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for step_val, batch in enumerate(tqdm(val_dataloader, desc="Validation Progress", file=sys.stdout)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()
            writer.add_scalar('Validation Loss/Step', loss.item(), epoch * len(val_dataloader) + step_val)

    avg_val_loss = val_loss / len(val_dataloader)
    writer.add_scalar('Validation Loss/epoch', avg_val_loss, epoch * len(val_dataloader) + step)
    print(f"Validation Loss: {avg_val_loss}")

# Dataset class to generate input tensors from the processed sequences
class AntibodyDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        encoding = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Training function
def train(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: torch.device, epochs: int, writer: SummaryWriter, tokenizer) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5) # AdamW optimizer
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", file=sys.stdout)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar('Training Loss/Step', loss.item(), epoch * len(train_dataloader) + step)

            # Save checkpoint every 500 steps
            if (step + 1) % 500 == 0:
                checkpoint_dir = f"./models/checkpoints/checkpoint_epoch_{epoch+1}_step_{step+1}"
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
                validate(model, val_dataloader, device, writer, epoch, step)

        avg_train_loss = train_loss / len(train_dataloader)
        writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch)
        print(f"Training Loss: {avg_train_loss}")
        
    # Save the model and tokenizer after training
    model.save_pretrained("./models/protgpt2_antibody_model")
    tokenizer.save_pretrained("./models/protgpt2_antibody_model")

def main():
    # Load training and validation sequences
    train_sequences = pd.read_csv('./data/processed/Prot_input_seq_train_data.csv', header=None, 
                                  names=['sequences'])['sequences'].tolist()
    valid_sequences = pd.read_csv('./data/processed/Prot_input_seq_val_data.csv', header=None, 
                                  names=['sequences'])['sequences'].tolist()

    model_name = "nferruz/ProtGPT2" # Model name
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Load tokenizer

    # Add a padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load the model configuration and update the vocab size for the special tokens
    # model is initialized with random weights
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = len(tokenizer)  # Update vocab size for special tokens
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    # Create datasets and dataloaders
    train_dataset = AntibodyDataset(train_sequences, tokenizer) # Training dataset
    val_dataset = AntibodyDataset(valid_sequences, tokenizer) # Validation dataset
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='/logs/')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs('./models/checkpoints/', exist_ok=True)
    os.makedirs('./models/protgpt2_antibody_model/', exist_ok=True)

    # Train the model
    train(model, train_dataloader, val_dataloader, device, epochs=2, writer=writer, tokenizer=tokenizer)

    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()


