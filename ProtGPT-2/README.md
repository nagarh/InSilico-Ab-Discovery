# ProtGPT-2: Generating Antibody Sequences In Silico with GPT-2

The existing ProtGPT-2 model has been trained on the most frequent V gene combinations (IGHV3-IGKV1) of human paired antibody sequences to generate antibody sequences in silico.

## Directory Structure

```
ProtGPT-2/
├── data/
│   ├── processed/ # Model input sequences (<vh|vl>) for training
│   │   ├── Prot_input_seq_train_data.csv # training data
│   │   ├── Prot_input_seq_val_data.csv   # validation data
│   │   └── Prot_input_seq.csv  # train + val data
│   ├── raw/
│   │   ├── IGHV3_IGKV1_Paired_BType_Naive_B_Cells_Disease_None.csv
│   ├── Training_loss.csv
│   ├── Training_loss.png
│   └── Validation_loss.png
├── logs/
│   └── events.out.tfevents.* # model training logs
├── models/
│   ├── checkpoints/
│   └── protgpt2_antibody_model/
│       ├── added_tokens.json
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── vovab.json
│       └── special_tokens_map.json
├── notebooks/
│   ├── Data_processing.ipynb # notebook to write input sequence format for model
│   └── train.ipynb # training noteboook of model
├── scripts/
│   └── train.py # model training script 
└── README.md

```

## Folder Descriptions

### [data/](./data)
Contains the datasets used for training and validation.

- [`processed/`](./data/processed): Preprocessed data ready for model training and validation.
  - `Prot_input_seq_train_data.csv`: Training sequences.
  - `Prot_input_seq_val_data.csv`: Validation sequences.
  - `Prot_input_seq.csv`: Combined sequences.
  
- [`raw/`](./data/raw): Raw data files before preprocessing.
  - `IGHV3_IGKV1_Paired_BType_Naive_B_Cells_Disease_None.csv`: Raw antibody sequences.
- `Training_loss.csv`: Training loss data.
- `Training_loss.png`
- `Validation_loss.png`

### [`logs/`](./logs)
Contains TensorBoard log files for visualizing training metrics.

- `events.out.tfevents.*`: TensorBoard event files.

### [`models/`](./models)
Contains the trained models and checkpoints.

- [`checkpoints/`](./models/checkpoints): Directory for saving model checkpoints during training.
- [`protgpt2_antibody_model/`](./models/protgpt2_antibody_model/): Directory for the final trained model and tokenizer.
  - `added_tokens.json`: Additional tokens added to the tokenizer.
  - `config.json`: Model configuration file.
  - `generation_config.json`: Configuration for text generation.
  - `merges.txt`: Byte pair encoding merges.
  - `model.safetensors`: Trained model weights.
  - `special_tokens_map.json`: Mapping of special tokens.
  - `tokenizer_config.json` : Configuration of tokenizer
  - `tokenizer.json` : Model tokenizer
  - `vocab.json` : Model vocabulary

### [`notebooks/`](./notebooks/)
Jupyter notebooks for data processing and training.

- [`Data_processing.ipynb`](./notebooks/Data_processing.ipynb): Notebook to process model input antibody sequences and write in input format (<vh|vl>)
- [`train.ipynb`](./notebooks/train.ipynb): Notebook for training and evaluating the model.

### [`scripts/`](./scripts/)

- [`train.py`](./scripts/train.py): Script for training the ProtGPT-2 model.

## Usage

To load the trained ProtGPT-2 model using the `transformers` library, follow these steps:

1. Install the `transformers` library:
    ```sh
    pip install transformers
    ```

2. Load the model and tokenizer:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "./models/protgpt2_antibody_model"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path)


### Model training parameters 
* Optimizer - Adam ( learning rate: 5e-4, weight_decay:1e-5)
* batch size -  16
* train/validation - 90/10
* The model was trained on a single NVIDIA H100 GPU with a batch size of 16 and for 2 epochs

### References
- Existing model is taken from Hugging face : https://huggingface.co/nferruz/ProtGPT2
- Research paper : https://doi.org/10.1038/s41467-022-32007-7





