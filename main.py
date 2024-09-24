import torch.cuda.amp as amp
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERTimbau tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')

# Function to load data


def load_data(file_path):
    return pd.read_json(file_path, lines=True)

# Custom Dataset class for PyTorch


class PunDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels) if labels is not None else torch.zeros(
            len(texts), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Function to create DataLoader


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PunDataset(
        texts=df['text'],
        labels=df['label'] if 'label' in df else None,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

# Function to train the model with mixed precision


def train_epoch(model, data_loader, optimizer, scheduler, scaler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Using autocast for mixed precision operations
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

        correct_predictions += (preds == labels).sum().item()
        losses.append(loss.item())

        # Backward pass and optimizer update using scaler for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    return correct_predictions / len(data_loader.dataset), np.mean(losses)

# Function to evaluate the model


def eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    losses = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            losses.append(loss.item())

    return correct_predictions / len(data_loader.dataset), np.mean(losses)

# Function for prediction


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Making Predictions"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    return predictions, real_values

# Function to create the submission file


def create_submission_file(df_test, predictions):
    submission_df = pd.DataFrame({'id': df_test['id'], 'label': predictions})
    submission_df.to_csv('submission.csv', index=False)


# Main block to ensure multiprocessing works correctly
if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased', num_labels=2)
    model.to(device)  # type: ignore

    df_train = load_data('train.jsonl')
    df_val = load_data('validation.jsonl')

    BATCH_SIZE = 32
    MAX_LEN = 40
    EPOCHS = 1
    LEARNING_RATE = 2e-5

    train_data_loader = create_data_loader(
        df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(
        df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    scaler = amp.GradScaler()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, scheduler, scaler, device)
        val_acc, val_loss = eval_model(model, val_data_loader, device)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    val_preds, val_labels = predict(model, val_data_loader, device)
    print("\nClassification Report (Validation Set):")
    print(classification_report(val_labels, val_preds))

    df_test = pd.read_csv('test.csv')
    test_data_loader = create_data_loader(
        df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    test_preds, _ = predict(model, test_data_loader, device)

    create_submission_file(df_test, test_preds)
    print("Submission file generated: submission.csv")
