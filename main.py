import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces`")
warnings.filterwarnings(
    "ignore", message="Torch was not compiled with flash attention")

# Verifica se o GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo em uso: {device}")

# Carrega o tokenizer do BERTimbau
tokenizer = BertTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')

# Função para carregar os dados


def load_data(file_path):
    return pd.read_json(file_path, lines=True)

# Função para carregar o conjunto de teste (test.csv)


def read_data(set):
    return pd.read_csv(set)

# Dataset customizado para PyTorch


class PunDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Função para criar DataLoader


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PunDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy() if 'label' in df else np.zeros(len(df)),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

# Função para treinar o modelo


def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    # Utilizando tqdm para exibir progresso
    for d in tqdm(data_loader, desc="Treinando"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Função para avaliar o modelo


def eval_model(model, data_loader, device):
    model = model.eval()
    correct_predictions = 0
    losses = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Avaliando"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Função para predição


def predict(model, data_loader, device):
    model = model.eval()
    texts = []
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Fazendo Predições"):
            texts.extend(d["text"])
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(outputs.logits, dim=1)

            # Mover os tensores para a CPU antes de convertê-los em listas
            predictions.extend(preds.cpu().numpy())  # Convertendo para NumPy
            real_values.extend(labels.cpu().numpy())  # Convertendo para NumPy

    return predictions, real_values

# Função para criar o arquivo de submissão


def create_submission_file(df_test, predictions):
    df = pd.DataFrame(columns=['id', 'label'])
    df['id'] = df_test['id']
    df['label'] = predictions
    df.to_csv('submission.csv', index=False)


# Bloco principal para garantir que o multiprocessing funcione corretamente
if __name__ == "__main__":
    # Carregar o modelo do BERTimbau para classificação
    model = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased', num_labels=2)
    model = model.to(device)

    # Carregar e preparar os dados
    df_train = load_data('train.jsonl')
    df_val = load_data('validation.jsonl')

    # Hiperparâmetros
    BATCH_SIZE = 16  # Testar valores como 16 ou 32, dependendo da GPU
    MAX_LEN = 128
    EPOCHS = 1 
    LEARNING_RATE = 2e-5  # Taxa de aprendizado ajustada para 3e-5

    # Criar DataLoaders
    train_data_loader = create_data_loader(
        df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(
        df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    # Otimizador com weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Treinamento
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, device)
        val_acc, val_loss = eval_model(model, val_data_loader, device)

        print(f'Train Loss: {train_loss} | Train Acc: {train_acc}')
        print(f'Val Loss: {val_loss} | Val Acc: {val_acc}')

    # Avaliação final no conjunto de validação
    val_preds, val_labels = predict(model, val_data_loader, device)

    # Relatório de classificação usando as predições e os valores reais
    print("\nRelatório de Classificação (Conjunto de Validação):")
    print(classification_report(val_labels, val_preds))

    # Carregar conjunto de teste (test.csv)
    df_test = read_data('test.csv')

    # Criar DataLoader para o conjunto de teste
    test_data_loader = create_data_loader(
        df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Fazer predições no conjunto de teste
    test_preds, _ = predict(model, test_data_loader, device)

    # Gerar arquivo de submissão
    create_submission_file(df_test, test_preds)
    print("Arquivo de submissão gerado: submission.csv")
