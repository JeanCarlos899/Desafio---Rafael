import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore")

# Verificar se a GPU está disponível
hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o tokenizador do BERTimbau
tokenizador = BertTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')


# Função simples para carregar os dados
def carregar_dados(caminho_arquivo):
    return pd.read_json(caminho_arquivo, lines=True)


# Classe simples de Dataset
class PunDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizador, tam_max):
        self.codificacoes = tokenizador(
            textos.tolist(),
            truncation=True,
            padding='max_length',
            max_length=tam_max,
            return_tensors='pt'
        )
        self.etiquetas = torch.tensor(etiquetas) if etiquetas is not None else torch.zeros(
            len(textos), dtype=torch.long)

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        item = {chave: tensor[idx]
                for chave, tensor in self.codificacoes.items()}
        item['etiquetas'] = self.etiquetas[idx]
        return item


# Função para criar o DataLoader
def criar_data_loader(df, tokenizador, tam_max, tam_lote):
    ds = PunDataset(
        textos=df['text'],
        etiquetas=df['label'] if 'label' in df else None,
        tokenizador=tokenizador,
        tam_max=tam_max
    )
    return DataLoader(ds, batch_size=tam_lote)


# Função de treino básica
def treinar_epoca(modelo, data_loader, otimizador, hardware):
    modelo.train()
    perdas = []
    predicoes_corretas = 0

    for lote in tqdm(data_loader, desc="Treinando"):
        input_ids = lote["input_ids"].to(hardware)
        attention_mask = lote["attention_mask"].to(hardware)
        etiquetas = lote["etiquetas"].to(hardware)

        otimizador.zero_grad()

        saidas = modelo(input_ids=input_ids,
                        attention_mask=attention_mask, labels=etiquetas)
        perda = saidas.loss
        predicoes = torch.argmax(saidas.logits, dim=1)

        predicoes_corretas += (predicoes == etiquetas).sum().item()
        perdas.append(perda.item())

        perda.backward()
        otimizador.step()

    return predicoes_corretas / len(data_loader.dataset), np.mean(perdas)


# Função de avaliação
def avaliar_modelo(modelo, data_loader, hardware):
    modelo.eval()
    predicoes_corretas = 0
    perdas = []

    with torch.no_grad():
        for lote in tqdm(data_loader, desc="Avaliando"):
            input_ids = lote["input_ids"].to(hardware)
            attention_mask = lote["attention_mask"].to(hardware)
            etiquetas = lote["etiquetas"].to(hardware)

            saidas = modelo(input_ids=input_ids,
                            attention_mask=attention_mask, labels=etiquetas)
            perda = saidas.loss
            predicoes = torch.argmax(saidas.logits, dim=1)

            predicoes_corretas += (predicoes == etiquetas).sum().item()
            perdas.append(perda.item())

    return predicoes_corretas / len(data_loader.dataset), np.mean(perdas)


# Função para previsões
def prever(modelo, data_loader, hardware):
    modelo.eval()
    predicoes = []
    valores_reais = []

    with torch.no_grad():
        for lote in tqdm(data_loader, desc="Fazendo Previsões"):
            input_ids = lote["input_ids"].to(hardware)
            attention_mask = lote["attention_mask"].to(hardware)
            etiquetas = lote["etiquetas"].to(hardware)

            saidas = modelo(input_ids=input_ids, attention_mask=attention_mask)
            predicoes_lote = torch.argmax(saidas.logits, dim=1)

            predicoes.extend(predicoes_lote.cpu().numpy())
            valores_reais.extend(etiquetas.cpu().numpy())

    return predicoes, valores_reais

# Função para criar o arquivo de submissão


def criar_arquivo_submissao(df_teste, predicoes):
    submissao_df = pd.DataFrame({'id': df_teste['id'], 'label': predicoes})
    submissao_df.to_csv('submission.csv', index=False)


# Função para plotar a matriz de confusão
def plotar_matriz_confusao(valores_reais, previsoes):
    matriz_confusao = confusion_matrix(y_true=valores_reais, y_pred=previsoes)
    vn, fp, fn, vp = matriz_confusao.ravel()
    matriz_customizada = [[vp, fp], [fn, vn]]

    sns.heatmap(matriz_customizada, annot=True, cmap="Blues", fmt='g',
                xticklabels=['Trocadilho', 'Não trocadilho'],
                yticklabels=['Trocadilho', 'Não trocadilho'])

    plt.xlabel('Classificação real')
    plt.ylabel('Classificação prevista')
    plt.title('Matriz de Confusão')
    plt.show()


# Parte principal do código
if __name__ == "__main__":
    modelo = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased', num_labels=2)
    modelo.to(hardware)

    df_treino = carregar_dados('corpus/train.jsonl')
    df_validacao = carregar_dados('corpus/validation.jsonl')

    '''
    TAM_LOTE - O tamanho do lote de dados que será processado a cada iteração
    foi definido como 32. Esse valor foi escolhido para garantir que o modelo
    seja treinado de forma eficiente sem consumir muita memória. Chegamos a este
    valor após testar diferentes tamanhos de lote e observar o desempenho do
    modelo.

    TAM_MAX - Após a análise dos comprimentos dos tokens gerados pelo BertTokenizer, 
    obtivemos os seguintes resultados:

        Comprimento mínimo: 9 tokens
        Comprimento máximo: 54 tokens
        Comprimento médio: 19.73 tokens
        Mediana do comprimento: 19.0 tokens

        Comprimento no percentil 90%: 27.0 tokens
        Comprimento no percentil 95%: 30.0 tokens
        Comprimento no percentil 99%: 38.0 tokens

    Decidimos definir o tamanho máximo dos textos como 54 tokens para garantir
    que todos dos textos sejam cobertos sem desperdiçar recursos computacionais.

    EPOCAS - O número de épocas foi definido como 1. Esse valor foi escolhido
    pois não obtivemos resultados relevantes ao aumentar o número de épocas. Até
    chegamos a reduzir a taxa de aprendizado para evitar o overfitting em n 
    épocas, mas obtivemos no máximo a mesma acurácia com apenas uma época em 
    troca de um tempo de treinamento absurdamente maior.

    TAXA_APRENDIZADO - A taxa de aprendizado foi definida como 2e-5. Esse valor
    foi escolhido após testar diferentes valores e observar o desempenho do modelo.
    '''
    
    TAM_LOTE = 32
    TAM_MAX = 54
    EPOCAS = 1
    TAXA_APRENDIZADO = 2e-5

    data_loader_treino = criar_data_loader(
        df_treino, tokenizador, TAM_MAX, TAM_LOTE)
    data_loader_validacao = criar_data_loader(
        df_validacao, tokenizador, TAM_MAX, TAM_LOTE)

    otimizador = AdamW(modelo.parameters(), lr=TAXA_APRENDIZADO)

    for epoca in range(EPOCAS):
        print(f'Época {epoca + 1}/{EPOCAS}')
        acc_treino, perda_treino = treinar_epoca(
            modelo, data_loader_treino, otimizador, hardware)
        acc_validacao, perda_validacao = avaliar_modelo(
            modelo, data_loader_validacao, hardware)

        print(
            f'Perda Treino: {perda_treino:.4f} | Acurácia Treino: {acc_treino:.4f}')
        print(
            f'Perda Validação: {perda_validacao:.4f} | Acurácia Validação: {acc_validacao:.4f}')

    previsoes_validacao, etiquetas_validacao = prever(
        modelo, data_loader_validacao, hardware)
    print("\nRelatório de Classificação (Conjunto de Validação):")
    print(classification_report(etiquetas_validacao, previsoes_validacao))

    # Plotando a matriz de confusão após a avaliação do modelo
    plotar_matriz_confusao(etiquetas_validacao, previsoes_validacao)

    df_teste = pd.read_csv('corpus/test.csv')
    data_loader_teste = criar_data_loader(
        df_teste, tokenizador, TAM_MAX, TAM_LOTE)
    previsoes_teste, _ = prever(modelo, data_loader_teste, hardware)

    criar_arquivo_submissao(df_teste, previsoes_teste)
    print("Arquivo de submissão gerado: submission.csv")
