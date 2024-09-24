# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
import warnings

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore")

# Inicialização do tokenizer do BERTimbau
tokenizer = BertTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')

# Função para carregar os dados


def load_data(file_path):
    return pd.read_json(file_path, lines=True)


# Carregamento do conjunto de treinamento
df_train = load_data('corpus/train.jsonl')

# Verifique se a coluna 'text' existe no dataframe
if 'text' not in df_train.columns:
    raise ValueError("A coluna 'text' não foi encontrada no dataframe.")

# Calcular os comprimentos dos textos após a tokenização
print("Calculando os comprimentos dos textos...")
text_lengths = [len(tokenizer.encode(text, max_length=512,
                    truncation=False)) for text in df_train['text']]

# Plotar o histograma dos comprimentos dos textos
print("Plotando o histograma dos comprimentos dos textos...")
plt.figure(figsize=(10, 6))
# gráfico cinza e branco
plt.hist(text_lengths, bins=50, color='gray', edgecolor='white')
plt.xlabel('Comprimento do Texto (número de tokens)')
plt.ylabel('Frequência')
plt.title('Distribuição dos Comprimentos de Texto')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Calcular estatísticas descritivas
print()
print("Estatísticas descritivas dos comprimentos dos textos:")
print(f'Comprimento mínimo: {np.min(text_lengths)} tokens')
print(f'Comprimento máximo: {np.max(text_lengths)} tokens')
print(f'Comprimento médio: {np.mean(text_lengths):.2f} tokens')
print(f'Mediana do comprimento: {np.median(text_lengths)} tokens')
print()
print(
    f'Comprimento no percentil 90%: {np.percentile(text_lengths, 90)} tokens')
print(
    f'Comprimento no percentil 95%: {np.percentile(text_lengths, 95)} tokens')
print(
    f'Comprimento no percentil 99%: {np.percentile(text_lengths, 99)} tokens')
