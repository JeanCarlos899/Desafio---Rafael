
# Documentação do Código

## Sumário
1. [Introdução](#introdução)
2. [Pré-requisitos](#pré-requisitos)
3. [Descrição das Funções e Classes](#descrição-das-funções-e-classes)
4. [Fluxo Principal do Código](#fluxo-principal-do-código)
5. [Treinamento do Modelo](#treinamento-do-modelo)
6. [Avaliação e Previsão](#avaliação-e-previsão)
7. [Criação do Arquivo de Submissão](#criação-do-arquivo-de-submissão)
8. [Execução do Código](#execução-do-código)
9. [Referências](#referências)

---

## Introdução
Este código implementa um processo completo de treinamento, validação, avaliação e previsão de um modelo de classificação de texto usando BERT (BERTimbau - BERT em português). Ele usa a biblioteca `transformers` da Hugging Face e é treinado em um conjunto de dados de entrada fornecido pelo usuário.

A estrutura inclui a definição de um dataset personalizado, carregamento e processamento de dados, treinamento do modelo, avaliação, e geração de previsões em um arquivo CSV para submissão.

---

## Pré-requisitos
Antes de executar o código, certifique-se de que você possui as seguintes bibliotecas instaladas:

- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `tqdm`

### Instalação
Você pode instalar as bibliotecas necessárias com o seguinte comando:
```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

---

## Descrição das Funções e Classes

### 1. Função `carregar_dados(caminho_arquivo)`
Esta função carrega um arquivo JSON em formato de linhas e retorna um DataFrame Pandas.

**Parâmetros:**
- `caminho_arquivo` (str): Caminho para o arquivo JSON.

**Retorno:**
- `pd.DataFrame`: DataFrame contendo os dados carregados.

---

### 2. Classe `PunDataset(Dataset)`
Esta classe herda de `torch.utils.data.Dataset` e cria um dataset personalizado para o modelo BERT.

#### `__init__(self, textos, etiquetas, tokenizador, tam_max)`
- **Parâmetros:**
  - `textos` (List): Lista de textos.
  - `etiquetas` (List): Lista de etiquetas associadas aos textos.
  - `tokenizador` (BertTokenizer): Instância do tokenizador BERT.
  - `tam_max` (int): Tamanho máximo da sequência de entrada.

- **Atributos:**
  - `self.codificacoes`: Armazena as codificações dos textos usando o tokenizador BERT.
  - `self.etiquetas`: Armazena as etiquetas associadas aos textos.

#### `__len__(self)`
Retorna o comprimento do dataset.

#### `__getitem__(self, idx)`
Retorna um item (codificações e etiqueta) do dataset com base no índice fornecido.

---

### 3. Função `criar_data_loader(df, tokenizador, tam_max, tam_lote)`
Esta função cria um DataLoader para um DataFrame dado.

**Parâmetros:**
- `df` (pd.DataFrame): DataFrame contendo os dados.
- `tokenizador` (BertTokenizer): Instância do tokenizador BERT.
- `tam_max` (int): Tamanho máximo da sequência de entrada.
- `tam_lote` (int): Tamanho do lote para o DataLoader.

**Retorno:**
- `DataLoader`: DataLoader configurado para o dataset.

---

### 4. Função `treinar_epoca(modelo, data_loader, otimizador, dispositivo)`
Executa uma época de treinamento do modelo.

**Parâmetros:**
- `modelo` (BertForSequenceClassification): Modelo BERT para classificação.
- `data_loader` (DataLoader): DataLoader com os dados de treinamento.
- `otimizador` (AdamW): Otimizador do modelo.
- `dispositivo` (torch.device): Dispositivo de hardware ('cpu' ou 'cuda').

**Retorno:**
- `acc_treino`: Acurácia do modelo na época.
- `perda`: Média das perdas durante o treinamento.

---

### 5. Função `avaliar_modelo(modelo, data_loader, dispositivo)`
Avalia o modelo no conjunto de validação.

**Parâmetros:**
- `modelo` (BertForSequenceClassification): Modelo BERT para classificação.
- `data_loader` (DataLoader): DataLoader com os dados de validação.
- `dispositivo` (torch.device): Dispositivo de hardware ('cpu' ou 'cuda').

**Retorno:**
- `acc_validacao`: Acurácia no conjunto de validação.
- `perda_validacao`: Média das perdas durante a validação.

---

### 6. Função `prever(modelo, data_loader, dispositivo)`
Gera previsões para os dados do DataLoader.

**Parâmetros:**
- `modelo` (BertForSequenceClassification): Modelo BERT para classificação.
- `data_loader` (DataLoader): DataLoader com os dados para previsão.
- `dispositivo` (torch.device): Dispositivo de hardware ('cpu' ou 'cuda').

**Retorno:**
- `predicoes`: Lista das previsões do modelo.
- `valores_reais`: Lista das etiquetas reais.

---

### 7. Função `criar_arquivo_submissao(df_teste, predicoes)`
Cria um arquivo CSV para submissão com as previsões.

**Parâmetros:**
- `df_teste` (pd.DataFrame): DataFrame contendo os dados de teste.
- `predicoes` (List): Lista de previsões feitas pelo modelo.

---

## Fluxo Principal do Código
1. Carrega os dados de treinamento e validação utilizando a função `carregar_dados`.
2. Define os hiperparâmetros, como tamanho do lote, tamanho máximo da sequência, número de épocas e taxa de aprendizado.
3. Cria DataLoaders para os datasets de treinamento e validação usando `criar_data_loader`.
4. Inicializa o modelo BERT para classificação com o número de rótulos definido.
5. Move o modelo para o dispositivo apropriado (GPU ou CPU).
6. Define o otimizador `AdamW`.

---

## Treinamento do Modelo
O treinamento é executado por um número especificado de épocas, utilizando a função `treinar_epoca`. Para cada época, são calculadas a perda média e a acurácia.

```python
for epoca in range(EPOCAS):
    acc_treino, perda_treino = treinar_epoca(modelo, data_loader_treino, otimizador, dispositivo)
    acc_validacao, perda_validacao = avaliar_modelo(modelo, data_loader_validacao, dispositivo)
```

---

## Avaliação e Previsão
Após o treinamento, o modelo é avaliado usando o conjunto de validação. O relatório de classificação é gerado com o `classification_report` da biblioteca `scikit-learn`.

```python
previsoes_validacao, etiquetas_validacao = prever(modelo, data_loader_validacao, dispositivo)
print(classification_report(etiquetas_validacao, previsoes_validacao))
```

---

## Criação do Arquivo de Submissão
O arquivo de submissão é criado utilizando a função `criar_arquivo_submissao`.

```python
df_teste = pd.read_csv('test.csv')
data_loader_teste = criar_data_loader(df_teste, tokenizador, TAM_MAX, TAM_LOTE)
previsoes_teste, _ = prever(modelo, data_loader_teste, dispositivo)
criar_arquivo_submissao(df_teste, previsoes_teste)
```

---

## Execução do Código
Para executar o código, basta rodar o script principal. O modelo será treinado, avaliado e os resultados serão salvos em um arquivo `submission.csv`.

```bash
python nome_do_script.py
```

---

## Referências
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Transformers Library - Hugging Face](https://github.com/huggingface/transformers)
- [BERTimbau: Pre-trained BERT Models for Brazilian Portuguese](https://github.com/neuralmind-ai/portuguese-bert)

