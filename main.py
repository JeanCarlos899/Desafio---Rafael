import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from pathlib import Path


def load_data(set):
    return pd.read_json(set, lines=True)


def read_data(set):
    return pd.read_csv(set)


def create_x_y(df, set):
    if set == 'train':
        x = df['text']
        y = df['label']
        return x, y
    else:
        return df['text']


def vectorize_text(x_train, x_val, vectorizer):
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_val_vectorized = vectorizer.transform(x_val)
    return x_train_vectorized, x_val_vectorized


def classify(x_train_vectorized, y_train, x_val_vectorized):
    model = GradientBoostingClassifier()
    model.fit(x_train_vectorized, y_train)
    y_pred = model.predict(x_val_vectorized)
    return y_pred, model


def create_submission_file(df_test, predictions):
    df = pd.DataFrame(columns=['id', 'label'])
    df['id'] = df_test['id']
    df['label'] = predictions
    df.to_csv('submission.csv', index=False)


input_dir = Path('C:/Users/jeanc/Desktop/Desafio - Rafael')

df_train = load_data(input_dir.joinpath('train.jsonl'))
x_train, y_train = create_x_y(df_train, 'train')

df_val = load_data(input_dir.joinpath('validation.jsonl'))
x_val, y_val = create_x_y(df_val, 'train')

vectorizer = TfidfVectorizer()
x_train_vectorized, x_val_vectorized = vectorize_text(
    x_train, x_val, vectorizer)
predictions, model = classify(x_train_vectorized, y_train, x_val_vectorized)

print("\nRelatório de Classificação (Conjunto de Validação):")
print(classification_report(y_val, predictions))

df_test = read_data(input_dir.joinpath('test.csv'))
x_test = create_x_y(df_test, 'test')
x_test_vectorized = vectorizer.transform(x_test)
test_predictions = model.predict(x_test_vectorized)
create_submission_file(df_test, test_predictions)
