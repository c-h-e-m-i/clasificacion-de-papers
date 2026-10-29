import os
import csv
import ast
import gc
import shutil
import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  hamming_loss,  precision_recall_fscore_support, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback, 
    set_seed, 
    default_data_collator
)

# ==========================================
# CONFIGURACIÓN GLOBAL Y HIPERPARÁMETROS
# ==========================================
SEED = 2627
DEV_PART = 0.15
MAX_LEN = 384
MODEL_NAME = "allenai/scibert_scivocab_uncased"
CORPUS_DIR = "data"
TRAIN_FILE = os.path.join(CORPUS_DIR, "dataset_train.csv")
TEST_FILE = os.path.join(CORPUS_DIR, "dataset_test.csv")
TOKEN_PATTERN = r"\S+" # Captura cualquier grupo de caracteres que NO contenga espacios

# Peso del Ensamble: 0.7 para SciBERT (Red Densa) + 0.3 para LR (TF-IDF Exacto)
ALPHA = 0.7  

set_seed(SEED)


# Función para cargar el corpus:
def load_corpus(filename, delimiter=',', duplicates=False, is_test=False):
    papers, terms = [], []

    # Abrimos el archivo 'filename' y lo leemos línea por línea:
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)

        # Saltamos la fila con los títulos de las columnas:
        next(reader)

        # Creamos un set para los títulos ya encontrados a fin de evitar duplicados:
        titles = set()

        # Extraemos las columnas del archivo:
        for row in reader:
            if not row or len(row) < 3:
                continue

            title = row[0].strip().lower()
            summary = row[1].strip().lower()
            keywords = row[2].strip().lower() if (is_test and len(row) > 2) else ""

            if not duplicates and not is_test:
                if title not in titles:
                    titles.add(title)
                else:
                    continue

            papers.append((title, summary, keywords))
            if not is_test:
                terms.append(ast.literal_eval(row[-1]))

    return papers, terms


# Función de tokenización:
def tokenize(text):
    # Buscamos palabras que empiecen por letras y puedan contener guiones
    # (para evitar ruido como "5k" y no excluir estructuras como "state-of-the-art"):
    pattern = re.compile(r"(?u)\b[a-zA-Z][\w-]+\b")
    return " ".join(pattern.findall(text))


# Función para calcular las métricas de evaluación:
def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    hl = hamming_loss(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hl,
    }


# Definimos una función para ajustar los umbrales de decisión de un clasificador:
def adjust_thresholds(scores, y_true, n_classes, min_th=0.01, max_th=0.99, n_th=200):
    # Creamos un vector con tantos elementos como clases, inicializado a ceros:
    optimal_thresholds = np.zeros(n_classes)
    
    # Definimos una lista de 'n_th' umbrales de decisión equiespaciados entre 'min_th' y 'max_th':
    thresholds = np.linspace(min_th, max_th, n_th)

    # Iteramos sobre cada clase:
    for c in range(n_classes):
        max_f1_c = -1.0
        best_th = 0.5

        # Obtenemos la columna de etiquetas y probabilidades para cada paper asociadas a la clase:
        y_c = y_true[:, c]
        scores_c = scores[:, c]

        # Iteramos sobre cada umbral:
        for th in thresholds:
            # Obtenemos las predicciones de etiquetas asociadas a ese umbral y calculamos su F1 individual:
            preds_c = (scores_c >= th).astype(int)
            f1_c = f1_score(y_c, preds_c, average="binary", zero_division=0)

            # Si el F1 mejora respecto al que tenía la clase ya, lo guardamos:
            if f1_c > max_f1_c:
                max_f1_c = f1_c
                best_th = th

        # En cada entrada 'i' de 'optimal_thresholds', almacenamos el mejor umbral encontrado para la clase 'c_i':
        optimal_thresholds[c] = best_th

    return optimal_thresholds


# Definimos asimismo una función para obtener las predicciones optimizadas dados los umbrales ajustados:
def predict_with_th(scores, thresholds):
    # Actualizamos las predicciones con los umbrales de decisión que hemos calculado:
    preds = (scores >= thresholds).astype(int)

    # Buscamos los papers que se hayan quedado sin clase:
    zero_rows = np.where(preds.sum(axis=1) == 0)[0] # NumPy devuelve una tupla de 1 elemento, por eso lo extraemos
    
    # Iteramos sobre cada paper sin clase, obtenemos la clase de mayor probabilidad y se la asignamos:
    for r in zero_rows:
        top_class = np.argmax(scores[r])
        preds[r, top_class] = 1

    return preds


def save_submission_csv(test_preds, mlb, test_file, csv_filename):
    pred_terms = mlb.inverse_transform(test_preds)
    pred_terms_formatted = [str(list(cats)) for cats in pred_terms]

    df_test_export = pd.read_csv(test_file, sep=',')

    df_test_export['terms'] = pred_terms_formatted
    df_test_export[['titles', 'summaries', 'terms']].to_csv(csv_filename, index=False)

    print(f"Guardado archivo CSV: '{csv_filename}'")


class PaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


### CARGA DE DATOS ###

raw_all, y_all = load_corpus(TRAIN_FILE, duplicates=False)

raw_test_papers, _ = load_corpus(TEST_FILE, delimiter=',', is_test=True)

mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y_all)
num_classes = len(mlb.classes_)

# Formatear textos
train_texts_raw = [
    f"title: {title}. abstract: {summary}"
    for title, summary, _ in raw_all
]

test_texts_raw = []
for title, summary, keywords in raw_test_papers:
    t_item = f"title: {title}. abstract: {summary}"
    if keywords and keywords != 'nan':
        t_item += f". keywords: {keywords}"
    test_texts_raw.append(t_item)

X_train_tok, X_dev_tok, y_train, y_dev = train_test_split(
    train_texts_raw, y_bin, test_size=DEV_PART, shuffle=True, random_state=SEED
)

### REGRESIÓN LOGÍSTICA (TF-IDF) ###

print("\n=== Entrenando Regresión Logística (TF-IDF) ===")
vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN, sublinear_tf=True, min_df=1)

X_train_tfidf = vectorizer.fit_transform(X_train_tok)
X_dev_tfidf = vectorizer.transform(X_dev_tok)
X_test_tfidf = vectorizer.transform(test_texts_raw)

lr_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.0, class_weight="balanced", random_state=SEED)).fit(X_train_tfidf, y_train)
dev_probs_lr = lr_model.predict_proba(X_dev_tfidf)

X_train_full = vectorizer.transform(train_texts_raw)

lr_model_full = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.0, class_weight="balanced", random_state=SEED)).fit(X_train_full, y_bin)
test_probs_lr = lr_model_full.predict_proba(X_test_tfidf)

# Evaluation rápida aislada de LR
metrics_lr_alone = compute_metrics(y_dev, predict_with_th(dev_probs_lr, np.full(num_classes, 0.5)))
print(f"-> Aislado LR (Fijo 0.5) | F1 Macro DEV: {metrics_lr_alone['f1']:.4f}")


### FINE-TUNING SciBERT ###

print("\n=== Fine-Tuning de SciBERT ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = PaperDataset(X_train_tok, y_train, tokenizer)
dev_dataset = PaperDataset(X_dev_tok, y_dev, tokenizer)
test_dataset = PaperDataset(test_texts_raw, None, tokenizer)

output_dir = "/tmp/results_scibert_hybrid"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_classes, 
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    report_to="none"
)

def compute_metrics_trainer(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return compute_metrics(labels, preds)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics_trainer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator=default_data_collator
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Inferencia de probabilidades de SciBERT
dev_pred_scibert = trainer.predict(dev_dataset)
dev_probs_scibert = 1 / (1 + np.exp(-dev_pred_scibert.predictions))


test_pred_scibert = trainer.predict(test_dataset)
test_probs_scibert = 1 / (1 + np.exp(-test_pred_scibert.predictions))

metrics_scibert_alone = compute_metrics(y_dev, predict_with_th(dev_probs_scibert, np.full(num_classes, 0.5)))
print(f"-> Aislado SciBERT (Fijo 0.5) | F1 Macro DEV: {metrics_scibert_alone['f1']:.4f}")


### ENSAMBLE ###

print("\n=== Ensamble y Calibración de Umbrales ===")

# Combinación ponderada de probabilidades
dev_probs_ensemble = ALPHA * dev_probs_scibert + (1 - ALPHA) * dev_probs_lr
test_probs_ensemble = ALPHA * test_probs_scibert + (1 - ALPHA) * test_probs_lr

metrics_fixed = compute_metrics(y_dev, predict_with_th(dev_probs_ensemble, np.full(num_classes, 0.5)))

opt_thresholds = adjust_thresholds(dev_probs_ensemble, y_dev)
preds_dev_opt = predict_with_th(dev_probs_ensemble, opt_thresholds)
metrics_opt = compute_metrics(y_dev, preds_dev_opt)


print("\n" + "="*80)
print(" RESULTADOS FINALES EN DEV (VALIDACIÓN)")
print("="*80)
print(f"1. LR Aislado (TF-IDF)                          | F1 Macro: {metrics_lr_alone['f1']:.4f}")
print(f"2. SciBERT Aislado                              | F1 Macro: {metrics_scibert_alone['f1']:.4f}")
print(f"3. Ensamble                                     | F1 Macro: {metrics_fixed['f1']:.4f}")
print(f"4. Ensamble + Umbrales Optimizados              | F1 Macro: {metrics_opt['f1']:.4f}")
print("="*80)


# Evaluación
test_preds_fixed = predict_with_th(test_probs_ensemble, predict_with_th(dev_probs_ensemble, np.full(num_classes, 0.5)))
metrics_final_fixed = compute_metrics(y_dev, test_preds_fixed)
print(f"Ensamble                            | F1 Macro: {metrics_final_fixed['f1']:.4f}")
save_submission_csv(test_preds_fixed, mlb, TEST_FILE, "submission_scibert_lr_ensemble.csv")

test_preds_final = predict_with_th(test_probs_ensemble, opt_thresholds)
metrics_final_th = compute_metrics(y_dev, test_preds_final)
print(f"Ensamble + Umbrales Optimizados     | F1 Macro: {metrics_final_th['f1']:.4f}")
save_submission_csv(test_preds_final, mlb, TEST_FILE, "submission_scibert_rl_ensemble_th.csv")
