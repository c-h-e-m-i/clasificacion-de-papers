import os
import csv
import ast
import re
import gc
import zipfile
import collections
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support, f1_score, precision_recall_curve
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


SEED = 2627
DEV_PART = 0.15
MAX_LEN = 384
MODEL_NAME = "allenai/scibert_scivocab_uncased"
CORPUS_DIR = "data"
TRAIN_FILE = os.path.join(CORPUS_DIR, "dataset_train.csv")
TEST_FILE = os.path.join(CORPUS_DIR, "dataset_test.csv")

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
def adjust_thresholds(scores, y_true, min_th=0.01, max_th=0.99, n_th=200):
    # Creamos un vector con tantos elementos como clases, inicializado a ceros:
    n_classes = y_true.shape[1]
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


### DATOS DE TEST ###

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Cargando conjunto de test...")
raw_test_papers, _ = load_corpus(TEST_FILE, delimiter=',', is_test=True)

test_texts = []
for title, summary, keywords in raw_test_papers:
    text_item = f"Title: {title}. Abstract: {summary}"
    if keywords and keywords != 'nan':
        text_item += f". Keywords: {keywords}"
    test_texts.append(text_item)

test_dataset = PaperDataset(test_texts, None, tokenizer)


### EXPERIMENTACIÓN ###

results_summary = []
best_overall_f1 = -1.0
best_test_preds = None
best_experiment_name = ""

print("\n" + "="*70)
print(f" INICIANDO EXPERIMENTO: Entrenamiento {dup_str}")
print("="*70)

raw_papers, raw_terms = load_corpus(TRAIN_FILE, delimiter=',')

mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(raw_terms)
num_classes = len(mlb.classes_)

all_texts = [
    f"Title: {title}. Abstract: {summary}"
    for title, summary, _ in raw_papers
]

X_train_tok, X_dev_tok, y_train, y_dev = train_test_split(
    all_texts, y_bin, test_size=DEV_PART, shuffle=True, random_state=SEED
)

train_dataset = PaperDataset(X_train_tok, y_train, tokenizer)
dev_dataset = PaperDataset(X_dev_tok, y_dev, tokenizer)

# Inicializar modelo
output_dir = f"/tmp/results_scibert"
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
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
    report_to="none"
)

def compute_metrics_trainer(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits)) # sigmoide
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

# Entrenar modelo
trainer.train()

# Evaluar en validación y test
dev_predictions = trainer.predict(dev_dataset)
dev_probs = 1 / (1 + np.exp(-dev_predictions.predictions)) # sigmoide

test_predictions = trainer.predict(test_dataset)
test_probs = 1 / (1 + np.exp(-test_predictions.predictions)) # sigmoide

# Sin ajuste de umbrales (Umbral = 0.5)
default_th = np.full(num_classes, 0.5)
preds_dev_fixed = predict_with_th(dev_probs, default_th)
metrics_fixed = compute_metrics(y_dev, preds_dev_fixed)
print("\n=== Resultados en Validación con Umbral Estándar (0.5) ===")
print(metrics_fixed)

test_preds_fixed = predict_with_th(test_probs, default_th)
csv_fixed = f"submission_scibert__fijo.csv"
save_submission_csv(test_preds_fixed, mlb, TEST_FILE, csv_fixed)

exp_name_fixed = f"SciBERT | Umbral Fijo (0.5)"
results_summary.append({
    "Experimento": exp_name_fixed,
    "Ajuste Umbrales": "No (0.5)",
    "F1 Macro": metrics_fixed["f1"],
    "Accuracy": metrics_fixed["accuracy"],
    "Precision": metrics_fixed["precision"],
    "Recall": metrics_fixed["recall"],
    "Hamming Loss": metrics_fixed["hamming_loss"]
})

if metrics_fixed["f1"] > best_overall_f1:
    best_overall_f1 = metrics_fixed["f1"]
    best_submission_csv = csv_fixed
    best_experiment_name = exp_name_fixed

# Con umbrales optimizados por clase (adjust_thresholds)
adj_thresholds = adjust_thresholds(dev_probs, y_dev)
preds_dev_adj = predict_with_th(dev_probs, adj_thresholds)
metrics_adj = compute_metrics(y_dev, preds_dev_adj)

test_preds_adj = predict_with_th(test_probs, adj_thresholds)
csv_adj = f"submission_scibert_adj.csv"
save_submission_csv(test_preds_adj, mlb, TEST_FILE, csv_adj)

exp_name_adj = f"SciBERT | Umbrales Ajustados"
results_summary.append({
    "Experimento": exp_name_adj,
    "Ajuste Umbrales": "Sí (Por Clase) Adjust",
    "F1 Macro": metrics_adj["f1"],
    "Accuracy": metrics_adj["accuracy"],
    "Precision": metrics_adj["precision"],
    "Recall": metrics_adj["recall"],
    "Hamming Loss": metrics_adj["hamming_loss"]
})

if metrics_adj["f1"] > best_overall_f1:
    best_overall_f1 = metrics_adj["f1"]
    best_submission_csv = csv_adj
    best_experiment_name = exp_name_adj


# ==========================================
# 4. TABLA COMPARATIVA Y GENERACIÓN DE ENTREGA
# ==========================================

df_results = pd.DataFrame(results_summary)

print("\n" + "="*80)
print(" RESUMEN DE EXPERIMENTOS Y EVALUACIÓN EN VALIDACIÓN")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

print(f"\nGanador absoluto: {best_experiment_name} (F1 Macro = {best_overall_f1:.4f})")

print(f"\nConfiguración Ganadora: {best_experiment_name}")
print(f" -> Mejor F1 Macro: {best_overall_f1:.4f}")
print(f" -> Archivo Especifico: '{best_submission_csv}'")
print(f" -> Archivo Copiado como Ganador: 'submission_scibert_best_overall.csv'")