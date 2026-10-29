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

LOCAL_SPECTER_PATH = "./specter2/no_duplicados"  # Carpeta que contiene model.safetensors y config.json
SPECTER_MODEL_NAME = "allenai/specter2_base"

LOCAL_SCIBERT_PATH = "./specter2/no_duplicados"
SCIBERT_MODEL_NAME = "allenai/specter2_base"

ALPHA = 0.5

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


# Cargamos archivos:
df_train = pd.read_csv("dataset_train.csv")
df_train['title_clean'] = df_train['titles'].astype(str).str.strip().str.lower()
df_train = df_train.drop_duplicates(subset=['title_clean']).reset_index(drop=True)

# Binarizamos las etiquetas originales de arXiv:
train_terms = [ast.literal_eval(terms) for terms in df_train['terms']]
mlb = MultiLabelBinarizer()
num_classes = len(mlb.classes_)
y_all = mlb.fit_transform(train_terms)

df_test = pd.read_csv("dataset_test.csv")

### SPECTER ###
specter_tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL_NAME)

# Unimos título y resumen usando el token separador nativo del modelo:
train_texts = [f"{t}{specter_tokenizer.sep_token}{s}" for t, s in zip(df_train['titles'], df_train['summaries'])]

# Definimos una partición de entrenamiento y otra de desarrollo:
X_train, X_dev, y_train, y_dev = train_test_split(train_texts, y_all, test_size=0.15, random_state=2627, shuffle=True)

test_texts = [f"{t}{specter_tokenizer.sep_token}{s}{specter_tokenizer.sep_token}{k}" for t, s, k in zip(df_test['titles'], df_test['summaries'], df_test["keywords"])]

test_dataset_specter = PaperDataset(test_texts, None, specter_tokenizer)

train_dataset_specter = PaperDataset(X_train, y_train, specter_tokenizer)
dev_dataset_specter = PaperDataset(X_dev, y_dev, specter_tokenizer)

specter_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_SPECTER_PATH)

# Crear el Trainer solo para inferencia/evaluación
specter_trainer = Trainer(
    model=specter_model,
    args=TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        report_to="none"
    ),
    data_collator=default_data_collator
)

print("Obteniendo predicciones del modelo SPECTER cargado...")
dev_pred_specter = specter_trainer.predict(dev_dataset_specter)
dev_probs_specter = 1 / (1 + np.exp(-dev_pred_specter.predictions))

test_pred_specter = specter_trainer.predict(test_dataset_specter)
test_probs_specter = 1 / (1 + np.exp(-test_pred_specter.predictions))

metrics_specter_alone = compute_metrics(y_dev, predict_with_th(dev_probs_specter, np.full(num_classes, 0.5)))
print(f"-> Aislado SciBERT (Fijo 0.5) | F1 Macro DEV: {metrics_specter_alone['f1']:.4f}")


### SCIBERT ###
scibert_tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)

# Unimos título y resumen usando el token separador nativo del modelo:
train_texts = [f"{t}{scibert_tokenizer.sep_token}{s}" for t, s in zip(df_train['titles'], df_train['summaries'])]

# Definimos una partición de entrenamiento y otra de desarrollo:
X_train, X_dev, y_train, y_dev = train_test_split(train_texts, y_all, test_size=0.15, random_state=2627, shuffle=True)

test_texts = [f"{t}{scibert_tokenizer.sep_token}{s}{scibert_tokenizer.sep_token}{k}" for t, s, k in zip(df_test['titles'], df_test['summaries'], df_test["keywords"])]

test_dataset_scibert = PaperDataset(test_texts, None, scibert_tokenizer)

train_dataset_scibert = PaperDataset(X_train, y_train, scibert_tokenizer)
dev_dataset_scibert = PaperDataset(X_dev, y_dev, scibert_tokenizer)

scibert_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_SCIBERT_PATH)

# Crear el Trainer solo para inferencia/evaluación
scibert_trainer = Trainer(
    model=scibert_model,
    args=TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        report_to="none"
    ),
    data_collator=default_data_collator
)

print("Obteniendo predicciones del modelo SCIBERT cargado...")
dev_pred_scibert = scibert_trainer.predict(dev_dataset_scibert)
dev_probs_scibert = 1 / (1 + np.exp(-dev_pred_scibert.predictions))

test_pred_scibert = scibert_trainer.predict(test_dataset_scibert)
test_probs_scibert = 1 / (1 + np.exp(-test_pred_scibert.predictions))

metrics_scibert_alone = compute_metrics(y_dev, predict_with_th(dev_probs_scibert, np.full(num_classes, 0.5)))
print(f"-> Aislado SciBERT (Fijo 0.5) | F1 Macro DEV: {metrics_scibert_alone['f1']:.4f}")


### ENSAMBLE ###

print("\n=== Ensamble y Calibración de Umbrales ===")

# Combinación ponderada de probabilidades
dev_probs_ensemble = ALPHA * dev_probs_scibert + (1 - ALPHA) * dev_probs_specter
test_probs_ensemble = ALPHA * test_probs_scibert + (1 - ALPHA) * test_probs_specter

metrics_fixed = compute_metrics(y_dev, predict_with_th(dev_probs_ensemble, np.full(num_classes, 0.5)))

opt_thresholds = adjust_thresholds(dev_probs_ensemble, y_dev)
preds_dev_opt = predict_with_th(dev_probs_ensemble, opt_thresholds)
metrics_opt = compute_metrics(y_dev, preds_dev_opt)


print("\n" + "="*80)
print(" RESULTADOS FINALES EN DEV (VALIDACIÓN)")
print("="*80)
print(f"1. SPECTER Aislado                              | F1 Macro: {metrics_specter_alone['f1']:.4f}")
print(f"2. SciBERT Aislado                              | F1 Macro: {metrics_scibert_alone['f1']:.4f}")
print(f"3. Ensamble                                     | F1 Macro: {metrics_fixed['f1']:.4f}")
print(f"4. Ensamble + Umbrales Optimizados              | F1 Macro: {metrics_opt['f1']:.4f}  <-- [MEJOR RESULTADO]")
print("="*80)


# Evaluación
test_preds_fixed = predict_with_th(test_probs_ensemble, predict_with_th(dev_probs_ensemble, np.full(num_classes, 0.5)))
metrics_final_fixed = compute_metrics(y_dev, test_preds_fixed)
print(f"Ensamble                            | F1 Macro: {metrics_final_fixed['f1']:.4f}")
save_submission_csv(test_preds_fixed, mlb, TEST_FILE, "submission_scibert_specter_ensemble.csv")

test_preds_final = predict_with_th(test_probs_ensemble, opt_thresholds)
metrics_final_th = compute_metrics(y_dev, test_preds_final)
print(f"Ensamble + Umbrales Optimizados     | F1 Macro: {metrics_final_th['f1']:.4f}")
save_submission_csv(test_preds_final, mlb, TEST_FILE, "submission_scibert_specter_ensemble_th.csv")
