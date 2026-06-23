# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def load_data():
    train_df = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    test_df = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    return train_df, test_df


def clean_data(df):
    out = df.copy()
    out = out.rename(columns={"default payment next month": "default"})
    out = out.drop(columns=["ID"])
    out = out.loc[df["MARRIAGE"] != 0]
    out = out.loc[df["EDUCATION"] != 0]
    out["EDUCATION"] = out["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    out = out.dropna()
    return out


def split_features_target(df):
    return df.drop(columns=["default"]), df["default"]


def build_pipeline(X):
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = list(set(X.columns).difference(cat_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("scaler", StandardScaler(with_mean=True, with_std=True), num_cols),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
        ]
    )


def build_grid_search(pipeline, X):
    param_grid = {
        "pca__n_components": [20, X.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }

    cv = StratifiedKFold(n_splits=10)
    scorer = make_scorer(balanced_accuracy_score)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
    )


def _reset_directory(directory):
    if os.path.exists(directory):
        for f in os.listdir(directory):
            os.remove(os.path.join(directory, f))
        os.rmdir(directory)
    os.makedirs(directory)


def save_model(path, model):
    _reset_directory("files/models/")
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def compute_metrics(split_name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": split_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion_matrix(split_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": split_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


raw_train, raw_test = load_data()
train_df = clean_data(raw_train)
test_df = clean_data(raw_test)

X_train, y_train = split_features_target(train_df)
X_test, y_test = split_features_target(test_df)

pipeline = build_pipeline(X_train)
model = build_grid_search(pipeline, X_train)
model.fit(X_train, y_train)

save_model(os.path.join("files/models/", "model.pkl.gz"), model)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_metrics = compute_metrics("train", y_train, y_train_pred)
test_metrics = compute_metrics("test", y_test, y_test_pred)
train_cm = compute_confusion_matrix("train", y_train, y_train_pred)
test_cm = compute_confusion_matrix("test", y_test, y_test_pred)

os.makedirs("files/output/", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_metrics) + "\n")
    out_file.write(json.dumps(test_metrics) + "\n")
    out_file.write(json.dumps(train_cm) + "\n")
    out_file.write(json.dumps(test_cm) + "\n")

