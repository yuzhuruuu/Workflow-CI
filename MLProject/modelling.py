"""
modelling.py (versi MLProject / CI)
Sama seperti Membangun_model/modelling.py, tapi tanpa hardcode tracking URI ke
127.0.0.1:5000 karena akan dijalankan otomatis oleh GitHub Actions runner
(tidak ada MLflow UI server yang hidup di sana). MLflow akan mencatat run
ke folder lokal ./mlruns pada runner, yang kemudian diupload sebagai artefak
workflow.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart_preprocessing.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog()

# `mlflow run` (dipanggil dari MLflow Project) sudah membuka run aktif untuk kita,
# jadi cukup pakai mlflow.start_run() tanpa argumen supaya menempel ke run tsb,
# alih-alih membuat experiment/run baru yang bisa bentrok.
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test Accuracy: {acc:.4f}")
