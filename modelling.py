import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Baca data (pastikan file CSV ada di direktori yang sama saat di-push ke GitHub)
df = pd.read_csv("heart_disease_cleaned.csv")
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Log model dengan nama folder "model"
    mlflow.sklearn.log_model(model, "model")
    print(f"Model berhasil dilatih dan disimpan. Run ID: {run.info.run_id}")