import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

CSV_PATH = "data/diagnosis_latent_dataset.csv"
MODEL_OUT = "models/xgb_diagnosis.pkl"
ENCODER_OUT = "models/diagnosis_label_encoder.pkl"

# ---------------- Load dataset ----------------
df = pd.read_csv(CSV_PATH)

latent_cols = [c for c in df.columns if c.startswith("z")]
X = df[latent_cols + ["reconstruction_error"]]
y_text = df["diagnosis"]

# ---------------- Encode labels ----------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
joblib.dump(label_encoder, ENCODER_OUT)


print("Label mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# ---------------- Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Train XGBoost ----------------
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = xgb.predict(X_test)
print("\n=== Diagnosis Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------- Save ----------------
joblib.dump(xgb, MODEL_OUT)
joblib.dump(label_encoder, ENCODER_OUT)

print(f"✅ Model saved: {MODEL_OUT}")
print(f"✅ Label encoder saved: {ENCODER_OUT}")
