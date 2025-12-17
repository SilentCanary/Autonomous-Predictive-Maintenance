import faiss
import pandas as pd
import numpy as np
import joblib

CSV_PATH = "data/diagnosis_latent_dataset.csv"
INDEX_PATH = "models/faiss.index"

df = pd.read_csv(CSV_PATH)

latent_cols = [c for c in df.columns if c.startswith("z")]
X = df[latent_cols].values.astype("float32")

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

faiss.write_index(index, INDEX_PATH)
joblib.dump(df, "models/faiss_meta.pkl")

print("✅ FAISS index built")
