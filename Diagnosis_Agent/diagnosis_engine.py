from .rules import rule_based_diagnosis
from .fallback import ml_diagnosis

def faiss_diagnosis(latent_vector, top_k=3):
    import faiss, joblib, numpy as np

    index = faiss.read_index("models/faiss.index")
    meta = joblib.load("models/faiss_meta.pkl")

    q = np.array(latent_vector, dtype="float32").reshape(1, -1)
    distances, indices = index.search(q, top_k)

    matches = meta.iloc[indices[0]]

    diagnosis = matches["diagnosis"].mode()[0]
    confidence = 1 / (1 + distances[0][0])

    return {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "explanation": f"Similar historical anomaly found ({top_k} neighbors)"
    }


def diagnose(sensor_row, latent_vector, recon_error):
    result = rule_based_diagnosis(sensor_row)
    if result:
        return {**result, "source": "rules"}
    
    if recon_error > 0.001:
        result = faiss_diagnosis(latent_vector)
        if result["confidence"] > 0.6:
            result["source"] = "faiss"
            return result

    result = ml_diagnosis(latent_vector,recon_error)
    result["source"] = "ml"
    return result
