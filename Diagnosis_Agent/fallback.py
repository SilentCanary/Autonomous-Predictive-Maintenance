import joblib
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_DIR = os.path.join(BASE_DIR, "models")
xgb = joblib.load(os.path.join(MODEL_DIR,"xgb_diagnosis.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "diagnosis_label_encoder.pkl"))
def ml_diagnosis(latent_vector,recon_error):
    feature_vector = np.array(latent_vector, dtype=np.float32)
    
    # Append reconstruction error
    feature_vector = np.append(feature_vector, recon_error)
    proba = xgb.predict_proba(feature_vector.reshape(1, -1))[0]
    cls = np.argmax(proba)
    diagnosis_name = label_encoder.inverse_transform([cls])[0]

    return {
        "diagnosis": diagnosis_name,
        "confidence": float(proba[cls]),
        "explanation": "ML-based diagnosis using learned patterns"
    }
