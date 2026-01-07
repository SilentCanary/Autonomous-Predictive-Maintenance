# Autonomous-Predictive-Maintenance
# Vehicle Predictive Maintenance & Customer Interaction System

## Overview

This project implements an end‑to‑end **vehicle predictive maintenance and customer interaction system**. It combines three intelligent agents working in a pipeline:

- **Prediction Agent** – Processes real‑time vehicle sensor data to detect anomalies.  
- **Diagnosis Agent** – Analyzes anomalies using ML models + rule‑based logic to determine the likely issue.  
- **Customer Interaction Agent** – Interacts with the vehicle owner in natural Hinglish, explains the issue, and schedules service appointments.

The architecture is **modular**, **node‑based**, and built using a **LangGraph-style pipeline**. Future agents like Scheduling, Insights, and Feedback can be plugged in easily.

---

## Features

### ✅ Predictive Analysis
- Autoencoder-based latent representation of sensor data  
- Reconstruction error to detect anomalies  

### ✅ Explainable Diagnosis
- Rule-based domain logic  
- XGBoost fallback model  
- Optional FAISS-based RAG for contextual support  
- Outputs structured JSON with:
  - `diagnosis`
  - `confidence`
  - `explanation`
  - `source`

### ✅ Human-like Customer Interaction
- LLM-powered Hinglish conversation  
- Maintains conversational memory  
- Supports MCP tool calls:
  - `find_center`
  - `normalise_date`
  - `book_slot`

### ✅ Simulation Ready
- Console-based interaction  
- Can later integrate with Twilio, WhatsApp, or voice interfaces  

### ✅ Security & Compliance (Future)
- UEBA for agent behavior monitoring  
- Semantic Kernel for RBAC  
- JSON-based structured inter-agent communication (MCP / A2A)

---

## Folder Structure
```
├── CustomerInteraction/
│   ├── data/
│   └── tools/
│       ├── agent3.py
│       ├── customer_node.py
│       ├── mcp_client.py
│       ├── node_base.py
│       └── test_client.py
│
├── Diagnosis_Agent/
│   ├── models/
│   │   ├── diagnosis_label_encoder.py
│   │   ├── diagnosis_node.py
│   │   ├── faiss.index
│   │   ├── faiss_meta.pkl
│   │   ├── node_base.py
│   │   ├── xgb_diagnosis.pkl
│   │   └── ...
│   └── tools/
│       ├── diagnosis_engine.py
│       ├── fallback.py
│       ├── generate_initial_csv.py
│       └── train_xg.py
│
├── Prediction_Agent/
│   ├── model/
│   │   └── models/
│   │       ├── autoencoder.pt
│   │       ├── scaler.pkl
│   │       ├── xgb_model.pkl
│   │       ├── classification_report.txt
│   │       ├── __init__.py
│   │       ├── node_base.py
│   │       ├── prediction_tool.py
│   │       ├── predictive_node.py
│   │       └── predictive_tool.py
│   └── train.py
│
├── insights_agent/
│   └── ... (future analytics agent)
│
├── schedulingAgent/
│   └── ... (future scheduling agent)
│
├── Langraph_master.py
├── sensor_simulator.py
└── requirements.txt
```
--

## Setup

### 1. Clone the repository
```bash
git clone <repo_url>
```

### 2. Environment Variables
Create a .env file:
GROQ_API_KEY=<your_groq_api_key>

### 3. Run Pipeline
python Langraph_master.py

# Future Agents
- Scheduling Agent – Auto-booking based on user preferences
- Insights Agent – Failure analytics + user behavior
- Feedback Agent – Post-service feedback collection

# Tech Stack
- Python 3.10+
- NumPy, Pandas
- scikit-learn, XGBoost
- PyTorch (autoencoder)
- FAISS (RAG)
- Groq (LLM inference)
- MCPClient (tool calling)
- LangGraph (node orchestration)
- asyncio

# Author
Advitiya Prakash



