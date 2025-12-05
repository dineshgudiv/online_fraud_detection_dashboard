# Online Fraud Detection Dashboard

**Online Fraud Detection Dashboard** is an end-to-end fraud analytics project built with Python, Streamlit, and machine-learning models.  
It provides an “executive command center” style UI to explore transactions, run real-time scoring, and analyze model + rule performance.

> ⚠️ Note: Large AIML dataset files are **not included** in this repo because of GitHub’s 100 MB limit.  
> Place your own dataset in the indicated folders to run full experiments.

---

## 🔍 Key Features

- **Interactive Streamlit dashboard**
  - Executive overview of fraud risk
  - Live / batch scoring views
  - Drill-down into suspicious transactions
- **ML + rules hybrid approach**
  - Model-based risk score for each transaction
  - Rule engine for threshold / business rules
- **Data pipeline utilities**
  - Data loading, preprocessing, feature creation
  - Configurable dataset selection (`dataset_config.json`)
- **Reusable backend code**
  - Shared modules for styling, data loading, and pipelines
  - Ready to connect with a FastAPI / REST scoring service

---

## 🧱 Tech Stack

- **Language:** Python 3.x  
- **Frontend:** Streamlit dashboard  
- **Data & ML:** pandas, numpy, scikit-learn (and others as listed in `requirements.txt`)  
- **OS / Dev:** Windows + PowerShell (but can run on any OS with Python)

---

## 📂 Project Structure

```text
online_fraud_detection_dashboard/
│
├─ .streamlit/           # Streamlit config (e.g., upload size)
├─ backend/              # (Optional) backend / API-related code
├─ dashboard/            # Extra dashboard utilities (if any)
├─ data/                 # Sample / small data files, configs
├─ fraud_lab/            # Main fraud detection logic + Streamlit app
├─ shared/               # Shared helpers (data loader, styling, etc.)
│
├─ README.md             # You are here
├─ requirements.txt      # Python dependencies
└─ run_all.ps1           # Convenience script to run the app on Windows
