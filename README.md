# Online Fraud Detection Enterprise Dashboard

Streamlit dashboard plus FastAPI backend for a demo fraud detection workflow. The backend serves scoring, metrics, and simulated stream data; the dashboard consumes those endpoints and also reads local sample CSVs.

## Project layout
- `dashboard/app.py` — primary Streamlit entrypoint (multi-page UI under `dashboard/pages`).
- `backend/api/main.py` — FastAPI app served by uvicorn.
- `fraud_lab/dashboard_app.py` — standalone Streamlit experience used for experimentation.
- `shared/` — styling and data loading helpers shared across apps.
- `data/` — sample dataset (`transactions_sample.csv`), audit log, and config files. Place your own Kaggle CSV at `data/kaggle/online_fraud.csv` to override the demo data.

## Prerequisites
- Python 3.10+ on Windows
- PowerShell

## One-step launch (dashboard + API)
From the project root:

```
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

What the script does:
- Creates `.venv` if missing and activates it.
- Installs `requirements.txt` on first run of the venv.
- Starts FastAPI at `http://127.0.0.1:8000`.
- Starts Streamlit dashboard at `http://localhost:8502`.

## Notes
- If you prefer to run manually, activate `.venv`, install `requirements.txt`, then run `uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000` in one terminal and `streamlit run dashboard/app.py --server.port 8502` in another.
- The dashboard falls back to the bundled sample CSV if a Kaggle file is not present.
