# Backend – FastAPI

## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

Data is stored under `backend/data/sessions/<session_id>/`.