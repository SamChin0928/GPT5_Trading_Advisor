# Objective

Trade based on decisions backed by historical training. Data for decisions.

Train a personal ML Model to aid trading.


## Usage

### Backend:
```
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ..
uvicorn backend.main:app --reload --port 8000
```

### Frontend:
```
cd frontend
npm install
npm run dev
```
