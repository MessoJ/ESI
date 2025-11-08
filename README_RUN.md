# Run ESI (minimal)

## Prereqs
- Python 3.11+
- Node.js 20+

## Backend
```
pip install -r requirements.txt
# Windows PowerShell
$env:FRED_API_KEY="YOUR_KEY"
uvicorn api.main:app --reload --port 8000
```

## Frontend
```
cd ui
npm i
npm run dev
```

Open http://localhost:3000

API endpoints:
- GET /healthz
- GET /api/index/latest
- GET /api/index/history
- GET /api/index/components


