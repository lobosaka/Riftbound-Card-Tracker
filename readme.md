# Riftbound Card Tracker

Local Riftbound card tracker with a `FastAPI` backend and `Streamlit` dashboard.

## Install

```bash
pip install -r requirements.txt
```

## Run

Start the API:

```bash
python src/api.py
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

The app uses `http://127.0.0.1:8000` for the API.

## API

OCR:

```bash
curl -X POST http://127.0.0.1:8000/predict/ocr \
  -F "file=@/absolute/path/to/card-image.jpg"
```

Classification:

```bash
curl -X POST http://127.0.0.1:8000/predict/classification \
  -F "file=@/absolute/path/to/card-image.jpg"
```

Representation learning:

```bash
curl -X POST http://127.0.0.1:8000/predict/representation-learning \
  -F "file=@/absolute/path/to/card-image.jpg"
```

All cards:

```bash
curl http://127.0.0.1:8000/cards
```

Collection stats:

```bash
curl "http://127.0.0.1:8000/cards?stats=true"
```

Update inventory:

```bash
curl -X PUT http://127.0.0.1:8000/cards/example-card-id/inventory \
  -H "Content-Type: application/json" \
  -d '{"difference": 1}'
```

Camera stream:

```bash
curl "http://127.0.0.1:8000/stream"
```

## Optional Data Scraping

```bash
python -m src.db.scraper
python -m src.db.picture_loader
```
