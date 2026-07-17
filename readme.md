# Riftbound Card Tracker

## API

Run the API with:

```bash
python3 -m src.api
```

### `POST /ocr`

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "src/OCR/images/captures/card-shot.jpg"
  }'
```

Runs OCR on either:
- an existing image via `image_path`
- a camera capture via `source: "camera"`
- a screen capture via `source: "screen"`

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source": "camera"
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source": "screen",
    "filename": "screen-card.png"
  }'
```

### `GET /cards`

```bash
curl http://127.0.0.1:8000/cards
```

Returns cards. Optional query params:
- `inventory=collection`
- `inventory=missing`
- `stats=true`

```bash
curl "http://127.0.0.1:8000/cards?inventory=collection"
```

```bash
curl "http://127.0.0.1:8000/cards?stats=true"
```

### `PUT /cards/{card_id}/inventory`

```bash
curl -X PUT http://127.0.0.1:8000/cards/example-card-id/inventory \
  -H "Content-Type: application/json" \
  -d '{
    "new_quantity": 3
  }'
```

Updates inventory either absolutely with `new_quantity` or relatively with `difference`.

```bash
curl -X PUT http://127.0.0.1:8000/cards/example-card-id/inventory \
  -H "Content-Type: application/json" \
  -d '{
    "difference": 1
  }'
```

## Streamlit

Run the API first, then start the dashboard with:

```bash
streamlit run dashboard/app.py
```

The app reads from the API at `http://127.0.0.1:8000` and provides the overview,
collection, missing-cards, and all-cards pages.
