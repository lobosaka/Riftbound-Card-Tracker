# Riftbound Card Tracker

## API

Run the API with:

```bash
python3 src/api
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
- the default sample image when `image_path` is omitted

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{}'
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

### `GET /stream`

```bash
curl "http://127.0.0.1:8000/stream"
```

Streams the backend camera as MJPEG. Optional query params:
- `camera_device_index`
- `camera_device_candidates` as comma-separated list, for example `1,0`
- `camera_width`
- `camera_height`
- `camera_warmup_frames`
- `fps`
- `jpeg_quality`

## Streamlit

Run the API first, then start the dashboard with:

```bash
streamlit run dashboard/app.py
```

The app reads from the API at `http://127.0.0.1:8000` and provides the overview,
collection, missing-cards, and all-cards pages.
Camera capture for OCR is handled in the Streamlit app via the browser camera API and
then sent to `POST /ocr` as an image path.
Switch to 'src/db/'. Execute scraper.py. A SQLite Database will be initialized, card 
data will be scraped and written into the DB. Second the picture_loader.py downloads 
the pictures from the URLs which are in the DB under the column 'image' on your local
machine. Then in 'src/trans/' execute transform.py. This script converts and resizes
the pictures. This is mandatory for future use in CNNs.

The commands for the terminal:

1. python -m src.db.scraper (optional)
2. python -m src.db.picture_loader
3. python -m src.trans.transform
