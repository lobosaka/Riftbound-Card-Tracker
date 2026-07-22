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
- a camera capture via `source: "camera"`

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source": "camera"
  }'
```

For an external camera on macOS, pass the camera index explicitly. The code now uses
OpenCV's AVFoundation backend on macOS by default, which is the correct backend for
USB/UVC webcams and document cameras.

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source": "camera",
    "camera_device_index": 1,
    "camera_width": 1920,
    "camera_height": 1080,
    "camera_warmup_frames": 15,
    "filename": "external-camera-shot.jpg"
  }'
```

If your iPhone is being selected first via Continuity Camera, probe several indices and
inspect the saved images. This writes `probe_camera_*.jpg` files into
`src/OCR/images/captures/`.

```bash
curl -X POST http://127.0.0.1:8000/camera/probe \
  -H "Content-Type: application/json" \
  -d '{
    "device_indices": [0, 1, 2, 3, 4],
    "camera_width": 1920,
    "camera_height": 1080
  }'
```

Then use the working camera in priority order:

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source": "camera",
    "camera_device_candidates": [2, 1, 0],
    "camera_width": 1920,
    "camera_height": 1080,
    "filename": "external-camera-shot.jpg"
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
