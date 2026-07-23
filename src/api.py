from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from OCR.main import main, process_image_with_debug
from services.camera_capture import CameraCaptureError, CameraCaptureService
from services.card_data import (
    change_inventory as change_inventory_in_db,
    load_all_cards as load_all_cards_from_db,
    load_collection as load_collection_from_db,
    load_collection_statistics as load_collection_statistics_from_db,
    load_missing_cards as load_missing_cards_from_db,
    update_inventory as update_inventory_in_db,
)


logger = logging.getLogger(__name__)
app = FastAPI()
OCR_IMAGE_PATH = Path(__file__).resolve().parent / "OCR/images/1.jpg"
CAPTURE_IMAGE_DIR = Path(__file__).resolve().parent / "OCR/images/captures"
camera_capture_service = CameraCaptureService(CAPTURE_IMAGE_DIR)


class InventoryUpdate(BaseModel):
    new_quantity: int | None = None
    difference: int | None = None


class OcrRequest(BaseModel):
    image_path: str | None = None
    source: str | None = None
    filename: str | None = None
    camera_device_index: int = 0
    camera_device_candidates: list[int] | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    camera_warmup_frames: int = 10
    debug: bool = False


class CameraProbeRequest(BaseModel):
    device_indices: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    camera_width: int | None = None
    camera_height: int | None = None
    camera_warmup_frames: int = 5


def capture_image_for_source(
    source: str,
    filename: str | None = None,
    camera_device_index: int = 0,
    camera_device_candidates: list[int] | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_warmup_frames: int = 10,
) -> Path:
    if source == "camera":
        try:
            device_indices = (
                camera_device_candidates
                if camera_device_candidates is not None
                else [camera_device_index]
            )
            logger.info(
                "Capturing image from camera device_indices=%s width=%s height=%s warmup_frames=%s",
                device_indices,
                camera_width,
                camera_height,
                camera_warmup_frames,
            )
            return camera_capture_service.capture_image_from_candidates(
                device_indices=device_indices,
                width=camera_width,
                height=camera_height,
                warmup_frames=camera_warmup_frames,
                filename=filename,
            )
        except (CameraCaptureError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    raise HTTPException(
        status_code=400,
        detail="Invalid OCR source. Use 'camera'.",
    )


def get_latest_captured_image() -> Path:
    image_files = [
        image_file
        for image_file in CAPTURE_IMAGE_DIR.iterdir()
        if image_file.is_file()
    ]
    if not image_files:
        raise FileNotFoundError(
            "No captured images found. Capture an image first or provide image_path explicitly."
        )
    return max(image_files, key=lambda image_file: image_file.stat().st_mtime)


def resolve_ocr_image_path(image_path: str | None) -> Path:
    if image_path is not None:
        resolved_path = Path(image_path).expanduser().resolve()
    else:
        try:
            resolved_path = get_latest_captured_image()
        except FileNotFoundError:
            resolved_path = OCR_IMAGE_PATH

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved_path}")

    return resolved_path


def run_ocr_for_image(image_path: Path, debug: bool = False):
    logger.info("Starting OCR for image: %s", image_path)

    try:
        if debug:
            result = process_image_with_debug(str(image_path))
        else:
            code, candidates = main(str(image_path))
            result = {
                "image_path": str(image_path),
                "code": code,
                "candidates": [
                    {"source": candidate.source, "code": candidate.code}
                    for candidate in candidates
                ],
            }
    except Exception as error:
        logger.exception("OCR failed for image: %s", image_path)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "OCR processing failed.",
                "image_path": str(image_path),
                "error": str(error),
            },
        ) from error

    logger.info(
        "Finished OCR for image: %s with %s candidates",
        image_path,
        len(result["candidates"]),
    )
    return result


@app.post("/ocr")
def run_ocr(payload: OcrRequest):
    if payload.image_path is not None and payload.source is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either image_path or source, not both.",
        )

    try:
        if payload.source is not None:
            image_path = capture_image_for_source(
                source=payload.source,
                filename=payload.filename,
                camera_device_index=payload.camera_device_index,
                camera_device_candidates=payload.camera_device_candidates,
                camera_width=payload.camera_width,
                camera_height=payload.camera_height,
                camera_warmup_frames=payload.camera_warmup_frames,
            )
        else:
            image_path = resolve_ocr_image_path(payload.image_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return run_ocr_for_image(image_path, debug=payload.debug)


@app.post("/camera/probe")
def probe_cameras(payload: CameraProbeRequest):
    return {
        "results": camera_capture_service.probe_devices(
            device_indices=payload.device_indices,
            width=payload.camera_width,
            height=payload.camera_height,
            warmup_frames=payload.camera_warmup_frames,
        )
    }


@app.get("/camera/stream")
def stream_camera(
    camera_device_index: int = 0,
    camera_device_candidates: str | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_warmup_frames: int = 10,
    fps: float = 10.0,
    jpeg_quality: int = 85,
):
    try:
        device_indices = None
        if camera_device_candidates:
            device_indices = [
                int(candidate.strip())
                for candidate in camera_device_candidates.split(",")
                if candidate.strip()
            ]

        stream = camera_capture_service.mjpeg_stream(
            device_index=camera_device_index,
            device_indices=device_indices,
            width=camera_width,
            height=camera_height,
            warmup_frames=camera_warmup_frames,
            fps=fps,
            jpeg_quality=jpeg_quality,
        )
    except (CameraCaptureError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return StreamingResponse(
        stream,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/cards")
def load_all_cards(
    inventory: str | None = None,
    stats: bool = False,
):
    if stats:
        if inventory is not None:
            raise HTTPException(
                status_code=400,
                detail="Statistics cannot be combined with inventory filters.",
            )
        return load_collection_statistics_from_db()

    if inventory is None:
        return load_all_cards_from_db()
    if inventory == "collection":
        return load_collection_from_db()
    if inventory == "missing":
        return load_missing_cards_from_db()
    raise HTTPException(
        status_code=400,
        detail="Invalid inventory filter. Use 'collection' or 'missing'.",
    )
@app.put("/cards/{card_id}/inventory")
def update_inventory(card_id: str, payload: InventoryUpdate):
    if payload.new_quantity is not None and payload.difference is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either new_quantity or difference, not both.",
        )

    if payload.new_quantity is None and payload.difference is None:
        raise HTTPException(
            status_code=400,
            detail="Provide new_quantity or difference.",
        )

    if payload.new_quantity is not None:
        if payload.new_quantity < 0:
            raise HTTPException(
                status_code=400,
                detail="Der Bestand darf nicht negativ sein.",
            )

        rowcount = update_inventory_in_db(card_id, payload.new_quantity)
        if rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Es wurde keine Karte mit der ID {card_id!r} gefunden."
                ),
            )

        inventory_count = payload.new_quantity
    else:
        inventory_count = change_inventory_in_db(
            card_id,
            payload.difference,
        )

    if inventory_count is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Es wurde keine Karte mit der ID {card_id!r} gefunden."
            ),
        )

    return {
        "card_id": card_id,
        "inventory_count": inventory_count,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
