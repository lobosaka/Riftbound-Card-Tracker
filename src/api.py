from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from OCR.main import process_image, process_image_with_debug
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
camera_capture_service = CameraCaptureService()
CARD_NOT_FOUND_DETAIL = "Es wurde keine Karte mit der ID {card_id!r} gefunden."


class InventoryUpdate(BaseModel):
    new_quantity: int | None = None
    difference: int | None = None


class OcrRequest(BaseModel):
    image_path: str | None = None
    debug: bool = False


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def get_default_ocr_image_path() -> Path:
    return OCR_IMAGE_PATH


def resolve_ocr_image_path(image_path: str | None) -> Path:
    resolved_path = (
        Path(image_path).expanduser().resolve()
        if image_path is not None
        else get_default_ocr_image_path()
    )

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved_path}")

    return resolved_path


def run_ocr_for_image(image_path: Path, debug: bool = False):
    logger.info("Starting OCR for image: %s", image_path)

    try:
        if debug:
            result = process_image_with_debug(str(image_path))
        else:
            code, candidates = process_image(str(image_path), emit_result=False)
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
    try:
        image_path = resolve_ocr_image_path(payload.image_path)
    except FileNotFoundError as error:
        raise not_found(str(error)) from error

    return run_ocr_for_image(image_path, debug=payload.debug)


@app.get("/stream")
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
        raise bad_request(str(error)) from error

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
            raise bad_request("Statistics cannot be combined with inventory filters.")
        return load_collection_statistics_from_db()

    if inventory is None:
        return load_all_cards_from_db()
    if inventory == "collection":
        return load_collection_from_db()
    if inventory == "missing":
        return load_missing_cards_from_db()
    raise bad_request("Invalid inventory filter. Use 'collection' or 'missing'.")


@app.put("/cards/{card_id}/inventory")
def update_inventory(card_id: str, payload: InventoryUpdate):
    if payload.new_quantity is not None and payload.difference is not None:
        raise bad_request("Provide either new_quantity or difference, not both.")

    if payload.new_quantity is None and payload.difference is None:
        raise bad_request("Provide new_quantity or difference.")

    if payload.new_quantity is not None:
        if payload.new_quantity < 0:
            raise bad_request("Der Bestand darf nicht negativ sein.")

        rowcount = update_inventory_in_db(card_id, payload.new_quantity)
        if rowcount == 0:
            raise not_found(CARD_NOT_FOUND_DETAIL.format(card_id=card_id))

        inventory_count = payload.new_quantity
    else:
        inventory_count = change_inventory_in_db(
            card_id,
            payload.difference,
        )

    if inventory_count is None:
        raise not_found(CARD_NOT_FOUND_DETAIL.format(card_id=card_id))

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
