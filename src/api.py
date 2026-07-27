import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from PIL import Image
from classification.inference import predict_classification
from services.logger import setup_logging

from ocr.run_ocr_pipeline import process_uploaded_image
from services.card_data import (
    change_inventory as change_inventory_in_db,
    load_all_cards as load_all_cards_from_db,
    load_collection as load_collection_from_db,
    load_collection_statistics as load_collection_statistics_from_db,
    load_missing_cards as load_missing_cards_from_db,
    update_inventory as update_inventory_in_db,
)


logger = logging.getLogger(__name__)
setup_logging(level=logging.DEBUG)
app = FastAPI(title="Card Recognition API")
CARD_NOT_FOUND_DETAIL = "Es wurde keine Karte mit der ID {card_id!r} gefunden."


class InventoryUpdate(BaseModel):
    new_quantity: int | None = None
    difference: int | None = None


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)

def parse_upload_image(contents: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
        logger.info(
            "Uploaded image parsed successfully with format '%s' and size %s.",
            getattr(img, "format", None),
            getattr(img, "size", None),
        )
        return img
    except Exception as error:
        logger.warning("Failed to parse uploaded image.", exc_info=error)
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.") from error


@app.post("/predict/classification")
async def predict_class(file: UploadFile = File(...)):
  contents = await file.read()
  image = parse_upload_image(contents)
  return predict_classification(image)


@app.post("/predict/representation-learning")
async def predict_rl(file: UploadFile = File(...)):
  from representation_learning.inference import (
      predict_representation_learning,
  )

  contents = await file.read()
  image = parse_upload_image(contents)
  return predict_representation_learning(image)

@app.post("/predict/ocr")
async def run_ocr(
    file: UploadFile = File(...),
):
    logger.info("Received OCR request for file '%s'.", file.filename)
    contents = await file.read()
    logger.debug(
        "Read %d bytes from uploaded file '%s'.",
        len(contents),
        file.filename,
    )
    image = parse_upload_image(contents)

    try:
        # Entry point for OCR processing
        response = process_uploaded_image(
            image=image,
            filename=file.filename,
            logger=logger,
        )
        logger.info(
            "OCR request for file '%s' completed with best code '%s'.",
            file.filename,
            response.get("code") or "",
        )
        return response
    except Exception as error:
        logger.exception("OCR failed for uploaded image: %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "OCR processing failed.",
                "image_path": file.filename,
                "error": str(error),
            },
        ) from error




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

    uvicorn.run(app, host="127.0.0.1", port=8000)
