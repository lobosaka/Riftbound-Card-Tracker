from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from OCR.main import main
from card_data import (
    change_inventory as change_inventory_in_db,
    load_all_cards as load_all_cards_from_db,
    load_collection as load_collection_from_db,
    load_collection_statistics as load_collection_statistics_from_db,
    load_missing_cards as load_missing_cards_from_db,
    update_inventory as update_inventory_in_db,
)


app = FastAPI()
OCR_IMAGE_PATH = Path(__file__).resolve().parent / "OCR/images/1.jpg"


class InventoryUpdate(BaseModel):
    new_quantity: int


class InventoryChange(BaseModel):
    difference: int


@app.get("/ocr")
def call_main():
    image_path = str(OCR_IMAGE_PATH)
    code, candidates = main(image_path)

    return {
        "image_path": image_path,
        "code": code,
        "candidates": [
            {"source": candidate.source, "code": candidate.code}
            for candidate in candidates
        ],
    }


@app.get("/cards")
def load_all_cards():
    return load_all_cards_from_db()


@app.get("/cards/collection")
def load_collection():
    return load_collection_from_db()


@app.get("/cards/missing")
def load_missing_cards():
    return load_missing_cards_from_db()


@app.get("/cards/statistics")
def load_collection_statistics():
    return load_collection_statistics_from_db()


@app.put("/cards/{card_id}/inventory")
def update_inventory(card_id: str, payload: InventoryUpdate):
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

    return {
        "card_id": card_id,
        "inventory_count": payload.new_quantity,
    }


@app.post("/cards/{card_id}/inventory/change")
def change_inventory(card_id: str, payload: InventoryChange):
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

    uvicorn.run(app, host="127.0.0.1", port=8000)
