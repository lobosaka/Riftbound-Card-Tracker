import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.classification.inference.inference import predict_classification

app = FastAPI(title="Card Recognition API")


def parse_upload_image(contents: bytes) -> Image.Image:
  try:
    img = Image.open(io.BytesIO(contents))
    img.load()
    return img
  except Exception as e:
    raise HTTPException(status_code=400, detail="Invalid image file uploaded.") from e


@app.post("/predict/classification")
async def predict_class(file: UploadFile = File(...)):
  contents = await file.read()
  image = parse_upload_image(contents)
  return predict_classification(image)


@app.post("/predict/representation-learning")
async def predict_rl(file: UploadFile = File(...)):
  from src.representation_learning.inference.inference import (
      predict_representation_learning,
  )

  contents = await file.read()
  image = parse_upload_image(contents)
  return predict_representation_learning(image)