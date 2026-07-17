from fastapi import FastAPI
from OCR.main import main


app = FastAPI()

@app.get("/")
def call_main(image_path):
    code, candidates = main(image_path)

    return {
        "image_path": image_path,
        "code": code,
        "candidates": [
            {"source": candidate.source, "code": candidate.code}
            for candidate in candidates
        ],
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
