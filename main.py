from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import os  # <-- ✅ Required to read environment variables

app = FastAPI()

# Serve static frontend from the "static" folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Optional: tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the API key securely from environment variable
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ENDPOINT = "https://detect.roboflow.com/my-first-project-lodrg/2"

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        files = {"file": (image.filename, await image.read(), image.content_type)}
        response = requests.post(
            f"{MODEL_ENDPOINT}?api_key={ROBOFLOW_API_KEY}",
            files=files
        )

        print("Roboflow response:", response.status_code, response.text)

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Detection failed"})

        data = response.json()
        count = len(data.get("predictions", []))
        return {
            "count": count,
            "predictions": data.get("predictions", [])
        }

    except Exception as e:
        print("Backend error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
