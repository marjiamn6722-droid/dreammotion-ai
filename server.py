from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import shutil
import os

app = FastAPI(title="DreamMotion AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SPACE_ID = "cbensimon/wan2-2-fp8da-aoti-preview2"

@app.get("/")
def home():
    return {"message": "DreamMotion AI server is running"}

@app.post("/generate")
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    path = f"/tmp/{image.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        client = Client(SPACE_ID)
        result = client.predict(
            image=handle_file(path),
            prompt=prompt,
            api_name="/generate_video"
        )
        return {"result": result}

    except Exception as error:
        return {"error": str(error)}

    finally:
        if os.path.exists(path):
            os.remove(path)
