import os
import io
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


PROMPT = """
You are an expert document understanding assistant.

Rules:
- Answer ONLY the question.
- Return ONLY the answer.
- Never explain.
- Never include currency symbols.
- Never include units.
- Numeric answers should contain only the number.
- If the answer is not found return an empty string.
"""


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/answer-image")
def answer_image(data: ImageRequest):

    image_string = data.image_base64

    # Remove data:image/png;base64, prefix if present
    if "," in image_string:
        image_string = image_string.split(",")[1]
    
    image_string = image_string + "=" * (-len(image_string) % 4)
    
    image_bytes = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image_bytes))
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT,
            image,
            data.question
        ]
    )

    answer = response.text.strip()

    return {
        "answer": answer
    }

