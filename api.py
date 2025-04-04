from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from inference import generate_responses
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class Question(BaseModel):
    text: str

@app.post("/ask")
async def get_llm_response(question: Question):
    result = generate_responses(question.text)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)