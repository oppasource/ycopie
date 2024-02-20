import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from gen_ai_helper import get_response_to_prompt

app = FastAPI()

# Define CORS settings
origins = ["*"]  # Replace with the origin of your client-side application

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=True,  # If you need to allow credentials (e.g., cookies)
)

@app.post("/ask")
async def generate_response_api(info : Request):
    req_info = await info.json()
    generated_response = get_response_to_prompt(req_info.get('prompt'))
    return {"answer": generated_response}


@app.post("/test")
async def test_api(info : Request):
    req_info = await info.json()
    generated_response = req_info.get('prompt')
    return {"generated_response": generated_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)