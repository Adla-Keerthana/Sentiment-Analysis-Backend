# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import predict_sentiment
from pydantic import BaseModel

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; customize for specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected request model
class Review(BaseModel):
    review: str

@app.post("/predict/")
async def predict(review: Review):
    sentiment, score = predict_sentiment(review.review)
    return {"sentiment": sentiment, "score": score}
