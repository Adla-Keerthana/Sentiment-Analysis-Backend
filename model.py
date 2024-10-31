# model.py
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the pre-trained model
model = load_model("sentiment_analysis_model.h5")

# Load the tokenizer
with open("tokenizer.json") as f:
    tokenizer_json = f.read()  # Read the content of the file as a string
tokenizer = tokenizer_from_json(tokenizer_json)  # Pass the JSON string to the function

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment, float(prediction[0][0])
