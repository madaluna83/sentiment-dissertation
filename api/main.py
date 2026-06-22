from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# initializare applicatie
app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for sentiment analysis using DistilBERT",
    version="1.0.0"
)


# incarca modelul
# MODEL_PATH = os.path.join(
#     os.path.dirname(__file__),
#     "..", "model", "model_disertatie_v2"
# )
MODEL_PATH = "madaluna83/sentiment-distilbert-imdb"

print(f"Se incarca modelul din: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model incarcat cu succes!")

device = torch.device("cpu")
model  = model.to(device)


# model de date(Pydantic)
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text:       str
    sentiment:  str
    confidence: float
    label:      int


# endpoints
@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health",
            "docs":    "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "DistilBERT fine-tuned pe IMDb",
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    # Tokenizare
    tokens = tokenizer(
        input.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Muta pe device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Inferenta
    with torch.no_grad():
        outputs = model(**tokens)
        logits  = outputs.logits

    # Calculeaza probabilitatile
    probs      = torch.softmax(logits, dim=-1)
    label      = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][label].item()

    sentiment = "POZITIV" if label == 1 else "NEGATIV"

    return PredictionOutput(
        text=input.text,
        sentiment=sentiment,
        confidence=round(confidence, 4),
        label=label
    )

@app.post("/predict/batch")
def predict_batch(inputs: list[TextInput]):
    results = []
    for item in inputs:
        result = predict(item)
        results.append(result)
    return results