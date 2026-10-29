import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(
    title="Production-Ready AI Sentiment Analysis API",
    description="Asynchronous backend service utilizing Hugging Face transformers.",
    version="1.0.0"
)

# Initialize the Hugging Face pipeline globally
# For production efficiency, this loads into memory once when the server starts
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    raise RuntimeError(f"Failed to load Hugging Face model: {str(e)}")

# Define input validation schema using Pydantic
class SentimentRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="The raw text payload to analyze for emotional sentiment."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Classroom theory is great, but production code hits differently!"
            }
        }

# Define output validation schema
class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint to confirm backend readiness."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/api/v1/analyze", response_model=SentimentResponse, tags=["AI Inference"])
async def analyze_sentiment(payload: SentimentRequest):
    """
    Parses input text data and runs machine learning sentiment analysis.
    Uses asyncio.to_thread to offload the heavy synchronous ML pipeline execution
    to a separate worker thread, ensuring the main FastAPI event loop remains unblocked.
    """
    try:
        # Offload the blocking model inference to a worker thread
        prediction = await asyncio.to_thread(sentiment_pipeline, payload.text)
        
        # Parse the raw results out of the pipeline response list
        result = prediction[0]
        
        return SentimentResponse(
            text=payload.text,
            label=result["label"],
            score=round(result["score"], 4)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference error during model execution: {str(e)}"
        )
