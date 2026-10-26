import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import pipeline

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Async AI Sentiment Analysis API",
    description="A production-ready FastAPI wrapper around Hugging Face Transformers.",
    version="1.0.0"
)

# 2. Load the ML Model locally on startup (Using a fast, lightweight DistilBERT model)
print("Loading Machine Learning model pipeline...")
try:
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sentiment_model = None

# 3. Define the strict Data Input Schema using Pydantic
class AnalysisRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3, 
        max_length=1000, 
        description="The raw text content to analyze for sentiment.",
        example="FastAPI and Machine Learning integration works beautifully!"
    )

# 4. Define the Data Output Schema for clean API documentation
class AnalysisResponse(BaseModel):
    status: str
    text_analyzed: str
    sentiment: str
    confidence_score: float

# 5. Create the Core API Endpoints
@app.get("/", tags=["Health Check"])
async def root():
    """Simple health-check endpoint to verify backend status."""
    return {"status": "online", "message": "AI Sentiment Backend is active."}

@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["ML Inference"])
async def analyze_sentiment(payload: AnalysisRequest):
    """
    Accepts text payload, feeds it directly to the local ML pipeline,
    and returns a structured json analysis.
    """
    if not sentiment_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Model pipeline is currently unavailable on the host system."
        )
    
    try:
        # Run inference through the pipeline
        model_output = sentiment_model(payload.text)[0]
        
        return AnalysisResponse(
            status="success",
            text_analyzed=payload.text,
            sentiment=model_output["label"],
            confidence_score=round(float(model_output["score"]), 4)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during model inference: {str(e)}"
        )

# 6. Local Server Execution block
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
