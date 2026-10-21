"""
TruthLens - FastAPI Server
--------------------------
Serves the TruthLens tiered classifier for the browser extension.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from backend.classifier import TruthLensClassifier

app = FastAPI(
    title="TruthLens Misinformation Detection API",
    description="Tiered multi-level classification engine for real-time claim verification.",
    version="1.0.0"
)

# CORS middleware for browser extension and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.google.com","http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True
)

classifier = TruthLensClassifier(model_dir="models")
start_time = time.time()

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Sentence or statement to verify", min_length=3)
    sensitivity: Optional[str] = Field("medium", description="Sensitivity level: low, medium, or high")

class BatchClassifyRequest(BaseModel):
    claims: List[str] = Field(..., description="List of sentence claims to verify")
    sensitivity: Optional[str] = Field("medium", description="Sensitivity level: low, medium, or high")

class ClassificationResponse(BaseModel):
    claim: str
    tier: str  # verified | uncertain | likely-false
    confidence: float
    reason: str
    model_tier: str
    sensitivity: str
    latency_ms: float

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    total_claims: int
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    active_model_tier: str
    uptime_seconds: float

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "active_model_tier": classifier.model_tier,
        "uptime_seconds": round(time.time() - start_time, 2)
    }

@app.post("/api/v1/classify", response_model=ClassificationResponse)
def classify_single(req: ClassifyRequest):
    t0 = time.perf_counter()
    res = classifier.classify(req.text, sensitivity=req.sensitivity)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    res["latency_ms"] = latency_ms
    return res

@app.post("/api/v1/classify-batch", response_model=BatchClassificationResponse)
def classify_batch(req: BatchClassifyRequest):
    t0 = time.perf_counter()
    results = []
    for claim in req.claims:
        ct0 = time.perf_counter()
        item = classifier.classify(claim, sensitivity=req.sensitivity)
        item["latency_ms"] = round((time.perf_counter() - ct0) * 1000, 2)
        results.append(item)
    
    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "results": results,
        "total_claims": len(results),
        "total_latency_ms": total_latency_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
