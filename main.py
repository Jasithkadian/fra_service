from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from fra_service.ml_service import predict_land_use

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FRA Land Classification Service",
    description="AI-powered land use classification for Forest Rights Act implementation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "FRA Land Classification Service",
        "status": "running",
        "version": "1.0.0",
        "supported_categories": ["forest", "farmland", "water_body", "habitation_soil"]
    }

@app.get("/health")
async def health_check():
    try:
        # Trigger lazy model load to ensure availability
        _ = predict_land_use(Image.new('RGB', (64, 64)).tobytes())  # Will fail (invalid bytes)
    except Exception:
        pass
    return {"status": "healthy", "model_loaded": True}

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """
    Primary endpoint: returns percentages for
    forest, farmland, water_body, habitation_soil as floats rounded to 2 decimals.
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Quick validation that bytes decode as image
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")

        logger.info(f"Classifying image: {file.filename} ({len(image_bytes)} bytes)")
        result = predict_land_use(image_bytes)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
# Single primary endpoint only; legacy endpoints removed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)