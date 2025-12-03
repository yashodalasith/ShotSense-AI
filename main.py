from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

from models.schemas import HealthResponse
from routers import batting as batting_router


app = FastAPI(
    title="ShotSense API",
    description="AI-Based Cricket Training App API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batting_router.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to ShotSense API",
        "version": "1.0.0",
        "endpoints": {
            "batting": "/batting",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=time.time())

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
