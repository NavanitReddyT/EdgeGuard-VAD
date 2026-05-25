import asyncio
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import PipelineManager
from .stream import generate_frames
from .schemas import ModeRequest, ThresholdRequest, MetricsResponse

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instance of the pipeline manager
manager = PipelineManager()

@app.on_event("startup")
async def startup_event():
    if not os.path.exists("results"):
        os.makedirs("results")
    logging.info("API ready.")

@app.post("/start")
async def start_pipeline():
    if manager.is_running():
        return {"status": "already running"}
    manager.start()
    return {"status": "started", "mode": manager.mode}

@app.post("/stop")
async def stop_pipeline():
    manager.stop()
    return {"status": "stopped"}

@app.post("/mode")
async def set_mode(request: ModeRequest):
    if manager.is_running():
        manager.stop()
        manager.set_mode(request.mode)
        manager.start()
    else:
        manager.set_mode(request.mode)
    return {"mode": request.mode, "status": "switched"}

@app.post("/threshold")
async def set_threshold(request: ThresholdRequest):
    if not (0.0 <= request.value <= 1.0):
        raise HTTPException(status_code=422, detail="Threshold must be between 0.0 and 1.0")
    manager.set_threshold(request.value)
    return {"threshold": request.value}

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return manager.get_metrics()

@app.get("/status")
async def get_status():
    return {
        "running": manager.is_running(),
        "mode": manager.mode,
        "threshold": manager.threshold
    }

@app.get("/video")
async def video_feed():
    return StreamingResponse(generate_frames(manager), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/report")
async def get_report():
    report = manager.benchmarker.generate_report(save_path="results/session_report.json")
    return JSONResponse(content=report)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.5)
            await websocket.send_json(manager.get_metrics())
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
