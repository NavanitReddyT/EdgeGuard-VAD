from pydantic import BaseModel, Field

class ModeRequest(BaseModel):
    mode: str = Field(..., description="The desired mode: 'webcam' or 'dataset'")

class ThresholdRequest(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Anomaly detection threshold")

class MetricsResponse(BaseModel):
    fps: float
    cpu: float
    ram: float
    score: float
    anomaly: bool
    frame_count: int
