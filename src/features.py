import numpy as np
from src.motion import compute_motion_magnitude

def extract_features(flow, fg_mask, frame_history):
    """
    Extracts a 1D feature vector from the given inputs.

    Features:
    1. Mean motion magnitude: Average pixel-level motion intensity.
    2. Direction change: Variance of flow angle to detect erratic movement.
    3. Object area variation: Change in foreground blob area between frames.
    4. Temporal instability: Standard deviation of last N frame differences.
    """
    # 1. Mean motion magnitude
    motion_magnitude = compute_motion_magnitude(flow)
    mean_motion_magnitude = np.mean(motion_magnitude)

    # 2. Direction change (variance of flow angle)
    _, angle = np.cos(flow[..., 0]), np.sin(flow[..., 1])
    direction_change = np.var(angle)

    # 3. Object area variation
    object_area = np.sum(fg_mask > 0)
    if len(frame_history) > 0:
        prev_fg_mask = frame_history[-1]["fg_mask"]
        prev_object_area = np.sum(prev_fg_mask > 0)
        object_area_variation = abs(object_area - prev_object_area)
    else:
        object_area_variation = 0

    # 4. Temporal instability (std dev of frame differences)
    if len(frame_history) > 1:
        frame_diffs = [f["frame_diff"] for f in frame_history]
        temporal_instability = np.std(frame_diffs)
    else:
        temporal_instability = 0

    return np.array([
        mean_motion_magnitude,
        direction_change,
        object_area_variation,
        temporal_instability
    ])
