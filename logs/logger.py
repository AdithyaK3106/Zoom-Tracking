import os
import json
import time
import math
from typing import List, Dict, Any, Tuple, Optional

class FrameLogger:
    """
    A modular, extensible logging system for real-time computer vision pipelines.
    Logs per-frame telemetry to JSON Lines (JSONL) and optionally CSV format.
    
    Performance optimized using buffered writes to avoid blocking inference.
    """

    def __init__(
        self, 
        model_version: str, 
        log_dir: str = "logs/frame_logs", 
        buffer_size: int = 60,
        enable_csv: bool = False
    ):
        """
        Initialize the logger.
        
        Args:
            model_version (str): Identifier for the model/experiment (e.g. 'v1', 'yolov8n').
                                 This will be used as the filename.
            log_dir (str): Base directory for logs.
            buffer_size (int): Number of frames to buffer in memory before flushing to disk.
                               Default 60 (~2 seconds at 30 FPS).
            enable_csv (bool): If True, also exports a flattened CSV representation.
        """
        self.model_version = model_version
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.enable_csv = enable_csv
        
        self.buffer: List[Dict[str, Any]] = []
        self.current_frame: Dict[str, Any] = {}
        
        # Ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup file paths
        # We use .json extension for compatibility, but the format is JSON-Lines (one JSON object per line)
        # to allow O(1) appends and easy resumption.
        self.json_path = os.path.join(self.log_dir, f"{self.model_version}.json")
        self.csv_path = os.path.join(self.log_dir, f"{self.model_version}.csv")
        
        # Determine if CSV needs headers written
        self._csv_header_written = os.path.exists(self.csv_path) if self.enable_csv else False

    def start_frame(self, frame_id: int):
        """
        Start logging for a new frame. Resets the frame state.
        
        Args:
            frame_id (int): Absolute frame index.
        """
        self.current_frame = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            # Default empty values in case no detection occurs in this frame
            "bbox": None,
            "confidence": None,
            "track_id": None,
            "zoom_factor": 1.0,
            "object_size": None,
            "center_error": None,
            "latency_ms": None
        }

    def log_detection(self, bbox: List[float], confidence: float, track_id: int):
        """
        Log detection targets of the frame.
        
        Args:
            bbox (List[float]): Bounding box in format [x, y, w, h].
            confidence (float): Detection confidence (0.0 to 1.0).
            track_id (int): Identifier for the tracked object.
        """
        self.current_frame["bbox"] = bbox
        self.current_frame["confidence"] = confidence
        self.current_frame["track_id"] = track_id

    def log_zoom(self, zoom_factor: float):
        """
        Log camera control / scaling metrics.
        
        Args:
            zoom_factor (float): The applied digital or optical zoom factor.
        """
        self.current_frame["zoom_factor"] = zoom_factor

    def log_metrics(self, object_size: float, center_error: float):
        """
        Log tracking error and derived bounding box metrics.
        
        Args:
            object_size (float): Normalized area of the target (bbox_area / frame_area).
            center_error (float): Euclidean distance from frame center to object center.
        """
        self.current_frame["object_size"] = object_size
        self.current_frame["center_error"] = center_error

    def end_frame(self, latency_ms: float):
        """
        Finalize the current frame, measure latency, and append to buffer.
        If buffer threshold is met, flushes to disk.
        
        Args:
            latency_ms (float): Processing time of the frame in milliseconds.
        """
        self.current_frame["latency_ms"] = latency_ms
        self.buffer.append(self.current_frame)
        self.current_frame = {}
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """
        Write buffered frames to disk in an append-only transaction.
        Designed to be non-blocking with small localized writes.
        """
        if not self.buffer:
            return

        # 1. Write to JSON-Lines
        with open(self.json_path, 'a', encoding='utf-8') as f:
            for frame in self.buffer:
                # json.dumps ensures it goes to a single line without newlines
                f.write(json.dumps(frame) + '\n')

        # 2. Optionally write to CSV
        if self.enable_csv:
            with open(self.csv_path, 'a', encoding='utf-8') as f:
                # If creating for the first time, write column headers
                if not self._csv_header_written:
                    headers = [
                        "frame_id", "timestamp", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                        "confidence", "track_id", "zoom_factor", "object_size", 
                        "center_error", "latency_ms"
                    ]
                    f.write(",".join(headers) + "\n")
                    self._csv_header_written = True
                
                # Write rows
                for frame in self.buffer:
                    b = frame.get('bbox') or [None, None, None, None]
                    row = [
                        frame.get('frame_id'), frame.get('timestamp'),
                        b[0], b[1], b[2], b[3], # flattened bbox
                        frame.get('confidence'), frame.get('track_id'),
                        frame.get('zoom_factor'), frame.get('object_size'),
                        frame.get('center_error'), frame.get('latency_ms')
                    ]
                    f.write(",".join(str(val) if val is not None else "" for val in row) + "\n")

        # Clear buffer after successful disk write
        self.buffer.clear()

    # --- Helper Methods ---

    @staticmethod
    def compute_derived_metrics(
        bbox: List[float], 
        frame_width: int, 
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Helper method to compute object_size and center_error.
        
        Args:
            bbox (List[float]): Bounding box in [x, y, w, h] format.
            frame_width (int): Frame width.
            frame_height (int): Frame height.
            
        Returns:
            Tuple[float, float]: (object_size, center_error)
        """
        x, y, w, h = bbox
        
        # Compute object size (fraction of bounding box area relative to frame area)
        bbox_area = w * h
        frame_area = frame_width * frame_height
        object_size = bbox_area / frame_area if frame_area > 0 else 0.0
        
        # Compute center error (Euclidean distance from object center to frame center)
        obj_center_x = x + (w / 2.0)
        obj_center_y = y + (h / 2.0)
        
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        
        dx = obj_center_x - frame_center_x
        dy = obj_center_y - frame_center_y
        center_error = math.sqrt(dx**2 + dy**2)
        
        return object_size, center_error
