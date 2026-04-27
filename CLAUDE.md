# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the application

```bash
# Activate the virtual environment first
source venv/bin/activate

# Webcam (requires usbipd passthrough on WSL2 — see note below)
python main.py

# Video file
python main.py --source cars.mp4

# Common overrides
python main.py --source cars.mp4 --device cuda       # GPU inference
python main.py --source cars.mp4 --classes 0 2       # COCO classes only (person, car)
python main.py --source cars.mp4 --no-zoom-redetect  # disable feedback loop
python main.py --source cars.mp4 --conf 0.25 --max-zoom 8.0
```

**WSL2 webcam note:** WSL2 has no camera devices by default. Use `usbipd attach --wsl --busid <id>` on the Windows host to pass through a physical webcam, or use a video file.

## Architecture

The system is a six-module pipeline in `modules/`. Each module is a single class with no cross-module imports except through the pipeline.

```
VideoCapture → DetectionModule (Pass 1: full frame)
                      ↓
              TrackingModule (ByteTrack via supervision)
                      ↓
         UserInteractionModule (click → selected track_id)
                      ↓
               ZoomEngine → zoomed_frame
                      ↓
              DetectionModule (Pass 2: zoomed frame) ← feedback loop
                      ↓
              merge + NMS → TrackingModule (same instance, same frame)
```

`ZoomTrackingPipeline` in `modules/pipeline.py` owns all module instances and runs the loop. `main.py` is only argument parsing + construction.

## Key design constraints

**Two-pass detection feedback loop** (`pipeline.py:_merge_and_nms`): Pass 1 runs YOLO on the original frame. If a track is selected, Pass 2 runs YOLO again on the zoomed crop. Pass-2 detections are remapped to original-frame coordinates via `ZoomEngine.map_bbox_to_original()`, merged with Pass-1, and deduplicated with `cv2.dnn.NMSBoxes` before the tracker sees them. Disable with `--no-zoom-redetect` to benchmark Pass-1 alone.

**Zoom formula** (`zoom_engine.py`): `zoom = REFERENCE_RATIO / (bbox_area / frame_area)`, clamped to `[min_zoom, max_zoom]`. `REFERENCE_RATIO = 0.08` means an object filling 8% of the frame gets 1× zoom. Both zoom level and crop center are EMA-smoothed (`α = 0.12`) to suppress bbox jitter. Call `zoom_engine.reset()` when switching the selected track.

**ByteTrack boundary**: `DetectionModule.to_bytetrack_input()` converts `Detection` objects to `(N, 6)` float32 `[x1,y1,x2,y2,conf,class_id]`. `TrackingModule.update()` wraps `supervision.ByteTracker` and returns `TrackedObject` instances. Neither module imports from the other.

**Click matching** (`user_interaction.py`): Priority 1 = click inside a bbox (nearest center wins on overlap). Priority 2 = nearest bbox center within 80 px. Clicks are stored on the OpenCV mouse callback and consumed lazily in `process_click()` in the main loop to avoid threading issues.

## Tunable parameters

All live on module instances and can be changed after `ZoomTrackingPipeline.__init__()`:

| Attribute | Default | Effect |
|---|---|---|
| `zoom_engine.reference_ratio` | `0.08` | Lower = zoom in sooner |
| `zoom_engine.smooth_alpha` | `0.12` | Higher = snappier, more jitter |
| `zoom_engine.max_zoom` | `6.0` | Hard ceiling on magnification |
| `zoom_engine.padding_factor` | `2.5` | Crop padding around tracked bbox |
| `detector.conf_threshold` | `0.35` | Lower = more detections, more false positives |
| `tracker.tracker` params | — | Re-instantiate `sv.ByteTracker` directly |

## Dependencies

- `ultralytics` — YOLO inference only (`model.predict()`). The tracking method `model.track()` is intentionally not used.
- `supervision` — provides `ByteTracker` and `Detections` as a standalone tracker decoupled from YOLO.
- `opencv-python` — capture, display, NMS (`cv2.dnn.NMSBoxes`), mouse callbacks.
- `torch` — pulled in by ultralytics; no direct usage in this codebase.
