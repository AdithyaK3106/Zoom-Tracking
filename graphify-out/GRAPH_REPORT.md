# Graph Report - zoom-tracking  (2026-04-27)

## Corpus Check
- 11 files · ~8,546 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 128 nodes · 262 edges · 9 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 117 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]

## God Nodes (most connected - your core abstractions)
1. `TrackedObject` - 24 edges
2. `UserInteractionModule` - 19 edges
3. `FrameLogger` - 18 edges
4. `ZoomTrackingPipeline` - 17 edges
5. `ZoomEngine` - 17 edges
6. `VideoCapture` - 16 edges
7. `DetectionModule` - 13 edges
8. `TrackingModule` - 13 edges
9. `EvaluationMetrics` - 11 edges
10. `Detection` - 11 edges

## Surprising Connections (you probably didn't know these)
- `TrackedObject` --uses--> `Module 4 — User Interaction ============================ Handles all user input:`  [INFERRED]
  modules\tracker.py → modules\user_interaction.py
- `TrackedObject` --uses--> `Attach mouse handler.  Call after cv2.namedWindow() has been created.`  [INFERRED]
  modules\tracker.py → modules\user_interaction.py
- `TrackedObject` --uses--> `Consume any pending click and resolve it to the best matching track.          Sh`  [INFERRED]
  modules\tracker.py → modules\user_interaction.py
- `TrackedObject` --uses--> `Returns True if the currently selected track is still active.         Sets the r`  [INFERRED]
  modules\tracker.py → modules\user_interaction.py
- `TrackedObject` --uses--> `Render instruction banner at the bottom of the frame (in-place).`  [INFERRED]
  modules\tracker.py → modules\user_interaction.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (19): Detection, DetectionModule, Single detection output from YOLO., Run inference on a single BGR frame.          Parameters         ----------, _draw_corners(), Main loop.  Blocks until 'Q' is pressed or the stream ends., Combine full-frame and zoomed detections, remap zoomed ones to         original, Apply class-agnostic Non-Maximum Suppression to remove duplicate         detecti (+11 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (12): Convert detections to a (N, 6) float32 array:             [x1, y1, x2, y2, confi, compute_derived_metrics(), FrameLogger, Finalize the current frame, measure latency, and append to buffer.         If bu, Write buffered frames to disk in an append-only transaction.         Designed to, Initialize the logger.                  Args:             model_version (str): I, Start logging for a new frame. Resets the frame state.                  Args:, Log detection targets of the frame.                  Args:             bbox (Lis (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (12): Drives the headless execution of the ZoomTrackingPipeline to collect     telemet, Loads GT JSON. Expects a list of frame dicts: {frame_id, bbox, object_id, occlus, Parses the JSONL output from logs/logger.py, Runs the mathematical evaluation on a completed log against its GT., Calculates Zoom Gain and performs paired comparison., Saves the final nested JSON output to results directory., SystemEvaluator, compute_iou() (+4 more)

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (7): Module 2 — Detection ===================== Wraps Ultralytics YOLO (nano by defau, Module 6 — Feedback Loop Pipeline =================================== Orchestrat, _track_color(), Module 3 — Tracking (ByteTrack) ================================ Wraps `supervis, Module 4 — User Interaction ============================ Handles all user input:, Module 1 — Video Capture ======================== Wraps OpenCV VideoCapture to p, Module 5 — Zoom Engine ======================= Implements dynamic zoom using bou

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (8): _dist2(), Returns True if the currently selected track is still active.         Sets the r, Render instruction banner at the bottom of the frame (in-place)., Externally force re-selection (e.g. via 'R' key)., Squared Euclidean distance — avoids sqrt for comparison., Attach mouse handler.  Call after cv2.namedWindow() has been created., Consume any pending click and resolve it to the best matching track.          Sh, UserInteractionModule

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (3): Grab next frame.  Returns (success, BGR frame | None)., Returns (width, height)., VideoCapture

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (3): main(), parse_args(), main.py — Entry point for the Real-Time Zoom Tracker ===========================

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (1): Computes IoU between two boxes [x, y, w, h]

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (1): Helper method to compute object_size and center_error.                  Args:

## Knowledge Gaps
- **24 isolated node(s):** `main.py — Entry point for the Real-Time Zoom Tracker ===========================`, `Parses the JSONL output from logs/logger.py`, `Computes IoU between two boxes [x, y, w, h]`, `Computes all standard and system-specific metrics defined in the research plan.`, `Calculates Zoom Gain between baseline and adaptive models.` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (1 nodes): `Computes IoU between two boxes [x, y, w, h]`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (1 nodes): `Helper method to compute object_size and center_error.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrackedObject` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.240) - this node is a cross-community bridge._
- **Why does `TrackingModule` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `TrackedObject` (e.g. with `ZoomTrackingPipeline` and `Module 6 — Feedback Loop Pipeline =================================== Orchestrat`) actually correct?**
  _`TrackedObject` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `UserInteractionModule` (e.g. with `ZoomTrackingPipeline` and `Module 6 — Feedback Loop Pipeline =================================== Orchestrat`) actually correct?**
  _`UserInteractionModule` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `FrameLogger` (e.g. with `ZoomTrackingPipeline` and `Module 6 — Feedback Loop Pipeline =================================== Orchestrat`) actually correct?**
  _`FrameLogger` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ZoomTrackingPipeline` (e.g. with `VideoCapture` and `DetectionModule`) actually correct?**
  _`ZoomTrackingPipeline` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ZoomEngine` (e.g. with `ZoomTrackingPipeline` and `Module 6 — Feedback Loop Pipeline =================================== Orchestrat`) actually correct?**
  _`ZoomEngine` has 10 INFERRED edges - model-reasoned connections that need verification._