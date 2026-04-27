<!-- converted from AutoTracker_Metrics_Guide.docx -->

AutoTracker — How to Measure Metrics from Test Videos

1. Overview
This document explains the two ways to collect performance metrics for the AutoTracker monocular zoom-tracking system:
  A) Live-session log analysis via analyze_logs.py
  B) Offline benchmark simulation via evaluation/benchmark_mobile.py

Both methods measure the same core quantities: FPS, zoom stability, detection count, and tracking persistence.
2. Method A — Live Session Metrics (analyze_logs.py)
Step 1: Run a test session
Run the tracker against your test video or webcam. Every frame writes a [METRIC] line to autotracker.log automatically.
To use a video file instead of a webcam, edit config/default.yaml:
device_id: 0          # 0 = webcam; change to a video path string if needed
Then launch the tracker from the project root:
python run.py
Step 2: Let it run, then quit
Press  Q  on the display window to stop the session. The file  autotracker.log  is saved in the project root and contains one [METRIC] line per processed frame, e.g.:
[METRIC] FPS=28.45  Detections=3  Zoom=1.230  TargetPresent=1
Step 3: Run the log analyser
From the project root:
python autotracker/analyze_logs.py
This reads autotracker.log and produces the following outputs inside the analysis_outputs/ folder:
- metrics.txt  — printed summary of all key statistics
- fps.png       — FPS over time graph
- zoom.png      — zoom level stability graph
- detections.png — detections-per-frame graph
- tracking.png  — target presence (0/1) over time graph
Step 4: Read the Metrics Output
Open analysis_outputs/metrics.txt. It contains:
- Average FPS — mean frames per second across the session
- FPS Std Dev — how stable the frame rate was (lower = more consistent)
- Zoom Variance / Std Dev — how much zoom fluctuated (lower = smoother)
- Tracking Persistence — fraction of frames the target was locked onto (1.0 = perfect, 0.0 = never tracked)
3. Method B — Offline Benchmark (benchmark_mobile.py)
This method does NOT need a camera or a recorded video. It synthesises moving-rectangle frames and measures raw pipeline speed at two resolutions (640×480 VGA and 1280×720 HD).
Step 1: Run the benchmark
From the project root:
python evaluation/benchmark_mobile.py
⚠  NOTE: Make sure the model file specified in the script (yolo26n.pt) is present in the project root before running.
Step 2: Read the Console Output
The benchmark prints results directly to the console for each resolution:
Results for 640x480 over 50 frames:
  Average FPS: 34.12
  Inference + Tracking Time: 21.30 ms
  Zoom/Parallax Logic Time:  0.41 ms
  Total Time per Frame:      29.30 ms
Key metrics explained:
- Average FPS — overall throughput of the full pipeline
- Inference + Tracking Time — time spent inside YOLO + tracker per frame (ms)
- Zoom/Parallax Logic Time — time spent computing zoom and applying warp (ms)
- Total Time per Frame — end-to-end latency per frame (ms)
4. Running the Test Takes
The Test_Takes/ folder contains two standalone scripts that run the tracker directly against a webcam feed for quick manual inspection:
Take 1 — botsort tracker (best.pt model)
python Test_Takes/Take_1/Project_Mr-Sippy.py
- Uses BoT-SORT tracker
- Requires best.pt in the working directory
- Press ESC to stop
Take 2 — bytetrack tracker (best_take2.pt model)
python Test_Takes/Take_2/Project-Mr_Sippy2.py
- Uses ByteTrack tracker
- Requires best_take2.pt in the working directory
- Prints detection count to console each frame — useful for quick sanity checks
- Press ESC to stop
⚠  NOTE: Neither Test Take writes a log file. To capture metrics from a Test Take, wrap the frame loop with the AutoTracker logger or redirect console output.
5. Metrics Quick-Reference Table

6. Tips & Troubleshooting
- autotracker.log not found: Make sure you ran `python run.py` first — the log is created automatically in the project root.
- Empty metrics.txt: The log file exists but no [METRIC] lines were found. Ensure the session ran for at least a few seconds with a selected target.
- Very low FPS: Check that your GPU drivers are installed. Ultralytics will use CUDA automatically if available.
- Model not found error: Place yolo26n.pt in the project root (same folder as run.py).
- ESC/Q not responding: Click on the OpenCV display window first to give it keyboard focus.
| Metric | What It Measures | Good Range | Source |
| --- | --- | --- | --- |
| Average FPS | Throughput of entire pipeline | > 25 FPS | analyze_logs / benchmark |
| FPS Std Dev | Frame rate stability | < 5 | analyze_logs |
| Zoom Variance | Smoothness of zoom control | < 0.05 | analyze_logs |
| Tracking Persistence | Fraction of frames target is locked | > 0.85 | analyze_logs |
| Inference Time (ms) | YOLO + tracker time per frame | < 30 ms | benchmark_mobile |
| Zoom Logic Time (ms) | Warp computation per frame | < 2 ms | benchmark_mobile |
| Total Frame Time (ms) | End-to-end latency | < 40 ms | benchmark_mobile |