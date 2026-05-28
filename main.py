"""
main.py — Entry point for the Real-Time Zoom Tracker
======================================================

Usage examples
--------------
  python main.py                             # default webcam (index 0)
  python main.py --source 1                  # second webcam
  python main.py --source video.mp4          # video file
  python main.py --source rtsp://...         # RTSP stream
  python main.py --device cuda               # GPU acceleration
  python main.py --conf 0.25 --classes 0     # detect only people (COCO class 0)
  python main.py --no-zoom-redetect          # disable Pass-2 feedback loop
  python main.py --max-zoom 8.0              # increase zoom ceiling
"""

import argparse
from modules import ZoomTrackingPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time YOLO + ByteTrack Zoom Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", default="0",
        help="Video source: webcam index (integer string), file path, or RTSP URL.",
    )
    p.add_argument(
        "--model", default="yolo26n.pt",
        help="Ultralytics YOLO model.  'yolo26n.pt' is the default for this version.",
    )
    p.add_argument(
        "--model-version", default=None,
        help="Explicit version name for the model log (e.g. 'yolo26n'). Defaults to filename without extension.",
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold.",
    )
    p.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device.",
    )
    p.add_argument(
        "--classes", type=int, nargs="*", default=None,
        help="COCO class IDs to detect (default: all classes).  "
             "Example: --classes 0 2 7  →  person, car, truck.",
    )
    p.add_argument(
        "--no-zoom-redetect", dest="no_zoom_redetect", action="store_true",
        help="Disable the zoomed re-detection feedback loop (Pass 2).",
    )
    p.add_argument(
        "--min-zoom", type=float, default=1.0,
        help="Minimum zoom level.",
    )
    p.add_argument(
        "--max-zoom", type=float, default=6.0,
        help="Maximum zoom level.",
    )
    p.add_argument(
        "--ref-ratio", type=float, default=0.08,
        help="Bbox/frame area ratio that maps to 1× zoom. "
             "Increase to zoom in more aggressively on large objects.",
    )
    p.add_argument(
        "--window", default="Zoom Tracker",
        help="OpenCV window title.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Coerce numeric source strings to int (webcam index)
    source: int | str = args.source
    try:
        source = int(args.source)
    except ValueError:
        pass  # leave as string (file path / URL)

    import os
    model_version = args.model_version if args.model_version else os.path.splitext(os.path.basename(args.model))[0]

    pipeline = ZoomTrackingPipeline(
        source=source,
        model_path=args.model,
        model_version=model_version,
        conf_threshold=args.conf,
        enable_zoom_redetect=not args.no_zoom_redetect,
        device=args.device,
        window_name=args.window,
    )

    # Apply optional overrides to ZoomEngine after construction
    pipeline.zoom_engine.min_zoom        = args.min_zoom
    pipeline.zoom_engine.max_zoom        = args.max_zoom
    pipeline.zoom_engine.reference_ratio = args.ref_ratio

    # Apply optional class filter to detector
    if args.classes is not None:
        pipeline.detector.target_classes = args.classes

    pipeline.run()


if __name__ == "__main__":
    main()
