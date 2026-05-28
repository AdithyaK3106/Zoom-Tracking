# Pipeline Validation Report

**Overall Status:** PASS

## Data Integrity
**Status:** PASS

```json
{
  "frame_match": true,
  "missing_fields": [],
  "null_values": 0,
  "status": "pass"
}
```

## Detection Validation
**Status:** PASS

```json
{
  "iou_mean": 0.898511198506783,
  "status": "pass"
}
```

## Tracking Validation
**Status:** PASS

```json
{
  "fragmentation": 0,
  "continuity": 1.0,
  "id_switches": 0,
  "status": "pass"
}
```

## Zoom Validation
**Status:** PASS

```json
{
  "correlation": 1.0,
  "jerk": 1.942890293094024e-16,
  "status": "pass"
}
```

## Feedback Validation
**Status:** PASS

```json
{
  "delta_iou": 0.08655612566748683,
  "status": "pass"
}
```

## Temporal Validation
**Status:** PASS

```json
{
  "lag_corr": 0.7948486680116015,
  "status": "pass"
}
```

## System Validation
**Status:** PASS

```json
{
  "metrics_valid": true,
  "pipeline_complete": true,
  "status": "pass"
}
```

## Visual Check Samples

![check_frame_0.png](visual_checks/check_frame_0.png)
![check_frame_5.png](visual_checks/check_frame_5.png)
![check_frame_9.png](visual_checks/check_frame_9.png)
