# Pipeline Validation Report

**Overall Status:** FAIL

## Data Integrity
**Status:** FAIL

```json
{
  "frame_match": true,
  "missing_fields": [],
  "null_values": 32,
  "status": "fail"
}
```

## Detection Validation
**Status:** FAIL

```json
{
  "iou_mean": 0.43491777366106993,
  "status": "fail"
}
```

## Tracking Validation
**Status:** FAIL

```json
{
  "fragmentation": 4,
  "tracking_stability": 0.3230088495575221,
  "id_switches": 0,
  "status": "fail"
}
```

## Zoom Validation
**Status:** PASS

```json
{
  "correlation": 0.0,
  "jerk": 0.0,
  "status": "pass"
}
```

## Feedback Validation
**Status:** PASS

```json
{
  "delta_iou": 0.17948755883921824,
  "status": "pass"
}
```

## Temporal Validation
**Status:** FAIL

```json
{
  "lag_corr": 0.0,
  "status": "fail"
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
![check_frame_339.png](visual_checks/check_frame_339.png)
![check_frame_677.png](visual_checks/check_frame_677.png)
