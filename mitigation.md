# Mitigation Strategies

## Overview

Two independent, show-only reconstruction systems that detect and correct manipulated SV fields without modifying the actual packet stream. Both run after ML inference in `real_time_ids.cpp`, using the ML prediction as a cross-validation gate.

---

## 1. Manipulation Mitigation — Kalman Filter

### What it does
Tracks the expected sinusoidal waveform for each of 6 channels (Ia, Ib, Ic, Va, Vb, Vc) using a 3-state Kalman filter per channel. When the received sample deviates from the predicted sinusoid AND the ML model says `manipulation`, the system flags reconstruction.

### Kalman model

```
State per channel:  [a, b, dc]^T
  a = in-phase amplitude (cosine component)
  b = quadrature amplitude (sine component)
  dc = DC offset

Measurement:  z = a·sin(ωt) + b·cos(ωt) + dc

State transition:  x_{t} = x_{t-1} + process_noise  (F = I)

Phase tracking:  θ_t = θ_{t-1} + ω·Δt  (local accumulator, 50Hz × 4000Hz)
```

### Flow per packet

```
1. Kalman.predict()    → advance state, get expected sample
2. innovation = |received - predicted|
3. If innovation > thresh AND ML says manipulation:
     → RECONSTRUCTING: use predicted value, DON'T update Kalman
   Else:
     → NORMAL: use received value, DO update Kalman
```

### Parameters

| CLI flag | Default | Description |
|---|---|---|
| `--kalman-thresh` | 0.05 | Innovation threshold in pu |
| `--kalman-Qa` | 1e-6 | Amplitude process noise |
| `--kalman-R` | 1e-4 | Measurement noise |

### Source
- `kalman.h` — `KalmanFilter` (3-state), `KalmanManager` (6 channels + phase)

### Output
```
[RECONSTRUCTING] Pred:manipulation (0.97) Ia:+0.92→+0.95 Va:-0.11→-0.12 Latency:47us
```

---

## 2. Timing/Protocol Mitigation — Rule-Based Reconstruction

### What it does
Classifies the specific timing attack subtype using feature-based rules, then corrects the manipulated protocol fields accordingly. Runs only when ML predicts `timing/protocol`.

### Classification rules (evaluated in order)

| Priority | Condition | Subtype | Fields corrected |
|---|---|---|---|
| 1 | `time_quality_change == 1` OR `smp_synch_change == 1` | **TIME SYNC** | refrTm (increment), mark quality/synch suspicious |
| 2 | `smpCnt_diff != 1` (skip/dup/zero) | **SEQ MANIP** | smpCnt only (increment from last valid) |
| 3 | `refrTm_gap > 1ms` | **REPLAY** | smpCnt + refrTm (increment) + also activates Kalman |
| 4 | none match | **UNKNOWN** | none |

### Correction logic

During an active timing attack:
- **smpCnt**: increments by 1 each sample from the last corrected value (`(recon + 1) % 4000`)
- **refrTm**: increments by 250µs each sample from the last corrected value
- **quality/synch**: flagged as "suspicious" for TIME SYNC only

When prediction returns to `normal`:
- `last_valid` values are updated from received data (re-syncs the tracker)

### Source
- `timing_reconstruct.h` — `TimingReconstructor` class, per-stream state

### Output examples
```
[REPLAY] [RECONSTRUCTING] Pred:timing/protocol (0.97) smpCnt:1520→1521 refrTm:12345.000246→12345.000250 Ia:+0.92→+0.95 Va:-0.11→-0.12 Latency:47us
[SEQ MANIP] Pred:timing/protocol (0.98) smpCnt:1523→1524 Latency:46us
[TIME SYNC] Pred:timing/protocol (0.96) refrTm:12345.002300→12345.000251 quality:suspicious synch:suspicious Latency:46us
[TIMING] Pred:timing/protocol (0.95) Latency:46us
```

---

## 3. Traffic Flood Mitigation — Packet Skip Mode

### What it does
When ML predicts `traffic` for consecutive packets beyond a sustain threshold, the system enters **FLOOD_SKIP** mode: feature extraction and ML inference are bypassed entirely to prevent processing lag. The packet rate is measured using a simple counter + wall-clock timer, and normal processing resumes when the rate drops below a recovery threshold.

### Flow

```
1. ML predicts traffic (class 3)
2. traffic_count++ 
3. if traffic_count >= flood_sustain (default 150):
     → FLOOD_SKIP: skip feature extraction & ML for all subsequent packets
4. Every flood_check_interval (default 1000) packets:
     → compute rate = interval_count / elapsed_wall_time
     → if rate < flood_recover_rate (default 6/s) OR rate <= 5000/s (normal SV rate):
         → exit FLOOD_SKIP, reset traffic_count (normal-rate traffic is NOT a flood)
       else:
         → print [FLOOD ACTIVE] rate:X/s (throttled to 1s)
5. continue → back to top of loop, skipping all downstream processing
```

### Why skip, not reconstruct
- Traffic floods are high-rate (thousands of packets/s) — Kalman and timing reconstruction cannot keep up
- The goal is to survive the flood, not reconstruct individual samples
- Rate-based recovery ensures the system doesn't permanently lock itself out

### Parameters

| CLI flag | Default | Description |
|---|---|---|
| `--flood-sustain` | 150 | Consecutive `traffic` predictions to trigger flood mode |
| `--flood-recover-rate` | 6.0 | Rate below which flood mode exits (packets/s) |
| `--flood-check` | 1000 | Packet interval between rate checks |

### Output
```
[Info] Flood detected. Entering skip mode.
[FLOOD ACTIVE] rate:12000/s
[FLOOD ACTIVE] rate:11500/s
[Info] Flood ended. Resuming full processing.
```

---

## 4. Design Principles

### Show-only
- ML model and feature extraction run on **raw** (attacked) data — never modified
- Reconstructed values are displayed only, not forwarded or reinjected
- No risk of introducing waveform discontinuities or feedback loops

### Cross-validation
- Kalman: both high innovation AND ML confirmation required before triggering
- Timing: ML says `timing/protocol` before subtype classification runs
- Prevents false triggers on transients (faults, switching events)

### Per-stream state
- Both Kalman and TimingReconstructor track state per source MAC
- Independent filters for each SV stream, no cross-contamination

### Minimal latency impact
- Kalman: ~1µs per sample (500 FLOPs for 6 channels)
- Timing reconstruction: ~0.1µs per sample
- Total overhead: < 2% of current ~46µs processing time
