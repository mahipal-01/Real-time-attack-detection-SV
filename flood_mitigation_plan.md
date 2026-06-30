# Flood Mitigation Plan

## Problem

During a flooding attack (random MACs or single source), the IDS:
- Wastes CPU on ML inference for every flood packet (~200µs each)
- Contaminates the ring buffer window features with flood samples
- Produces garbage predictions for 80+ samples after the flood ends
- Cannot distinguish flood packets at software level without network hardware control

## Architecture

### 1. Global Rate Monitor

In `real_time_ids.cpp`, add:

```cpp
static std::atomic<uint64_t> g_sec_pkt_count{0};
static std::atomic<int> g_global_rate{0};
static std::atomic<bool> g_flood_active{false};
```

A background check every 1 second computes pkts/sec. If > threshold (default 6000/s), set `g_flood_active = true`. If rate stays below threshold for 2 consecutive seconds, set it back to `false`.

### 2. Per-MAC Stream Re-initialization

In `StreamState` (`feature_extraction.h`):

```cpp
bool needs_reinit = false;
int reinit_remaining = 80;

void reset_window() {
    ring_idx = 0;
    ring_filled = 0;
    memset(sum, 0, sizeof(sum));
    memset(sum_sq, 0, sizeof(sum_sq));
    kcl_sum = 0;
}
```

When `g_flood_active` transitions to `true`:
- Set `needs_reinit = true` for ALL streams

When `g_flood_active` transitions to `false`:
- Call `reset_window()` on each stream with `needs_reinit`
- Set `reinit_remaining = WINDOW` (80)

### 3. Processing Flow

```
Normal:
  g_flood_active = false → full pipeline (parse → features → model → print)

Flood start:
  rate > 6000/s → g_flood_active = true
  → compute_features returning early (no ring buffer push, no model features)
  → packet_handler: skip model.predict(), just count packet
  → Mark all streams needs_reinit = true

Flood ongoing:
  All processing skipped → CPU drops to <0.1µs/pkt
  Continue counting for rate monitoring

Flood ends:
  rate < 6000/s for 2 consecutive seconds → g_flood_active = false
  For each stream with needs_reinit:
    → reset_window()
    → reinit_remaining = 80
  For each packet:
    → push to ring buffer
    → if reinit_remaining > 0:
        → decrement, skip model.predict()
    → when reinit_remaining == 0:
        → resume full processing
```

### 4. Summary Output

```
[Summary] Packets processed: 72450 | Predictions: 61200 | Alerts: 1420
  Flood discarded: 10000 | Reinit skipped: 1250
  normal: 59800
  manipulation: 340
  timing/protocol: 280
  traffic: 800
```

### 5. CLI Arguments

```
--flood-rate 6000      # pkts/sec threshold to trigger mitigation
--flood-cooldown 2     # seconds below threshold before declaring flood over
--reinit-window 80     # samples to rebuild window after flood
```

### 6. Trade-offs

| Preserved | Sacrificed |
|-----------|------------|
| CPU protection during flood (2000x reduction) | Per-packet classification during flood (not possible — flood packets are noise) |
| Detection of flood itself (via rate counter) | Detection during 80-sample reinit (~20ms, negligible) |
| Clean window features post-flood (via reinit) | |
| ML model hidden state preserved | |

### 7. Files to modify

- `feature_extraction.h` — add `needs_reinit`, `reinit_remaining`, `reset_window()` to `StreamState`
- `feature_extraction.cpp` — add flood check gate at start of `compute_features`
- `real_time_ids.cpp` — add globals, rate monitoring, skip logic in packet_handler, updated summary

---

# Manipulation Mitigation Plan

## Problem

During a manipulation attack, the ML model only detects ~32% of packets as "manipulation":
- 58% are missed entirely (predicted "normal")
- No mitigation actions differentiate low vs high severity
- No evidence preservation for forensics

## Philosophy

- Model detects first → then mitigation fires if severity is high enough
- **Never skip ML inference** — we need to know when the attack ends
- **Never reinit the ring buffer** — window features naturally recover when clean values return
- "Drop" is a **recommendation** to the operator, not an actual network drop

## Architecture

### 1. Detection flow

```
Model predicts
  └── class 0 (normal) → continue normally
  └── class 1 (manipulation) with confidence > 0.5:
       ├── Check 5 rule violations
       ├── Compute severity = f(conf, violations, hit_window)
       └── If severity > threshold + sustained for N hits → RECOMMEND DROP
```

### 2. Rule violations (checked per-packet when model predicts manipulation)

| Rule | Source | Threshold |
|------|--------|-----------|
| KCL violation | f[44] (Ia+Ib+Ic vs In) | `|Ia+Ib+Ic| - |In| > 0.1pu` |
| Impedance jump | f[18] (Za delta) | `|Za - last_Za| / last_Za > 0.5` |
| Phase angle instability | f[47] (phi_VI RMS) | Running RMS of phi_VI > 0.5 rad |
| Step norm spike | f[29] (step_norm) | `step_norm > 5 * longterm_mean` |
| Anomaly streak | f[46] (already computed) | `anomaly_streak > 3` |

### 3. StreamState additions

```cpp
int manip_hits = 0;               // total manipulation predictions seen
int manip_hit_window[10];         // rolling bitmask (1=manip, 0=normal)
int manip_window_idx = 0;
bool recommend_drop = false;      // true = operator should drop this MAC
int clean_streak = 0;             // consecutive normal predictions
```

### 4. State machine

```cpp
// Called after model.predict() returns
if (pred_idx == 1 && confidence > 0.5) {
    // Count rule violations
    int violations = 0;
    if (/* KCL violation */) violations++;
    if (/* impedance jump */) violations++;
    if (/* phase instability */) violations++;
    if (/* step_norm spike */) violations++;
    if (/* anomaly streak */) violations++;

    manip_hits++;
    clean_streak = 0;

    // Update rolling window
    manip_hit_window[manip_window_idx % 10] = 1;
    manip_window_idx++;
    int sum_window = 0;
    for (int i = 0; i < 10; i++) sum_window += manip_hit_window[i];

    // Severity = weighted combination (0.0 - 1.0)
    float severity = confidence * 0.4f +
                     (violations / 5.0f) * 0.3f +
                     (sum_window / 10.0f) * 0.3f;

    if (severity > 0.7f && manip_hits >= 5) {
        if (!recommend_drop) {
            recommend_drop = true;
            // Log: "[ALERT] RECOMMEND DROP MAC:xx:xx conf=0.92 viol=3/5 streak=7"
        }
    }
} else {
    clean_streak++;

    // Update rolling window (shift in a 0)
    manip_hit_window[manip_window_idx % 10] = 0;
    manip_window_idx++;

    if (recommend_drop && clean_streak >= 200) {
        recommend_drop = false;
        manip_hits = 0;
        // Log: "[INFO] MAC:xx:xx recovered. Drop recommendation lifted."
    }
}
```

### 5. What "recommend drop" means

| Aspect | Behavior |
|--------|----------|
| Ring buffer | Packets still pushed → window features stay valid |
| ML inference | Still runs → detects when attack ends |
| Console | Shows `RECOMMEND DROP!` in RED with severity score |
| Log file | Writes to `manip_alerts.log` with full packet context |
| PCAP dump | Saves raw Ethernet frame to `manip_dump.pcap` (evidence) |
| Actual network drop | No — operator must act on the recommendation |

### 6. Console output during drop recommendation

```
[1699123456.789] Pkt#12345 RECOMMEND DROP! MAC:aa:bb:cc:dd:ee:ff
  Pred:manipulation Conf:0.924 | Severity:0.87 | Violations:3/5 | Streak:7
```

### 7. CLI arguments

```
--manip-drop-conf 0.7       # severity threshold to recommend drop
--manip-drop-hits 5         # manipulation hits needed before recommending
--manip-clean-streak 200    # normal packets before lifting recommendation
```

### 8. Summary output addition

```
[Summary] Packets processed: 72450 | Predictions: 72450 | Alerts: 1420
  normal: 60230
  manipulation: 9500 (RECOMMEND DROP: 4500 packets from aa:bb:cc:dd:ee:ff)
  timing/protocol: 9200
  traffic: 1020
```

### 9. Trade-offs

| Preserved | Sacrificed |
|-----------|------------|
| Full ML inference always runs (know when attack ends) | CPU not saved (but manipulation is rare — irrelevant) |
| Ring buffer stays clean naturally | None |
| Severity-based graduated response | |
| Evidence preserved in PCAP dump | |

### 10. Files to modify

- `feature_extraction.h` — add `manip_hits`, `manip_hit_window[10]`, `manip_window_idx`, `recommend_drop`, `clean_streak` to `StreamState`
- `feature_extraction.cpp` — add KCL threshold comparison, impedance jump check (already computed, just threshold against them)
- `real_time_ids.cpp` — add severity computation, state machine after `model.predict()`, console/log mitigation actions, updated summary
- `manip_alerts.log` (new) — append-only log file created at startup
- `manip_dump.pcap` (new) — PCAP dump per compromised MAC (created on first CONFIRMED)

---

# Timing/Protocol Mitigation Plan

## Problem

Timing/protocol attacks (replay, desync, delay, quality flag manipulation) are often intermittent — a single delayed packet doesn't mean a sustained attack. The model also produces 8.7% false timing/protocol predictions during manipulation attacks, so mitigation must tolerate jitter.

## Philosophy

Same as manipulation: **keep ML inference active always, never skip, never reinit**. Use severity scoring with rule violations. Recommend drop only on high severity sustained over a long window. Higher confirmation threshold than manipulation to tolerate normal network jitter.

## Attack types covered

| Attack | What they do | Detection hook |
|--------|-------------|----------------|
| **Replay** | Capture valid SV, re-inject later | smpCnt repeats or goes backwards, refrTm not monotonic |
| **Desynchronization** | Modify refrTm to shift sampling window | refrTm delta jumps, offset deviation spikes |
| **Packet delay** | Hold packets before forwarding | forwardTime - refrTm offset suddenly increases |
| **Protocol manipulation** | Flip smpSynch/TimeQuality bits | f[1], f[2] spike repeatedly |

## Architecture

### 1. Detection flow

Same as manipulation: model predicts → if timing/protocol (class 2) with conf > 0.5 → check 6 rule violations → compute severity → if high + sustained ≥ 10 hits → RECOMMEND DROP.

### 2. StreamState additions

```cpp
int timing_hits = 0;
int timing_hit_window[10];
int timing_window_idx = 0;
bool timing_recommend_drop = false;
int timing_clean_streak = 0;
```

### 3. Rule violations (checked when model predicts class 2)

| Rule | Source | Threshold |
|------|--------|-----------|
| Non-monotonic smpCnt | f[3] raw (pre-scaler) | `raw_diff <= 0` (duplicate or backward) |
| refrTm goes backwards | refrTm - last_refrTm | `delta < -0.0001s` |
| refrTm delta spike | f[5] raw (pre-scaler) | `abs(delta - 0.00025) > 0.001s` |
| Offset deviation | f[7] raw (pre-scaler) | `offset - baseline > 2ms` |
| Quality flag changes | f[1], f[2] | Sum > 0 for 3+ consecutive packets |
| smpCnt wrap error | smpCnt | Wrap outside expected 0/3999 boundary |

### 4. State machine

```cpp
// Called after model.predict() returns
if (pred_idx == 2 && confidence > 0.5) {
    int violations = 0;
    if (/* non-monotonic smpCnt */) violations++;
    if (/* refrTm backwards */) violations++;
    if (/* refrTm delta spike */) violations++;
    if (/* offset deviation */) violations++;
    if (/* quality flags */) violations++;
    if (/* smpCnt wrap error */) violations++;

    timing_hits++;
    timing_clean_streak = 0;

    timing_hit_window[timing_window_idx % 10] = 1;
    timing_window_idx++;
    int sum_window = 0;
    for (int i = 0; i < 10; i++) sum_window += timing_hit_window[i];

    // Severity = same weighting formula as manipulation
    float severity = confidence * 0.4f +
                     (violations / 6.0f) * 0.3f +
                     (sum_window / 10.0f) * 0.3f;

    // Higher hit threshold than manipulation (10 vs 5) to tolerate jitter
    if (severity > 0.7f && timing_hits >= 10) {
        if (!timing_recommend_drop) {
            timing_recommend_drop = true;
            // Log: "[ALERT] RECOMMEND DROP MAC:xx:xx (timing/protocol)"
        }
    }
} else {
    timing_clean_streak++;

    timing_hit_window[timing_window_idx % 10] = 0;
    timing_window_idx++;

    if (timing_recommend_drop && timing_clean_streak >= 500) {
        timing_recommend_drop = false;
        timing_hits = 0;
        // Log: "[INFO] MAC:xx:xx timing recovered. Drop recommendation lifted."
    }
}
```

### 5. What "recommend drop" means

| Aspect | Behavior |
|--------|----------|
| Ring buffer | Packets still pushed → window features stay valid |
| ML inference | Still runs → detects when attack ends |
| Console | Shows `RECOMMEND DROP!` in **YELLOW** (lower criticality than manipulation's RED) |
| Log file | Writes to `timing_alerts.log` with full packet context |
| PCAP dump | Saves raw Ethernet frame to `timing_dump.pcap` (evidence) |
| Actual network drop | No — operator must act on the recommendation |

### 6. Console output during drop recommendation

```
[1699123456.789] Pkt#12345 RECOMMEND DROP! MAC:aa:bb:cc:dd:ee:ff
  Pred:timing/protocol Conf:0.881 | Severity:0.79 | Violations:4/6
```

(YELLOW color)

### 7. CLI arguments

```
--timing-drop-conf 0.7      # severity threshold to recommend drop
--timing-drop-hits 10       # timing hits needed (higher = jitter tolerance)
--timing-clean-streak 500   # normal packets before lifting recommendation
```

### 8. Summary output

```
[Summary] Packets processed: 72450 | Predictions: 72450 | Alerts: 1420
  normal: 60230
  manipulation: 9500 (RECOMMEND DROP: 4500 packets)
  timing/protocol: 9200 (RECOMMEND DROP: 3100 packets)
  traffic: 1020
```

### 9. Trade-offs

| Preserved | Sacrificed |
|-----------|------------|
| Full ML inference always runs | CPU not saved (timing attacks are rare) |
| Ring buffer stays clean naturally | None |
| Higher threshold avoids jitter false positives | Delays detection of true attacks by a few packets |
| Evidence preserved in PCAP dump | |

### 10. Files to modify

- `feature_extraction.h` — add `timing_hits`, `timing_hit_window[10]`, `timing_window_idx`, `timing_recommend_drop`, `timing_clean_streak` to `StreamState`
- `feature_extraction.cpp` — no new computations needed (all violations use existing features), just threshold comparisons
- `real_time_ids.cpp` — add timing state machine alongside manipulation state machine, YELLOW console color, separate PCAP dump file, updated summary
- `timing_dump.pcap` (new) — PCAP dump for timing/protocol attacks
