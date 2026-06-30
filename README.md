# Real-Time IDS — IEC 61850 SV Intrusion Detection System

Real-time packet capture and ML-based attack classification for IEC 61850 Sampled Value (SV) networks. Captures SV packets from a live interface, extracts 49 or 52 features per sample, and runs streaming ONNX models (S5 / RWKV) to classify each packet as `normal`, `manipulation`, `timing/protocol`, or `traffic`.

---

## Requirements

- **Windows** (uses Npcap + WinPcap API)
- **g++** (MinGW-w64, tested with GCC bundled with MSYS2)
- **Npcap SDK** — path set in `build.bat` as `NPACK_SDK`
- **ONNX Runtime** — pre-downloaded to `onnxruntime/` directory

---

## How to Build

```bat
cd real_time_cpp
build.bat
```

Produces `real_time_ids.exe`. `onnxruntime.dll` is copied automatically alongside the exe.

---

## How to Run

```bat
real_time_ids.exe <interface> [options]
```

### List available interfaces
```bat
:: Use pcap_findalldevs or check your network adapters
```

### Examples

```bat
real_time_ids.exe Ethernet0 --model s5
real_time_ids.exe "Wi-Fi" --model rwkv --csv output.csv
real_time_ids.exe eth0 --model s51 --csv output.csv --csv-interval 10
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `<interface>` | Network interface name (from pcap) | **required** |
| `--model` | Model to use: `s5`, `s51`, `rwkv`, `rwkv1` | `s51` |
| `--csv <path>` | Log predictions + features to CSV | (none) |
| `--csv-interval <N>` | Log every Nth packet to CSV | 1 |

---

## Model Reference

| `--model` | ONNX File | Features | Source `.pth` | Description |
|-----------|-----------|----------|---------------|-------------|
| `s5` | `s5_streaming.onnx` | 52 (v3) | `models/s5.pth` | State Space Model, v3 training |
| `rwkv` | `rwkv_streaming.onnx` | 52 (v3) | `models/rwkv.pth` | RWKV, v3 training |
| `s51` | `s51_streaming.onnx` | 49 (v2) | `s51.pth` | State Space Model, v2 training |
| `rwkv1` | `rwkv1_streaming.onnx` | 49 (v2) | `rwkv1.pth` | RWKV, v2 training |

All models have 4 output classes: `normal` (0), `manipulation` (1), `timing/protocol` (2), `traffic` (3).

---

## Output

State transitions are printed with the destination class's color:

| Class | Color |
|-------|-------|
| normal | Green |
| manipulation | Red |
| timing/protocol | Yellow |
| traffic | Magenta |

Example output:

```
[1782816456.695] Pkt#1760758 Pred:manipulation -> normal Conf:0.999 Latency:124.5us
[1782816456.741] Pkt#1760942 Pred:normal -> manipulation Conf:1.000 Latency:98.3us
```

On exit, a summary is printed:

```
[Summary] Packets processed: 1760942 | Predictions: 1760942 | Alerts: 34521
 | Avg Latency: 12.3us | Max Latency: 892.9us
  normal: 1726421
  manipulation: 23421
  timing/protocol: 7890
  traffic: 3210
```

`Latency` = packet decode + feature extraction + ML inference (microseconds). Capture-to-decode OS queuing delay is excluded.

---

## Feature Extraction

### v2 models (49 features — `s51`, `rwkv1`)
- 8 protocol/timing features, 7 raw electrical, 4 impedance, 2 phase angle, 4 sequence components, 4 impedance ratios, 3 step/share, 3 power, 2 waveform residual, 12 window-based features
- **Scaler:** root `scaler.pt` (5 columns: refrTm_diff, refrTm_gap, smpCnt_diff, forward_time_diff, offset_deviation)

### v3 models (52 features — `s5`, `rwkv`)
- Same 49 v2 features + 3 additional engineered features:
  - `smpCnt_zero_roll80` — rolling count of stuck sample counters (replay detection)
  - `fwd_refrTm_delta` — `forward_time_diff - refrTm_diff` (timing manipulation)
  - `pkt_rate_80` — `80 / sum(forward_time_diff)` clipped to [0,500] (traffic flood)
- **Scaler:** `models/scaler.pt` (8 columns: above 5 + the 3 new features)

---

## Evaluation

To compute accuracy, F1, precision, recall, and confusion matrix against a labeled dataset:

```bat
cd ..
python evaluate_real_time.py --gt <attacker_csv> --pred real_time_cpp/<pred_csv>
```

- `--gt`: raw SV CSV with `classification` column (e.g., from attacker simulation)
- `--pred`: CSV from `real_time_ids.exe --csv` output

Both are aligned by row index after dropping the first 80 warmup samples.

---

## File Structure

| File | Purpose |
|------|---------|
| `real_time_ids.cpp` | Main entry: pcap capture, parsing, feature extraction, model inference, output |
| `feature_extraction.h` / `.cpp` | Streaming feature extraction (49 or 52 features) |
| `onnx_model.h` / `.cpp` | ONNX Runtime wrapper for streaming S5 / RWKV models |
| `build.bat` | MinGW build script |
| `*.onnx` + `*.onnx.data` | Trained streaming ONNX models |
| `random_attacker.cpp` | (Separate) attacker simulation tool |
