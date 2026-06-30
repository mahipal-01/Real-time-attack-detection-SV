#include "feature_extraction.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

std::string g_baseline_src;

void StreamState::push_sample(const double vals[10]) {
    if (ring_filled == WINDOW) {
        double* old = ring_buf[ring_idx];
        for (int i = 0; i < 10; ++i) {
            sum[i] -= old[i];
            sum_sq[i] -= old[i] * old[i];
        }
        kcl_sum -= std::abs(old[0] + old[1] + old[2] - old[6]);
    }
    std::memcpy(ring_buf[ring_idx], vals, sizeof(double) * 10);
    for (int i = 0; i < 10; ++i) {
        sum[i] += vals[i];
        sum_sq[i] += vals[i] * vals[i];
    }
    kcl_sum += std::abs(vals[0] + vals[1] + vals[2] - vals[6]);
    ring_idx = (ring_idx + 1) % WINDOW;
    if (ring_filled < WINDOW) ++ring_filled;
}

double StreamState::rms(int col) const {
    int n = ring_filled;
    if (n == 0) return 0.0;
    return std::sqrt(sum_sq[col] / n);
}

double StreamState::mean(int col) const {
    int n = ring_filled;
    if (n == 0) return 0.0;
    return sum[col] / n;
}

double StreamState::var_min(int col) const {
    if (ring_filled == 0) return 0.0;
    double m = ring_buf[0][col];
    int n = ring_filled;
    for (int i = 1; i < n; ++i)
        if (ring_buf[i][col] < m) m = ring_buf[i][col];
    return m;
}

double StreamState::var_max(int col) const {
    if (ring_filled == 0) return 0.0;
    double m = std::abs(ring_buf[0][col]);
    int n = ring_filled;
    for (int i = 1; i < n; ++i)
        if (std::abs(ring_buf[i][col]) > m) m = std::abs(ring_buf[i][col]);
    return m;
}

void compute_features(
    const double channels[8],
    int smpCnt, double refrTm, double forwardTime,
    int smpSynch, int refrTmQuality,
    const std::string& src_mac,
    float out_features[], int num_features)
{
    static StreamState s;

    double ia_pu = channels[0] / I_BASE;
    double ib_pu = channels[1] / I_BASE;
    double ic_pu = channels[2] / I_BASE;
    double in_pu = channels[3] / I_BASE;
    double va_pu = channels[4] / V_BASE;
    double vb_pu = channels[5] / V_BASE;
    double vc_pu = channels[6] / V_BASE;

    double f[52] = {0};

    if (g_baseline_src.empty())
        g_baseline_src = src_mac;
    f[0] = (src_mac != g_baseline_src) ? 1.0f : 0.0f;

    // f[1]: refrTmQuality change (vs first sample, matching batch)
    if (s.first_refrTmQuality == -1)
        s.first_refrTmQuality = refrTmQuality;
    if (refrTmQuality != s.first_refrTmQuality)
        f[1] = 1.0;

    // f[2]: smpSynch change (vs first sample, matching batch)
    if (s.first_smpSynch == -1)
        s.first_smpSynch = smpSynch;
    if (smpSynch != s.first_smpSynch)
        f[2] = 1.0;

    // f[3]: smpCnt difference (match batch: only 3999->0 is legitimate rollover)
    if (s.last_smpCnt != -1) {
        int raw_diff = smpCnt - s.last_smpCnt;
        if (s.last_smpCnt == 3999 && smpCnt == 0) {
            f[3] = 1.0;
        } else {
            f[3] = (double)raw_diff;
        }
    } else {
        f[3] = 0.0;  // batch: first row diff is NaN -> fillna(0)
    }

    // f[4]: max refrTm deviation
    if (s.max_refrTm_seen < 0) {
        s.max_refrTm_seen = refrTm;
    } else if (refrTm > s.max_refrTm_seen) {
        s.max_refrTm_seen = refrTm;
    }
    f[4] = (s.max_refrTm_seen - refrTm) * 1000.0;

    // f[5]: refrTm delta
    if (s.last_refrTm >= 0)
        f[5] = (refrTm - s.last_refrTm) * 1000.0;

    // f[6]: forwardTime delta
    if (s.last_forwardTime >= 0)
        f[6] = (forwardTime - s.last_forwardTime) * 1000.0;

    // f[7]: offset deviation (mean of first 80 samples as baseline, matching batch median)
    double offset = (forwardTime - refrTm) * 1000.0;
    if (s.offset_count < WINDOW) {
        s.offset_sum += offset;
        s.offset_count++;
        if (s.offset_count == WINDOW)
            s.baseline_offset = s.offset_sum / s.offset_count;
    }
    if (s.baseline_offset >= 0)
        f[7] = std::abs(offset - s.baseline_offset);

    // f[8..14]: raw per-unit values
    f[8] = ia_pu;
    f[9] = ib_pu;
    f[10] = ic_pu;
    f[11] = in_pu;
    f[12] = va_pu;
    f[13] = vb_pu;
    f[14] = vc_pu;

    // f[15..17]: impedance magnitudes
    double Za_mag = (std::abs(ia_pu) >= I_MIN) ? std::abs(va_pu / ia_pu) : 0.0;
    double Zb_mag = (std::abs(ib_pu) >= I_MIN) ? std::abs(vb_pu / ib_pu) : 0.0;
    double Zc_mag = (std::abs(ic_pu) >= I_MIN) ? std::abs(vc_pu / ic_pu) : 0.0;
    f[15] = Za_mag;
    f[16] = Zb_mag;
    f[17] = Zc_mag;

    // f[18]: impedance change
    if (s.last_Za_mag >= 0)
        f[18] = Za_mag - s.last_Za_mag;

    // Clarke transform
    double Ialpha = (2.0 / 3.0) * (ia_pu - 0.5 * ib_pu - 0.5 * ic_pu);
    double Ibeta = (2.0 / 3.0) * (SQRT3_2 * ib_pu - SQRT3_2 * ic_pu);
    double Izero = (1.0 / 3.0) * (ia_pu + ib_pu + ic_pu);
    double Valpha = (2.0 / 3.0) * (va_pu - 0.5 * vb_pu - 0.5 * vc_pu);
    double Vbeta = (2.0 / 3.0) * (SQRT3_2 * vb_pu - SQRT3_2 * vc_pu);
    double Vzero = (1.0 / 3.0) * (va_pu + vb_pu + vc_pu);

    double V_angle = std::atan2(Vbeta, Valpha + ANGLE_EPS);
    double I_angle = std::atan2(Ibeta, Ialpha + ANGLE_EPS);
    double phi_VI = V_angle - I_angle;
    phi_VI = std::atan2(std::sin(phi_VI), std::cos(phi_VI));

    // f[19..20]: phase angles
    f[19] = phi_VI;
    f[20] = std::abs(phi_VI);

    // f[21..24]: sequence components
    double V1_mag = std::sqrt(Valpha * Valpha + Vbeta * Vbeta);
    double I1_mag = std::sqrt(Ialpha * Ialpha + Ibeta * Ibeta);
    f[21] = V1_mag;
    f[22] = std::abs(Vzero);
    f[23] = I1_mag;
    f[24] = Vzero;

    // f[25]: Z1 magnitude
    double I1_clamp = (I1_mag >= I_MIN) ? I1_mag : std::numeric_limits<double>::quiet_NaN();
    f[25] = std::isnan(I1_clamp) ? 0.0 : V1_mag / (I1_clamp + ANGLE_EPS);

    // f[26]: Z0 magnitude
    double Izero_abs = std::abs(Izero);
    double Iz_clamp = (Izero_abs >= I_MIN) ? Izero_abs : std::numeric_limits<double>::quiet_NaN();
    f[26] = std::isnan(Iz_clamp) ? 0.0 : f[22] / (Iz_clamp + ANGLE_EPS);

    // f[27..28]: sequence ratios (0 when Z1 couldn't be computed, matching batch fillna(0) after NaN propagation)
    if (f[25] == 0.0) {
        f[27] = 0.0;
        f[28] = 0.0;
    } else {
        f[27] = V1_mag / 2.0 / (f[25] + ANGLE_EPS);
        f[28] = f[26] / (f[25] + ANGLE_EPS);
    }

    // Step-wise features (dIa, dIb, dIc, dVa, dVb, dVc)
    double dIa = 0, dIb = 0, dIc = 0, dVa = 0, dVb = 0, dVc = 0;
    if (s.ring_filled > 0) {
        double* prev = s.ring_buf[(s.ring_idx - 1 + WINDOW) % WINDOW];
        dIa = ia_pu - prev[0];
        dIb = ib_pu - prev[1];
        dIc = ic_pu - prev[2];
        dVa = va_pu - prev[3];
        dVb = vb_pu - prev[4];
        dVc = vc_pu - prev[5];
    }

    double step_norm = std::sqrt(dIa * dIa + dIb * dIb + dIc * dIc +
                                  dVa * dVa + dVb * dVb + dVc * dVc);
    f[29] = step_norm;

    double step_norm_safe = step_norm + ANGLE_EPS;
    double max_step = std::max({std::abs(dIa), std::abs(dIb), std::abs(dIc),
                                std::abs(dVa), std::abs(dVb), std::abs(dVc)});
    f[30] = max_step / step_norm_safe;
    f[31] = std::abs(dVa) / step_norm_safe;

    // f[32..34]: instantaneous reactive power, power factor
    f[32] = Valpha * Ibeta - Vbeta * Ialpha;
    double inst_power = va_pu * ia_pu + vb_pu * ib_pu + vc_pu * ic_pu;
    f[33] = std::atan2(f[32], inst_power + ANGLE_EPS);
    f[34] = std::abs(f[33] - 0.318);

    // f[35..36]: waveform curvature with 1-sample delay to match batch .shift(-1)
    if (s.ring_filled >= 2) {
        double* prev1 = s.ring_buf[(s.ring_idx - 1 + WINDOW) % WINDOW];
        double* prev2 = s.ring_buf[(s.ring_idx - 2 + WINDOW) % WINDOW];
        double Ia_curv = ia_pu - 2 * prev1[0] + prev2[0];
        double Va_curv = va_pu - 2 * prev1[3] + prev2[3];
        if (s.prev_curv_valid) {
            f[35] = std::abs(s.prev_Ia_curv - EXPECTED_FACTOR * ia_pu);
            f[36] = std::abs(s.prev_Va_curv - EXPECTED_FACTOR * va_pu);
        }
        s.prev_Ia_curv = Ia_curv;
        s.prev_Va_curv = Va_curv;
        s.prev_curv_valid = true;
    }

    // Track smpCnt_zero for v3 rolling feature
    double is_smpCnt_zero = (f[3] == 0.0) ? 1.0 : 0.0;
    if (s.ring_filled == WINDOW) {
        int old_idx = s.ring_idx;
        if (s.zero_smpCnt_ring[old_idx])
            s.zero_smpCnt_sum -= 1.0;
    }
    s.zero_smpCnt_ring[s.ring_idx] = (is_smpCnt_zero > 0.5);
    s.zero_smpCnt_sum += is_smpCnt_zero;

    // Store sample in ring buffer before window stats (col 9 = forward_time_diff)
    double sample[10] = {ia_pu, ib_pu, ic_pu, va_pu, vb_pu, vc_pu, in_pu, step_norm, phi_VI, f[6]};
    s.push_sample(sample);

    // Window features (O(1) via running sums)
    int n = s.filled();
    if (n > 0) {
        f[37] = s.rms(0);   // Ia RMS
        f[38] = s.rms(1);   // Ib RMS
        f[39] = s.rms(2);   // Ic RMS
        f[40] = s.rms(3);   // Va RMS

        f[41] = s.var_min(3);  // Va min

        double Ia_rms = f[37];
        f[42] = s.var_max(0) / (Ia_rms + ANGLE_EPS);  // Ia crest factor

        double Vb_rms = s.rms(4);
        double Vc_rms = s.rms(5);
        double v_means[3] = {f[40], Vb_rms, Vc_rms};
        double v_mean = (v_means[0] + v_means[1] + v_means[2]) / 3.0;
        double v_var = ((v_means[0]-v_mean)*(v_means[0]-v_mean) +
                        (v_means[1]-v_mean)*(v_means[1]-v_mean) +
                        (v_means[2]-v_mean)*(v_means[2]-v_mean)) / 2.0;
        f[43] = std::sqrt(v_var);  // V unbalance stddev

        f[44] = s.kcl_sum / n;  // KCL violation mean
        f[45] = s.mean(7);      // step_norm mean

        f[47] = s.rms(8);       // phi_VI RMS
        f[48] = s.mean(0);      // Ia DC offset (mean)
    }

    // f[46]: anomaly streak
    if (s.ring_filled > 0) {
        s.step_norm_total += step_norm;
        s.step_norm_count++;
        double longterm_mean = s.step_norm_total / s.step_norm_count;
        if (step_norm > 3 * longterm_mean)
            s.anomaly_streak++;
        else
            s.anomaly_streak = 0;
    }
    f[46] = (double)s.anomaly_streak;

    // f[49]: smpCnt_zero_roll80 — rolling count of smpCnt_diff==0 in window
    f[49] = s.zero_smpCnt_sum;

    // f[50]: fwd_refrTm_delta — forward_time_diff - refrTm_diff
    f[50] = f[6] - f[5];

    // f[51]: pkt_rate_80 — 80 / sum(forward_time_diff over window), clipped [0,500]
    if (s.ring_filled > 0) {
        double fwd_sum = s.sum[9];
        double span = std::max(fwd_sum, 1e-6);
        f[51] = std::min(80.0 / span, 500.0);
    }

    // Scaler normalization (v2 or v3 depending on feature count)
    if (num_features == NUM_FEATURES_V3) {
        for (int i = 0; i < 8; ++i) {
            int idx = SCALED_INDICES_V3[i];
            f[idx] = (f[idx] - SCALER_MEAN_V3[i]) / SCALER_SCALE_V3[i];
        }
    } else {
        for (int i = 0; i < 5; ++i) {
            int idx = SCALED_INDICES[i];
            f[idx] = (f[idx] - SCALER_MEAN[i]) / SCALER_SCALE[i];
        }
    }

    // Update state
    s.last_refrTmQuality = refrTmQuality;
    s.last_smpSynch = smpSynch;
    s.last_smpCnt = smpCnt;
    s.last_refrTm = refrTm;
    s.last_forwardTime = forwardTime;
    s.last_Za_mag = Za_mag;

    // Copy to output
    int nf = (num_features == NUM_FEATURES_V3) ? NUM_FEATURES_V3 : NUM_FEATURES;
    for (int i = 0; i < nf; ++i)
        out_features[i] = std::isfinite(f[i]) ? (float)f[i] : 0.0f;
}
