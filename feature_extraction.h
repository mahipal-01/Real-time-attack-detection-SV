#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstring>

constexpr int WINDOW = 80;
constexpr int NUM_FEATURES = 49;
constexpr double I_BASE = 400.0;
constexpr double V_BASE = 107784.0;
constexpr double SQRT3_2 = 0.8660254037844386;
constexpr double ANGLE_EPS = 1e-10;
constexpr double I_MIN = 0.01;

constexpr double OMEGA_DT = 2.0 * 3.14159265358979323846 * 50.0 / 4000.0;
constexpr double EXPECTED_FACTOR = -2.0 * std::pow(std::sin(OMEGA_DT / 2.0), 2);

constexpr double SCALER_MEAN[5] = {
    0.25364324809353017, 8.039491415795533,
    1.1650035727269308, 0.2536413159163042, 99.23748894505066
};
constexpr double SCALER_SCALE[5] = {
    8.440975237817378, 29.896706662076262,
    251.07876692258165, 0.46327298227694935, 300.8134778237377
};
constexpr int SCALED_INDICES[5] = {5, 4, 3, 6, 7};

struct StreamState {
    double ring_buf[WINDOW][9];
    int ring_idx = 0;
    int ring_filled = 0;

    double sum[9] = {0};
    double sum_sq[9] = {0};
    double kcl_sum = 0;

    double step_norm_total = 0;
    int step_norm_count = 0;
    int anomaly_streak = 0;

    int last_refrTmQuality = -1;
    int last_smpSynch = -1;
    int last_smpCnt = -1;
    double max_refrTm_seen = -1;
    double baseline_offset = -1;
    double last_refrTm = -1;
    double last_forwardTime = -1;
    double last_Za_mag = -1;

    int first_refrTmQuality = -1;
    int first_smpSynch = -1;
    double offset_sum = 0;
    int offset_count = 0;
    double prev_Ia_curv = 0;
    double prev_Va_curv = 0;
    bool prev_curv_valid = false;

    void push_sample(const double vals[9]);
    double rms(int col) const;
    double mean(int col) const;
    double var_min(int col) const;
    double var_max(int col) const;
    int filled() const { return ring_filled; }
};

struct ParsedRecord {
    uint8_t dst_mac[6]{};
    uint8_t src_mac[6]{};
    bool has_vlan = false;
    uint16_t vlan_id = 0;
    uint8_t vlan_priority = 0;
    uint16_t app_id = 0;
    uint16_t length = 0;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t noAsdu = 0;
    char svID[65] = {0};
    char DatSet[65] = {0};
    uint32_t smpCnt = 0;
    uint32_t confrev = 0;
    uint8_t refrTm[8] = {0};
    double capture_time_sec = 0.0;
    uint32_t smpSynch = 0;
    uint32_t smpRate = 0;
    float channels[8] = {0.0f};
    uint32_t quality[8] = {0};
    bool valid = false;
};

inline std::string mac_to_string(const uint8_t* mac) {
    char buf[18];
    std::snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return std::string(buf);
}

void compute_features(
    const double channels[8],
    int smpCnt, double refrTm, double forwardTime,
    int smpSynch, int refrTmQuality,
    const std::string& src_mac,
    float out_features[49]);

extern std::string g_baseline_src;
