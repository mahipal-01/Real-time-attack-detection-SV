#pragma once

#include <cmath>
#include <string>
#include <unordered_map>

enum class TimingAttackType {
    NONE,
    REPLAY,
    SEQ_MANIP,
    TIME_SYNC,
    UNKNOWN
};

struct TimingResult {
    TimingAttackType type = TimingAttackType::NONE;
    int recon_smpCnt = -1;
    double recon_refrTm = -1.0;
    double recon_refrTm_seconds = -1.0;
    bool quality_suspicious = false;
    bool synch_suspicious = false;
};

class TimingReconstructor {
public:
    TimingResult process(int smpCnt, double refrTm_seconds,
                          int refrTmQuality, int smpSynch,
                          double smpCnt_diff, double refrTm_gap_ms,
                          double time_quality_change, double smp_synch_change,
                          bool is_timing_prediction) {
        TimingResult result;

        if (!initialized_) {
            last_valid_smpCnt_ = smpCnt;
            last_valid_refrTm_ = refrTm_seconds;
            baseline_quality_ = refrTmQuality;
            baseline_synch_ = smpSynch;
            recon_smpCnt_ = smpCnt;
            recon_refrTm_ = refrTm_seconds;
            initialized_ = true;
            return result;
        }

        if (!is_timing_prediction) {
            update_valid(smpCnt, refrTm_seconds, refrTmQuality, smpSynch);
            return result;
        }

        // Classify timing attack subtype
        if (std::abs(time_quality_change) > 0.5 || std::abs(smp_synch_change) > 0.5) {
            result.type = TimingAttackType::TIME_SYNC;
        } else if (std::abs(smpCnt_diff - 1.0) > 0.1) {
            result.type = TimingAttackType::SEQ_MANIP;
        } else if (refrTm_gap_ms > REFRTM_GAP_THRESH_MS) {
            result.type = TimingAttackType::REPLAY;
        } else {
            result.type = TimingAttackType::UNKNOWN;
        }

        // Apply corrections per subtype
        if (result.type == TimingAttackType::REPLAY) {
            recon_smpCnt_ = (recon_smpCnt_ + 1) % 4000;
            recon_refrTm_ = recon_refrTm_ + REFRTM_DT_SEC;
            result.recon_smpCnt = recon_smpCnt_;
            result.recon_refrTm_seconds = recon_refrTm_;
        } else if (result.type == TimingAttackType::SEQ_MANIP) {
            recon_smpCnt_ = (recon_smpCnt_ + 1) % 4000;
            result.recon_smpCnt = recon_smpCnt_;
            result.recon_refrTm_seconds = refrTm_seconds;
        } else if (result.type == TimingAttackType::TIME_SYNC) {
            recon_refrTm_ = recon_refrTm_ + REFRTM_DT_SEC;
            result.recon_refrTm_seconds = recon_refrTm_;
            result.recon_smpCnt = smpCnt;
            result.quality_suspicious = (std::abs(time_quality_change) > 0.5);
            result.synch_suspicious = (std::abs(smp_synch_change) > 0.5);
        } else {
            result.recon_smpCnt = smpCnt;
            result.recon_refrTm_seconds = refrTm_seconds;
        }

        return result;
    }

    void reset() {
        initialized_ = false;
        last_valid_smpCnt_ = -1;
        last_valid_refrTm_ = -1.0;
        baseline_quality_ = -1;
        baseline_synch_ = -1;
        recon_smpCnt_ = -1;
        recon_refrTm_ = -1.0;
    }

private:
    void update_valid(int smpCnt, double refrTm_seconds,
                       int refrTmQuality, int smpSynch) {
        last_valid_smpCnt_ = smpCnt;
        last_valid_refrTm_ = refrTm_seconds;
        baseline_quality_ = refrTmQuality;
        baseline_synch_ = smpSynch;
        recon_smpCnt_ = smpCnt;
        recon_refrTm_ = refrTm_seconds;
    }

    int last_valid_smpCnt_ = -1;
    double last_valid_refrTm_ = -1.0;
    int baseline_quality_ = -1;
    int baseline_synch_ = -1;
    int recon_smpCnt_ = -1;
    double recon_refrTm_ = -1.0;
    bool initialized_ = false;

    static constexpr double REFRTM_DT_SEC = 0.00025;
    static constexpr double REFRTM_GAP_THRESH_MS = 1.0;
};
