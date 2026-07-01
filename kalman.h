#pragma once

#include <cmath>
#include <cstring>
#include <cstdio>

constexpr int NUM_KALMAN_CHANNELS = 6;  // Ia, Ib, Ic, Va, Vb, Vc

struct KalmanState {
    double a = 0.0, b = 0.0, dc = 0.0;
    double P[9] = {1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0};
};

class KalmanFilter {
public:
    KalmanFilter() = default;

    KalmanFilter(double q_amp, double q_dc, double r_meas)
        : q_amp_(q_amp), q_dc_(q_dc), r_meas_(r_meas) {}

    void reset() {
        std::memset(&s_, 0, sizeof(s_));
        s_.P[0] = s_.P[4] = s_.P[8] = 1.0;
    }

    void predict(double dt_norm = 1.0) {
        s_.P[0] += q_amp_ * dt_norm;
        s_.P[4] += q_amp_ * dt_norm;
        s_.P[8] += q_dc_ * dt_norm;
    }

    double get_prediction(double sin_wt, double cos_wt) const {
        return s_.a * sin_wt + s_.b * cos_wt + s_.dc;
    }

    void update(double z, double sin_wt, double cos_wt) {
        double H[3] = {sin_wt, cos_wt, 1.0};
        double y = z - (s_.a * H[0] + s_.b * H[1] + s_.dc * H[2]);

        double HP[3];
        for (int i = 0; i < 3; ++i)
            HP[i] = s_.P[i*3+0]*H[0] + s_.P[i*3+1]*H[1] + s_.P[i*3+2]*H[2];
        double S = HP[0]*H[0] + HP[1]*H[1] + HP[2]*H[2] + r_meas_;

        double K[3];
        for (int i = 0; i < 3; ++i) K[i] = HP[i] / S;

        s_.a += K[0] * y;
        s_.b += K[1] * y;
        s_.dc += K[2] * y;

        double P_new[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                P_new[r*3+c] = s_.P[r*3+c] - K[r] * HP[c];
        std::memcpy(s_.P, P_new, sizeof(s_.P));
    }

private:
    KalmanState s_;
    double q_amp_ = 1e-6;
    double q_dc_ = 1e-8;
    double r_meas_ = 1e-4;
};

class KalmanManager {
public:
    KalmanManager() = default;

    KalmanManager(double thresh, double q_amp, double q_dc, double r_meas)
        : thresh_(thresh) {
        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i)
            filters_[i] = KalmanFilter(q_amp, q_dc, r_meas);
    }

    void reset() {
        theta_ = 0.0;
        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i)
            filters_[i].reset();
    }

    bool process(double channels_pu[6], bool is_manipulation) {
        theta_ += OMEGA_DT;
        if (theta_ >= TWO_PI) theta_ -= TWO_PI;

        double sin_wt = std::sin(theta_);
        double cos_wt = std::cos(theta_);

        max_innovation_ = 0.0;
        reconstructing_ = false;

        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i) {
            filters_[i].predict(1.0);
            double pred = filters_[i].get_prediction(sin_wt, cos_wt);
            double innov = std::fabs(channels_pu[i] - pred);

            if (innov > max_innovation_)
                max_innovation_ = innov;

            predicted_pu_[i] = pred;
            received_pu_[i] = channels_pu[i];

            if (innov > thresh_ && is_manipulation) {
                reconstructing_ = true;
            } else {
                filters_[i].update(channels_pu[i], sin_wt, cos_wt);
            }
        }

        return reconstructing_;
    }

    double max_innovation() const { return max_innovation_; }
    bool is_reconstructing() const { return reconstructing_; }
    double predicted_pu(int ch) const { return predicted_pu_[ch]; }
    double received_pu(int ch) const { return received_pu_[ch]; }

private:
    static constexpr double TWO_PI = 6.283185307179586;
    static constexpr double OMEGA_DT = 0.07853981633974483;  // 2*pi*50/4000

    KalmanFilter filters_[NUM_KALMAN_CHANNELS];
    double thresh_ = 0.05;
    double theta_ = 0.0;
    double max_innovation_ = 0.0;
    bool reconstructing_ = false;
    double predicted_pu_[NUM_KALMAN_CHANNELS]{};
    double received_pu_[NUM_KALMAN_CHANNELS]{};
};
