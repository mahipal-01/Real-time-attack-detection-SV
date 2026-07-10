#pragma once

#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

constexpr int NUM_KALMAN_CHANNELS = 6;  // Ia, Ib, Ic, Va, Vb, Vc

class KalmanFilter {
public:
    KalmanFilter() = default;

    KalmanFilter(int num_harmonics, double q_amp, double q_dc, double r_meas)
        : num_harmonics_(num_harmonics)
        , n_(2 * num_harmonics + 1)
        , q_amp_(q_amp)
        , q_dc_(q_dc)
        , r_meas_(r_meas)
    {
        x_.assign(n_, 0.0);
        P_.assign(n_ * n_, 0.0);
        for (int i = 0; i < n_; ++i)
            P_[i * n_ + i] = 1.0;
    }

    void reset() {
        x_.assign(n_, 0.0);
        P_.assign(n_ * n_, 0.0);
        for (int i = 0; i < n_; ++i)
            P_[i * n_ + i] = 1.0;
    }

    void predict(double dt_norm = 1.0) {
        int h_states = 2 * num_harmonics_;
        for (int i = 0; i < h_states; ++i)
            P_[i * n_ + i] += q_amp_ * dt_norm;
        P_[n_ * n_ - 1] += q_dc_ * dt_norm;
    }

    double get_prediction(double wt) const {
        double pred = 0.0;
        for (int k = 0; k < num_harmonics_; ++k) {
            double k_wt = (k + 1) * wt;
            double s = std::sin(k_wt);
            double c = std::cos(k_wt);
            pred += s * x_[2 * k] + c * x_[2 * k + 1];
        }
        pred += x_[n_ - 1];
        return pred;
    }

    void update(double z, double wt) {
        std::vector<double> H(n_);
        for (int k = 0; k < num_harmonics_; ++k) {
            double k_wt = (k + 1) * wt;
            H[2 * k]     = std::sin(k_wt);
            H[2 * k + 1] = std::cos(k_wt);
        }
        H[n_ - 1] = 1.0;

        double y = z;
        for (int i = 0; i < n_; ++i)
            y -= H[i] * x_[i];

        std::vector<double> HP(n_, 0.0);
        for (int j = 0; j < n_; ++j)
            for (int i = 0; i < n_; ++i)
                HP[j] += H[i] * P_[i * n_ + j];

        double S = 0.0;
        for (int j = 0; j < n_; ++j)
            S += HP[j] * H[j];
        S += r_meas_;

        double S_inv = 1.0 / S;

        std::vector<double> K(n_);
        for (int i = 0; i < n_; ++i)
            K[i] = HP[i] * S_inv;

        for (int i = 0; i < n_; ++i)
            x_[i] += K[i] * y;

        std::vector<double> P_new(n_ * n_);
        for (int r = 0; r < n_; ++r)
            for (int c = 0; c < n_; ++c)
                P_new[r * n_ + c] = P_[r * n_ + c] - K[r] * HP[c];
        P_ = std::move(P_new);
    }

    int num_harmonics() const { return num_harmonics_; }
    int state_dim() const { return n_; }

private:
    int num_harmonics_ = 1;
    int n_ = 3;
    std::vector<double> x_;  // [a1, b1, a2, b2, ..., aN, bN, dc]
    std::vector<double> P_;  // n_ × n_ covariance (row-major)
    double q_amp_ = 1e-6;
    double q_dc_ = 1e-8;
    double r_meas_ = 1e-4;
};

class KalmanManager {
public:
    KalmanManager() = default;

    KalmanManager(int num_harmonics, double thresh,
                  double q_amp, double q_dc, double r_meas)
        : num_harmonics_(num_harmonics)
        , thresh_(thresh)
    {
        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i)
            filters_[i] = KalmanFilter(num_harmonics, q_amp, q_dc, r_meas);
    }

    void reset() {
        theta_ = 0.0;
        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i)
            filters_[i].reset();
    }

    bool process(double channels_pu[6], bool is_manipulation) {
        theta_ += OMEGA_DT;
        if (theta_ >= TWO_PI) theta_ -= TWO_PI;

        max_innovation_ = 0.0;
        reconstructing_ = false;

        for (int i = 0; i < NUM_KALMAN_CHANNELS; ++i) {
            filters_[i].predict(1.0);
            double pred = filters_[i].get_prediction(theta_);
            double innov = std::fabs(channels_pu[i] - pred);

            if (innov > max_innovation_)
                max_innovation_ = innov;

            predicted_pu_[i] = pred;
            received_pu_[i] = channels_pu[i];

            if (innov > thresh_ && is_manipulation) {
                reconstructing_ = true;
            } else {
                filters_[i].update(channels_pu[i], theta_);
            }
        }

        return reconstructing_;
    }

    double max_innovation() const { return max_innovation_; }
    bool is_reconstructing() const { return reconstructing_; }
    double predicted_pu(int ch) const { return predicted_pu_[ch]; }
    double received_pu(int ch) const { return received_pu_[ch]; }
    int num_harmonics() const { return num_harmonics_; }

private:
    static constexpr double TWO_PI = 6.283185307179586;
    static constexpr double OMEGA_DT = 0.07853981633974483;  // 2*pi*50/4000

    int num_harmonics_ = 1;
    KalmanFilter filters_[NUM_KALMAN_CHANNELS];
    double thresh_ = 0.05;
    double theta_ = 0.0;
    double max_innovation_ = 0.0;
    bool reconstructing_ = false;
    double predicted_pu_[NUM_KALMAN_CHANNELS]{};
    double received_pu_[NUM_KALMAN_CHANNELS]{};
};
