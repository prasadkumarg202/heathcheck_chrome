#include "aurapulse.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurapulse {

AuraPulseNativeEngine::AuraPulseNativeEngine(
    float min_duration_s,
    float window_duration_s,
    float target_fs
) : min_duration_s_(min_duration_s),
    window_duration_s_(window_duration_s),
    target_fs_(target_fs),
    session_start_s_(-1.0) {}

void AuraPulseNativeEngine::startSession() {
    forehead_buffer_.clear();
    left_cheek_buffer_.clear();
    right_cheek_buffer_.clear();
    session_start_s_ = -1.0;
}

void AuraPulseNativeEngine::pushFrameSignals(
    double timestamp_s,
    float fh_r, float fh_g, float fh_b,
    float lc_r, float lc_g, float lc_b,
    float rc_r, float rc_g, float rc_b
) {
    if (session_start_s_ < 0.0) {
        session_start_s_ = timestamp_s;
    }

    forehead_buffer_.push_back({fh_r, fh_g, fh_b, timestamp_s});
    left_cheek_buffer_.push_back({lc_r, lc_g, lc_b, timestamp_s});
    right_cheek_buffer_.push_back({rc_r, rc_g, rc_b, timestamp_s});

    size_t max_samples = static_cast<size_t>(window_duration_s_ * target_fs_ * 1.5f);
    if (forehead_buffer_.size() > max_samples) {
        size_t excess = forehead_buffer_.size() - max_samples;
        forehead_buffer_.erase(forehead_buffer_.begin(), forehead_buffer_.begin() + excess);
        left_cheek_buffer_.erase(left_cheek_buffer_.begin(), left_cheek_buffer_.begin() + excess);
        right_cheek_buffer_.erase(right_cheek_buffer_.begin(), right_cheek_buffer_.begin() + excess);
    }
}

VitalMetrics AuraPulseNativeEngine::computeVitals() {
    VitalMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false, "Invalid", ""};

    if (forehead_buffer_.empty()) {
        metrics.rejection_reason = "NO_DATA_BUFFERED";
        return metrics;
    }

    double elapsed = forehead_buffer_.back().timestamp_s - session_start_s_;
    if (elapsed < min_duration_s_) {
        metrics.rejection_reason = "MEASUREMENT_IN_PROGRESS_NEED_" + std::to_string(static_cast<int>(min_duration_s_)) + "S";
        return metrics;
    }

    auto bvp_fh = SignalProcessor::extractPOS(forehead_buffer_, target_fs_);
    auto bvp_lc = SignalProcessor::extractPOS(left_cheek_buffer_, target_fs_);
    auto bvp_rc = SignalProcessor::extractPOS(right_cheek_buffer_, target_fs_);

    size_t N = bvp_fh.size();
    if (N < 30) {
        metrics.rejection_reason = "INSUFFICIENT_SAMPLES";
        return metrics;
    }

    std::vector<float> fused(N, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        fused[i] = 0.50f * bvp_fh[i] + 0.25f * bvp_lc[i] + 0.25f * bvp_rc[i];
    }

    auto filtered_hr = SignalProcessor::butterworthBandpass(fused, 0.65f, 3.5f, target_fs_, 3);
    auto detrended_hr = SignalProcessor::detrend(filtered_hr);

    size_t n_fft = 1024;
    std::vector<float> padded(n_fft, 0.0f);
    for (size_t i = 0; i < N && i < n_fft; ++i) {
        float w = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N - 1)));
        padded[i] = detrended_hr[i] * w;
    }

    float max_mag = 0.0f;
    float peak_freq = 0.0f;
    for (size_t k = 0; k < n_fft / 2; ++k) {
        float freq = static_cast<float>(k) * target_fs_ / n_fft;
        if (freq < 0.65f || freq > 3.5f) continue;

        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < n_fft; ++n) {
            float angle = -2.0f * M_PI * k * n / n_fft;
            sum += padded[n] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        float mag = std::abs(sum);
        if (mag > max_mag) {
            max_mag = mag;
            peak_freq = freq;
        }
    }

    float hr_bpm = peak_freq * 60.0f;

    auto resp_filtered = SignalProcessor::butterworthBandpass(fused, 0.1f, 0.5f, target_fs_, 2);
    float max_resp_mag = 0.0f;
    float peak_resp_freq = 0.25f;
    for (size_t k = 0; k < n_fft / 2; ++k) {
        float freq = static_cast<float>(k) * target_fs_ / n_fft;
        if (freq < 0.1f || freq > 0.5f) continue;

        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < n_fft; ++n) {
            float angle = -2.0f * M_PI * k * n / n_fft;
            sum += resp_filtered[std::min(n, N - 1)] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        float mag = std::abs(sum);
        if (mag > max_resp_mag) {
            max_resp_mag = mag;
            peak_resp_freq = freq;
        }
    }
    float resp_rpm = peak_resp_freq * 60.0f;

    std::vector<double> peak_times;
    int min_dist = static_cast<int>(target_fs_ * 0.4f);
    for (size_t i = 1; i < N - 1; ++i) {
        if (detrended_hr[i] > detrended_hr[i - 1] && detrended_hr[i] > detrended_hr[i + 1] && detrended_hr[i] > 0.0f) {
            if (peak_times.empty() || (i - peak_times.back() * target_fs_) >= min_dist) {
                float y0 = detrended_hr[i - 1], y1 = detrended_hr[i], y2 = detrended_hr[i + 1];
                float delta = (y0 - y2) / (2.0f * (y0 - 2.0f * y1 + y2 + 1e-7f));
                double exact_idx = static_cast<double>(i) + delta;
                peak_times.push_back(exact_idx / target_fs_);
            }
        }
    }

    float rmssd = 0.0f;
    float sdnn = 0.0f;
    if (peak_times.size() >= 4) {
        std::vector<double> ibis;
        for (size_t i = 1; i < peak_times.size(); ++i) {
            double ibi_ms = (peak_times[i] - peak_times[i - 1]) * 1000.0;
            if (ibi_ms >= 300.0 && ibi_ms <= 1500.0) {
                ibis.push_back(ibi_ms);
            }
        }

        if (ibis.size() >= 3) {
            double ssq_diff = 0.0;
            for (size_t i = 1; i < ibis.size(); ++i) {
                double diff = ibis[i] - ibis[i - 1];
                ssq_diff += diff * diff;
            }
            rmssd = static_cast<float>(std::sqrt(ssq_diff / (ibis.size() - 1)));

            double mean_ibi = std::accumulate(ibis.begin(), ibis.end(), 0.0) / ibis.size();
            double var = 0.0;
            for (double ibi : ibis) {
                var += (ibi - mean_ibi) * (ibi - mean_ibi);
            }
            sdnn = static_cast<float>(std::sqrt(var / (ibis.size() - 1)));
        }
    }

    float sqi = std::min(100.0f, std::max(0.0f, (max_mag / 10.0f) * 80.0f + 15.0f));
    bool is_valid = (hr_bpm >= 45.0f && hr_bpm <= 190.0f && sqi >= 40.0f);

    metrics.heart_rate_bpm = hr_bpm;
    metrics.respiration_rate_rpm = resp_rpm;
    metrics.prv_rmssd_ms = rmssd;
    metrics.prv_sdnn_ms = sdnn;
    metrics.signal_quality_index = sqi;
    metrics.confidence = sqi / 100.0f;
    metrics.is_valid = is_valid;
    metrics.quality_category = (sqi >= 70.0f) ? "Optimal" : (sqi >= 40.0f ? "Acceptable" : "Degraded");

    return metrics;
}

} // namespace aurapulse
