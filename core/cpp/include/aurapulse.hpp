#ifndef AURAPULSE_CORE_HPP
#define AURAPULSE_CORE_HPP

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace aurapulse {

struct RGBPoint {
    float r;
    float g;
    float b;
    double timestamp_s;
};

struct VitalMetrics {
    float heart_rate_bpm;
    float respiration_rate_rpm;
    float prv_rmssd_ms;
    float prv_sdnn_ms;
    float signal_quality_index;
    float confidence;
    bool is_valid;
    std::string quality_category;
    std::string rejection_reason;
};

class SignalProcessor {
public:
    static std::vector<float> butterworthBandpass(
        const std::vector<float>& input,
        float lowcut_hz,
        float highcut_hz,
        float fs,
        int order = 3
    );

    static std::vector<float> detrend(const std::vector<float>& input);
    static std::vector<float> extractPOS(const std::vector<RGBPoint>& rgb_series, float fs, float window_len_s = 1.6f);
    static std::vector<float> extractCHROM(const std::vector<RGBPoint>& rgb_series, float fs, float window_len_s = 1.6f);
};

class AuraPulseNativeEngine {
public:
    AuraPulseNativeEngine(float min_duration_s = 6.0f, float window_duration_s = 25.0f, float target_fs = 30.0f);
    ~AuraPulseNativeEngine() = default;

    void startSession();
    void pushFrameSignals(double timestamp_s, float fh_r, float fh_g, float fh_b,
                         float lc_r, float lc_g, float lc_b,
                         float rc_r, float rc_g, float rc_b);
    VitalMetrics computeVitals();

private:
    float min_duration_s_;
    float window_duration_s_;
    float target_fs_;
    std::vector<RGBPoint> forehead_buffer_;
    std::vector<RGBPoint> left_cheek_buffer_;
    std::vector<RGBPoint> right_cheek_buffer_;
    double session_start_s_;
};

} // namespace aurapulse

#endif // AURAPULSE_CORE_HPP
