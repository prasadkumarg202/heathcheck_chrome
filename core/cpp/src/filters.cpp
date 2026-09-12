#include "aurapulse.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurapulse {

std::vector<float> SignalProcessor::detrend(const std::vector<float>& input) {
    size_t n = input.size();
    if (n < 2) return input;

    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_x += i;
        sum_y += input[i];
        sum_xy += i * input[i];
        sum_xx += i * i;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    std::vector<float> detrended(n);
    for (size_t i = 0; i < n; ++i) {
        detrended[i] = static_cast<float>(input[i] - (slope * i + intercept));
    }
    return detrended;
}

std::vector<float> SignalProcessor::butterworthBandpass(
    const std::vector<float>& input,
    float lowcut_hz,
    float highcut_hz,
    float fs,
    int order
) {
    size_t n = input.size();
    if (n < 6) return input;

    double w1 = 2.0 * fs * std::tan(M_PI * lowcut_hz / fs);
    double w2 = 2.0 * fs * std::tan(M_PI * highcut_hz / fs);
    double bw = w2 - w1;
    double w0 = std::sqrt(w1 * w2);

    double Q = w0 / (bw > 1e-6 ? bw : 1e-6);
    double k = std::tan(M_PI * (lowcut_hz + highcut_hz) / (2.0 * fs));
    double norm = 1.0 + k / Q + k * k;

    double b0 = (k / Q) / norm;
    double b1 = 0.0;
    double b2 = -b0;
    double a1 = 2.0 * (k * k - 1.0) / norm;
    double a2 = (1.0 - k / Q + k * k) / norm;

    std::vector<float> forward(n, 0.0f);
    double x1 = input[0], x2 = input[0];
    double y1 = 0.0, y2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double x0 = input[i];
        double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        forward[i] = static_cast<float>(y0);
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }

    std::vector<float> output(n, 0.0f);
    x1 = forward[n - 1]; x2 = forward[n - 1];
    y1 = 0.0; y2 = 0.0;

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double x0 = forward[i];
        double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        output[i] = static_cast<float>(y0);
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }

    return output;
}

} // namespace aurapulse
