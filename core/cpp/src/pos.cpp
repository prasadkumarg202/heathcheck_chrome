#include "aurapulse.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace aurapulse {

std::vector<float> SignalProcessor::extractPOS(
    const std::vector<RGBPoint>& rgb_series,
    float fs,
    float window_len_s
) {
    size_t N = rgb_series.size();
    int l = static_cast<int>(std::round(window_len_s * fs));
    if (l < 4) l = 4;
    if (N < static_cast<size_t>(l)) {
        return std::vector<float>(N, 0.0f);
    }

    std::vector<float> H(N, 0.0f);

    for (size_t m = 0; m <= N - l; ++m) {
        double mean_r = 0.0, mean_g = 0.0, mean_b = 0.0;
        for (int i = 0; i < l; ++i) {
            mean_r += rgb_series[m + i].r;
            mean_g += rgb_series[m + i].g;
            mean_b += rgb_series[m + i].b;
        }
        mean_r = (mean_r / l > 1e-6) ? mean_r / l : 1.0;
        mean_g = (mean_g / l > 1e-6) ? mean_g / l : 1.0;
        mean_b = (mean_b / l > 1e-6) ? mean_b / l : 1.0;

        std::vector<double> s1(l, 0.0), s2(l, 0.0);
        for (int i = 0; i < l; ++i) {
            double rn = rgb_series[m + i].r / mean_r;
            double gn = rgb_series[m + i].g / mean_g;
            double bn = rgb_series[m + i].b / mean_b;

            s1[i] = gn - bn;
            s2[i] = gn + bn - 2.0 * rn;
        }

        double std1 = 0.0, std2 = 0.0;
        double m1 = std::accumulate(s1.begin(), s1.end(), 0.0) / l;
        double m2 = std::accumulate(s2.begin(), s2.end(), 0.0) / l;
        for (int i = 0; i < l; ++i) {
            std1 += (s1[i] - m1) * (s1[i] - m1);
            std2 += (s2[i] - m2) * (s2[i] - m2);
        }
        std1 = std::sqrt(std1 / (l - 1 > 0 ? l - 1 : 1));
        std2 = std::sqrt(std2 / (l - 1 > 0 ? l - 1 : 1));

        double alpha = (std2 > 1e-6) ? (std1 / std2) : 0.0;

        for (int i = 0; i < l; ++i) {
            double h_val = s1[i] + alpha * s2[i];
            H[m + i] += static_cast<float>(h_val);
        }
    }

    return H;
}

} // namespace aurapulse
