#include "aurapulse.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace aurapulse {

std::vector<float> SignalProcessor::extractCHROM(
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

    std::vector<float> S(N, 0.0f);

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

        std::vector<double> xs(l, 0.0), ys(l, 0.0);
        for (int i = 0; i < l; ++i) {
            double rn = rgb_series[m + i].r / mean_r;
            double gn = rgb_series[m + i].g / mean_g;
            double bn = rgb_series[m + i].b / mean_b;

            xs[i] = 3.0 * rn - 2.0 * gn;
            ys[i] = 1.5 * rn + gn - 1.5 * bn;
        }

        double std_x = 0.0, std_y = 0.0;
        double mx = std::accumulate(xs.begin(), xs.end(), 0.0) / l;
        double my = std::accumulate(ys.begin(), ys.end(), 0.0) / l;
        for (int i = 0; i < l; ++i) {
            std_x += (xs[i] - mx) * (xs[i] - mx);
            std_y += (ys[i] - my) * (ys[i] - my);
        }
        std_x = std::sqrt(std_x / (l - 1 > 0 ? l - 1 : 1));
        std_y = std::sqrt(std_y / (l - 1 > 0 ? l - 1 : 1));

        double alpha = (std_y > 1e-6) ? (std_x / std_y) : 0.0;

        for (int i = 0; i < l; ++i) {
            double s_val = xs[i] - alpha * ys[i];
            S[m + i] += static_cast<float>(s_val);
        }
    }

    return S;
}

} // namespace aurapulse
