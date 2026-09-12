/**
 * AuraPulse TypeScript Web SDK
 * High-performance on-device / edge rPPG client with zero third-party cloud dependencies.
 */

export interface AuraPulseVitals {
  heartRateBpm: number | null;
  respirationRateRpm: number | null;
  prvRmssdMs: number | null;
  prvSdnnMs: number | null;
  stressIndex: number | null;
  signalQualityIndex: number;
  qualityCategory: 'Optimal' | 'Acceptable' | 'Degraded' | 'Invalid';
  confidence: number;
  isValid: boolean;
  reason: string;
}

export interface RGBPoint {
  r: number;
  g: number;
  b: number;
  timestampS: number;
}

export class AuraPulseClient {
  private serverUrl: string;
  private minDurationS: number;
  private targetFs: number;

  constructor(serverUrl: string = '', minDurationS: number = 6.0, targetFs: number = 30.0) {
    this.serverUrl = serverUrl;
    this.minDurationS = minDurationS;
    this.targetFs = targetFs;
  }

  public static sampleBoxMeanRGB(
    ctx: CanvasRenderingContext2D,
    box: [number, number, number, number]
  ): [number, number, number] {
    const [x, y, w, h] = box.map(Math.round);
    if (w <= 2 || h <= 2) return [0, 0, 0];

    const imgData = ctx.getImageData(x, y, w, h);
    const data = imgData.data;
    let sumR = 0, sumG = 0, sumB = 0;
    const step = 4;
    const totalPixels = data.length / 4;

    for (let i = 0; i < data.length; i += step) {
      sumR += data[i];
      sumG += data[i + 1];
      sumB += data[i + 2];
    }

    return [sumR / totalPixels, sumG / totalPixels, sumB / totalPixels];
  }

  public async pushTelemetry(
    timestampS: number,
    foreheadRGB: [number, number, number],
    leftCheekRGB: [number, number, number],
    rightCheekRGB: [number, number, number]
  ): Promise<AuraPulseVitals> {
    const payload = {
      timestamp: timestampS,
      forehead: foreheadRGB,
      left_cheek: leftCheekRGB,
      right_cheek: rightCheekRGB
    };

    const response = await fetch(${this.serverUrl} + '/api/push_signals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error('AuraPulse server error code: ' + response.status);
    }

    const data = await response.json();
    return {
      heartRateBpm: data.heart_rate,
      respirationRateRpm: data.respiration_rate,
      prvRmssdMs: data.pulse_rate_variability ? data.pulse_rate_variability.rmssd : null,
      prvSdnnMs: data.pulse_rate_variability ? data.pulse_rate_variability.sdnn : null,
      stressIndex: data.stress_index,
      signalQualityIndex: data.signal_quality || 0,
      qualityCategory: data.quality_category || 'Invalid',
      confidence: (data.signal_quality || 0) / 100.0,
      isValid: data.status === 'SUCCESS',
      reason: data.reason || ''
    };
  }
}
