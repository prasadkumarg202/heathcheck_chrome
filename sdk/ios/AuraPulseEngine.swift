import Foundation

public struct AuraPulseMeasurement {
    public let heartRateBpm: Float
    public let respirationRateRpm: Float
    public let prvRmssdMs: Float
    public let prvSdnnMs: Float
    public let signalQualityIndex: Float
    public let confidence: Float
    public let isValid: Bool
    public let qualityCategory: String
    public let rejectionReason: String
}

public class AuraPulseEngine {
    private let bridge: AuraPulseBridge

    public init(minDurationS: Float = 6.0, windowDurationS: Float = 25.0, targetFs: Float = 30.0) {
        self.bridge = AuraPulseBridge(minDuration: minDurationS, windowDuration: windowDurationS, targetFs: targetFs)
    }

    public func startSession() {
        bridge.startSession()
    }

    public func pushFrame(
        timestampS: Double,
        fhR: Float, fhG: Float, fhB: Float,
        lcR: Float, lcG: Float, lcB: Float,
        rcR: Float, rcG: Float, rcB: Float
    ) {
        bridge.pushFrameSignals(atTimestamp: timestampS,
                                fhR: fhR, fhG: fhB, fhB: fhB,
                                lcR: lcR, lcG: lcB, lcB: lcB,
                                rcR: rcR, rcG: rcB, rcB: rcB)
    }

    public func computeVitals() -> AuraPulseMeasurement {
        let v = bridge.computeVitals()
        return AuraPulseMeasurement(
            heartRateBpm: v.heartRateBpm,
            respirationRateRpm: v.respirationRateRpm,
            prvRmssdMs: v.prvRmssdMs,
            prvSdnnMs: v.prvSdnnMs,
            signalQualityIndex: v.signalQualityIndex,
            confidence: v.confidence,
            isValid: v.isValid,
            qualityCategory: v.qualityCategory,
            rejectionReason: v.rejectionReason
        )
    }
}
