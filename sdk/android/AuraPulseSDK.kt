package com.aurapulse.sdk

data class AuraPulseVitals(
    val heartRateBpm: Float,
    val respirationRateRpm: Float,
    val prvRmssdMs: Float,
    val prvSdnnMs: Float,
    val signalQualityIndex: Float,
    val confidence: Float,
    val isValid: Boolean,
    val qualityCategory: String,
    val rejectionReason: String?
)

class AuraPulseNativeEngine {
    private var nativeHandle: Long = 0

    init {
        System.loadLibrary("aurapulse_jni")
        nativeHandle = nativeCreateEngine(6.0f, 25.0f, 30.0f)
    }

    fun startSession() {
        if (nativeHandle != 0L) {
            nativeStartSession(nativeHandle)
        }
    }

    fun pushFrame(
        timestampS: Double,
        fhR: Float, fhG: Float, fhB: Float,
        lcR: Float, lcG: Float, lcB: Float,
        rcR: Float, rcG: Float, rcB: Float
    ) {
        if (nativeHandle != 0L) {
            nativePushFrame(
                nativeHandle,
                timestampS,
                fhR, fhG, fhB,
                lcR, lcG, lcB,
                rcR, rcG, rcB
            )
        }
    }

    fun computeVitals(): AuraPulseVitals {
        if (nativeHandle == 0L) {
            return AuraPulseVitals(0f, 0f, 0f, 0f, 0f, 0f, false, "Invalid", "ENGINE_UNINITIALIZED")
        }
        return nativeComputeVitals(nativeHandle)
    }

    fun close() {
        if (nativeHandle != 0L) {
            nativeDestroyEngine(nativeHandle)
            nativeHandle = 0L
        }
    }

    private external fun nativeCreateEngine(minDurationS: Float, windowDurationS: Float, targetFs: Float): Long
    private external fun nativeStartSession(handle: Long)
    private external fun nativePushFrame(
        handle: Long,
        timestampS: Double,
        fhR: Float, fhG: Float, fhB: Float,
        lcR: Float, lcG: Float, lcB: Float,
        rcR: Float, rcG: Float, rcB: Float
    )
    private external fun nativeComputeVitals(handle: Long): AuraPulseVitals
    private external fun nativeDestroyEngine(handle: Long)
}
