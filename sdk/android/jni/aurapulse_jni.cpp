#include <jni.h>
#include <string>
#include "aurapulse.hpp"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_aurapulse_sdk_AuraPulseNativeEngine_nativeCreateEngine(
    JNIEnv* env,
    jobject /* this */,
    jfloat minDurationS,
    jfloat windowDurationS,
    jfloat targetFs
) {
    auto* engine = new aurapulse::AuraPulseNativeEngine(minDurationS, windowDurationS, targetFs);
    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_com_aurapulse_sdk_AuraPulseNativeEngine_nativeStartSession(
    JNIEnv* env,
    jobject /* this */,
    jlong handle
) {
    auto* engine = reinterpret_cast<aurapulse::AuraPulseNativeEngine*>(handle);
    if (engine) engine->startSession();
}

JNIEXPORT void JNICALL
Java_com_aurapulse_sdk_AuraPulseNativeEngine_nativePushFrame(
    JNIEnv* env,
    jobject /* this */,
    jlong handle,
    jdouble timestampS,
    jfloat fhR, jfloat fhG, jfloat fhB,
    jfloat lcR, jfloat lcG, jfloat lcB,
    jfloat rcR, jfloat rcG, jfloat rcB
) {
    auto* engine = reinterpret_cast<aurapulse::AuraPulseNativeEngine*>(handle);
    if (engine) {
        engine->pushFrameSignals(timestampS, fhR, fhG, fhB, lcR, lcG, lcB, rcR, rcG, rcB);
    }
}

JNIEXPORT jobject JNICALL
Java_com_aurapulse_sdk_AuraPulseNativeEngine_nativeComputeVitals(
    JNIEnv* env,
    jobject /* this */,
    jlong handle
) {
    auto* engine = reinterpret_cast<aurapulse::AuraPulseNativeEngine*>(handle);
    aurapulse::VitalMetrics v;
    if (engine) {
        v = engine->computeVitals();
    }

    jclass cls = env->FindClass("com/aurapulse/sdk/AuraPulseVitals");
    jmethodID ctor = env->GetMethodID(
        cls,
        "<init>",
        "(FFFFFZLjava/lang/String;Ljava/lang/String;)V"
    );

    jstring categoryStr = env->NewStringUTF(v.quality_category.c_str());
    jstring reasonStr = env->NewStringUTF(v.rejection_reason.c_str());

    return env->NewObject(
        cls,
        ctor,
        v.heart_rate_bpm,
        v.respiration_rate_rpm,
        v.prv_rmssd_ms,
        v.prv_sdnn_ms,
        v.signal_quality_index,
        v.confidence,
        static_cast<jboolean>(v.is_valid),
        categoryStr,
        reasonStr
    );
}

JNIEXPORT void JNICALL
Java_com_aurapulse_sdk_AuraPulseNativeEngine_nativeDestroyEngine(
    JNIEnv* env,
    jobject /* this */,
    jlong handle
) {
    auto* engine = reinterpret_cast<aurapulse::AuraPulseNativeEngine*>(handle);
    delete engine;
}

}
