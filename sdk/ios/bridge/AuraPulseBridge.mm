#import "AuraPulseBridge.h"
#import "aurapulse.hpp"

@implementation AuraPulseVitalsObjC
@end

@interface AuraPulseBridge () {
    std::unique_ptr<aurapulse::AuraPulseNativeEngine> _engine;
}
@end

@implementation AuraPulseBridge

- (instancetype)initWithMinDuration:(float)minDuration windowDuration:(float)windowDuration targetFs:(float)targetFs {
    self = [super init];
    if (self) {
        _engine = std::make_unique<aurapulse::AuraPulseNativeEngine>(minDuration, windowDuration, targetFs);
    }
    return self;
}

- (void)startSession {
    if (_engine) {
        _engine->startSession();
    }
}

- (void)pushFrameSignalsAtTimestamp:(double)timestamp
                                fhR:(float)fhR fhG:(float)fhG fhB:(float)fhB
                                lcR:(float)lcR lcG:(float)lcG lcB:(float)lcB
                                rcR:(float)rcR rcG:(float)rcG rcB:(float)rcB {
    if (_engine) {
        _engine->pushFrameSignals(timestamp, fhR, fhG, fhB, lcR, lcG, lcB, rcR, rcG, rcB);
    }
}

- (AuraPulseVitalsObjC *)computeVitals {
    AuraPulseVitalsObjC *res = [[AuraPulseVitalsObjC alloc] init];
    if (_engine) {
        auto v = _engine->computeVitals();
        res.heartRateBpm = v.heart_rate_bpm;
        res.respirationRateRpm = v.respiration_rate_rpm;
        res.prvRmssdMs = v.prv_rmssd_ms;
        res.prvSdnnMs = v.prv_sdnn_ms;
        res.signalQualityIndex = v.signal_quality_index;
        res.confidence = v.confidence;
        res.isValid = v.is_valid;
        res.qualityCategory = [NSString stringWithUTF8String:v.quality_category.c_str()];
        res.rejectionReason = [NSString stringWithUTF8String:v.rejection_reason.c_str()];
    }
    return res;
}

@end
