#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface AuraPulseVitalsObjC : NSObject
@property (nonatomic, assign) float heartRateBpm;
@property (nonatomic, assign) float respirationRateRpm;
@property (nonatomic, assign) float prvRmssdMs;
@property (nonatomic, assign) float prvSdnnMs;
@property (nonatomic, assign) float signalQualityIndex;
@property (nonatomic, assign) float confidence;
@property (nonatomic, assign) BOOL isValid;
@property (nonatomic, strong) NSString *qualityCategory;
@property (nonatomic, strong) NSString *rejectionReason;
@end

@interface AuraPulseBridge : NSObject
- (instancetype)initWithMinDuration:(float)minDuration windowDuration:(float)windowDuration targetFs:(float)targetFs;
- (void)startSession;
- (void)pushFrameSignalsAtTimestamp:(double)timestamp
                                fhR:(float)fhR fhG:(float)fhG fhB:(float)fhB
                                lcR:(float)lcR lcG:(float)lcG lcB:(float)lcB
                                rcR:(float)rcR rcG:(float)rcG rcB:(float)rcB;
- (AuraPulseVitalsObjC *)computeVitals;
@end

NS_ASSUME_NONNULL_END
