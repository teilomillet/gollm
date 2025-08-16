// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

// Grade point values for letter grades (GPA scale)
const (
	GradeValueAPlus  = 4.3
	GradeValueA      = 4.0
	GradeValueAMinus = 3.7
	GradeValueBPlus  = 3.3
	GradeValueB      = 3.0
	GradeValueBMinus = 2.7
	GradeValueCPlus  = 2.3
	GradeValueC      = 2.0
	GradeValueCMinus = 1.7
	GradeValueDPlus  = 1.3
	GradeValueD      = 1.0
	GradeValueDMinus = 0.7
	GradeValueF      = 0.0
)

// Minimum grade value for optimization goal (A- or better)
const MinimumOptimizationGradeValue = GradeValueAMinus

// Default configuration values
const (
	DefaultThreshold             = 0.8
	DefaultMaxRetries            = 3
	DefaultMemorySize            = 2
	DefaultIterations            = 5
	DefaultGoalMetThreshold      = 0.9 // 90% or higher
	DefaultRetryDelaySeconds     = 2
	MaxRatingScale               = 20 // Scale for numerical ratings (0-20)
	MinValidationArraySize       = 5  // Minimum required items in validation arrays
	DefaultValidationReturnValue = 20 // Default return value for validation
	StringSliceMinSplitParts     = 2  // Minimum parts when splitting strings
	DefaultRateLimitSeconds      = 3  // Default rate limit interval
)

// Grade thresholds for converting numerical scores to letter grades (0-20 scale)
const (
	GradeThresholdAPlus  = 19
	GradeThresholdA      = 17
	GradeThresholdAMinus = 15
	GradeThresholdBPlus  = 13
	GradeThresholdB      = 11
	GradeThresholdBMinus = 9
	GradeThresholdCPlus  = 7
	GradeThresholdC      = 5
	GradeThresholdCMinus = 3
	GradeThresholdDPlus  = 2
	GradeThresholdD      = 1
)
