// File: internal/llm/logging.go

package llm

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"sync"
)

var (
	// Logger is the global logger for the LLM package
	Logger    *zap.Logger
	once      sync.Once
	isVerbose bool
)

func init() {
	// Initialize with default logger (warn level)
	initLogger(zapcore.WarnLevel)
}

// SetLogLevel sets the log level for the LLM package
func SetLogLevel(level zapcore.Level) {
	once.Do(func() {
		initLogger(level)
	})
	isVerbose = level <= zapcore.InfoLevel
}

// LogLevelFromString converts a string log level to zapcore.Level
func LogLevelFromString(level string) zapcore.Level {
	switch level {
	case "debug":
		return zapcore.DebugLevel
	case "info":
		return zapcore.InfoLevel
	case "warn":
		return zapcore.WarnLevel
	case "error":
		return zapcore.ErrorLevel
	default:
		return zapcore.WarnLevel // Default to WarnLevel if unknown
	}
}

// InitLogging initializes the logger with the specified log level
func InitLogging(logLevel string) {
	level := LogLevelFromString(logLevel)
	SetLogLevel(level)
}

// initLogger is a helper function to initialize the logger
func initLogger(level zapcore.Level) {
	config := zap.NewProductionConfig()
	config.Level = zap.NewAtomicLevelAt(level)
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	var err error
	Logger, err = config.Build()
	if err != nil {
		panic(err)
	}
}

// IsVerbose returns true if verbose logging is enabled (info or debug level)
func IsVerbose() bool {
	return isVerbose
}
