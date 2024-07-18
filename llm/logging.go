// File: llm/logging.go

package llm

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	// Logger is the global logger for the LLM package
	Logger *zap.Logger
)

func init() {
	var err error
	config := zap.NewProductionConfig()
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	Logger, err = config.Build()
	if err != nil {
		panic(err)
	}
}

// SetLogLevel sets the log level for the LLM package
func SetLogLevel(level zapcore.Level) {
	newLogger := Logger.WithOptions(zap.IncreaseLevel(level))
	Logger = newLogger
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
		return zapcore.InfoLevel // Default to InfoLevel if unknown
	}
}
