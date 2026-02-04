package config_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// testLogger captures log messages for testing
type testLogger struct {
	messages []string
}

func (l *testLogger) Debug(msg string, keysAndValues ...interface{}) {
	l.messages = append(l.messages, "DEBUG: "+msg)
}

func (l *testLogger) Info(msg string, keysAndValues ...interface{}) {
	l.messages = append(l.messages, "INFO: "+msg)
}

func (l *testLogger) Warn(msg string, keysAndValues ...interface{}) {
	l.messages = append(l.messages, "WARN: "+msg)
}

func (l *testLogger) Error(msg string, keysAndValues ...interface{}) {
	l.messages = append(l.messages, "ERROR: "+msg)
}

func (l *testLogger) SetLevel(level utils.LogLevel) {}

// TestSetLogger verifies that a custom logger can be set via config option.
// This addresses GitHub issue #40.
func TestSetLogger(t *testing.T) {
	customLogger := &testLogger{}

	cfg := config.NewConfig()
	config.ApplyOptions(cfg, config.SetLogger(customLogger))

	assert.Equal(t, customLogger, cfg.Logger, "Custom logger should be set in config")
}

// TestNopLogger verifies that NopLogger discards all output without error.
func TestNopLogger(t *testing.T) {
	logger := utils.NewNopLogger()

	// These should not panic or produce any output
	logger.Debug("debug message")
	logger.Info("info message")
	logger.Warn("warn message")
	logger.Error("error message")
	logger.SetLevel(utils.LogLevelDebug)
}

// TestLoggerInterface verifies that custom loggers implement the Logger interface.
func TestLoggerInterface(t *testing.T) {
	var _ utils.Logger = &testLogger{}
	var _ utils.Logger = utils.NewNopLogger()
	var _ utils.Logger = utils.NewLogger(utils.LogLevelInfo)
}
