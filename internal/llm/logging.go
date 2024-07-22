// File: internal/llm/logging.go

package llm

import (
	"fmt"
	"log"
	"os"
)

// Logger defines the interface for logging within the LLM package
type Logger interface {
	Debug(msg string, keysAndValues ...interface{})
	Info(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
	Error(msg string, keysAndValues ...interface{})
}

// DefaultLogger is a simple implementation of the Logger interface
type DefaultLogger struct {
	logger *log.Logger
	level  LogLevel
}

type LogLevel int

const (
	DebugLevel LogLevel = iota
	InfoLevel
	WarnLevel
	ErrorLevel
)

// NewDefaultLogger creates a new DefaultLogger with the specified log level
func NewDefaultLogger(level string) Logger {
	logLevel := stringToLogLevel(level)
	return &DefaultLogger{
		logger: log.New(os.Stderr, "", log.LstdFlags),
		level:  logLevel,
	}
}

func (l *DefaultLogger) log(level LogLevel, msg string, keysAndValues ...interface{}) {
	if level >= l.level {
		l.logger.Printf("%s: %s %v", level, msg, keysAndValues)
	}
}

func (l *DefaultLogger) Debug(msg string, keysAndValues ...interface{}) {
	l.log(DebugLevel, msg, keysAndValues...)
}

func (l *DefaultLogger) Info(msg string, keysAndValues ...interface{}) {
	l.log(InfoLevel, msg, keysAndValues...)
}

func (l *DefaultLogger) Warn(msg string, keysAndValues ...interface{}) {
	l.log(WarnLevel, msg, keysAndValues...)
}

func (l *DefaultLogger) Error(msg string, keysAndValues ...interface{}) {
	l.log(ErrorLevel, msg, keysAndValues...)
}

func stringToLogLevel(level string) LogLevel {
	switch level {
	case "debug":
		return DebugLevel
	case "info":
		return InfoLevel
	case "warn":
		return WarnLevel
	case "error":
		return ErrorLevel
	default:
		return WarnLevel
	}
}

func (l LogLevel) String() string {
	switch l {
	case DebugLevel:
		return "DEBUG"
	case InfoLevel:
		return "INFO"
	case WarnLevel:
		return "WARN"
	case ErrorLevel:
		return "ERROR"
	default:
		return fmt.Sprintf("LogLevel(%d)", int(l))
	}
}
