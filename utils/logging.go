package utils

import (
	"fmt"
	"log"
	"os"
	"strings"
)

type LogLevel int

const (
	LogLevelOff LogLevel = iota
	LogLevelError
	LogLevelWarn
	LogLevelInfo
	LogLevelDebug
)

type Logger interface {
	Debug(msg string, keysAndValues ...interface{})
	Info(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
	Error(msg string, keysAndValues ...interface{})
	SetLevel(level LogLevel)
}

type DefaultLogger struct {
	logger *log.Logger
	level  LogLevel
}

func NewLogger(level LogLevel) Logger {
	return &DefaultLogger{
		logger: log.New(os.Stderr, "", log.LstdFlags),
		level:  level,
	}
}

func (l *DefaultLogger) SetLevel(level LogLevel) {
	l.level = level
}

func (l *DefaultLogger) log(level LogLevel, msg string, keysAndValues ...interface{}) {
	if level <= l.level {
		l.logger.Printf("%s: %s %v", level, msg, keysAndValues)
	}
}

func (l *DefaultLogger) Debug(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelDebug, msg, keysAndValues...)
}

func (l *DefaultLogger) Info(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelInfo, msg, keysAndValues...)
}

func (l *DefaultLogger) Warn(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelWarn, msg, keysAndValues...)
}

func (l *DefaultLogger) Error(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelError, msg, keysAndValues...)
}

func (l LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "DEBUG"}[l]
}

func (l *LogLevel) UnmarshalText(text []byte) error {
	switch strings.ToUpper(string(text)) {
	case "OFF":
		*l = LogLevelOff
	case "ERROR":
		*l = LogLevelError
	case "WARN":
		*l = LogLevelWarn
	case "INFO":
		*l = LogLevelInfo
	case "DEBUG":
		*l = LogLevelDebug
	default:
		return fmt.Errorf("invalid log level: %s", string(text))
	}
	return nil
}
