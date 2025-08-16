package utils

import (
	"fmt"
	"log/slog"
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
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
	SetLevel(level LogLevel)
}

type DefaultLogger struct {
	logger *slog.Logger
	level  LogLevel
}

func slogLevel(level LogLevel) slog.Level {
	switch level {
	case LogLevelDebug:
		return slog.LevelDebug
	case LogLevelInfo:
		return slog.LevelInfo
	case LogLevelWarn:
		return slog.LevelWarn
	case LogLevelError:
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

func NewLogger(level LogLevel) *DefaultLogger {
	opts := &slog.HandlerOptions{
		Level: slogLevel(level),
	}
	return &DefaultLogger{
		logger: slog.New(slog.NewTextHandler(os.Stderr, opts)),
		level:  level,
	}
}

func (l *DefaultLogger) SetLevel(level LogLevel) {
	l.level = level
}

func (l *DefaultLogger) Debug(msg string, keysAndValues ...any) {
	if l.level >= LogLevelDebug {
		l.logger.Debug(msg, keysAndValues...)
	}
}

func (l *DefaultLogger) Info(msg string, keysAndValues ...any) {
	if l.level >= LogLevelInfo {
		l.logger.Info(msg, keysAndValues...)
	}
}

func (l *DefaultLogger) Warn(msg string, keysAndValues ...any) {
	if l.level >= LogLevelWarn {
		l.logger.Warn(msg, keysAndValues...)
	}
}

func (l *DefaultLogger) Error(msg string, keysAndValues ...any) {
	if l.level >= LogLevelError {
		l.logger.Error(msg, keysAndValues...)
	}
}

func (l *LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "DEBUG"}[*l]
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
