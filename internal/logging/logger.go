// Package logging provides logging utilities for the GoLLM library.
package logging

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
)

// LogLevel represents the logging level
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
)

// Logger interface for the GoLLM library
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
	SetLevel(level LogLevel)
}

// DefaultLogger is the default logger implementation
type DefaultLogger struct {
	logger *slog.Logger
	level  LogLevel
}

// NewLogger creates a new logger with the specified level
func NewLogger(level LogLevel) *DefaultLogger {
	opts := &slog.HandlerOptions{
		Level: toSlogLevel(level),
	}
	handler := slog.NewTextHandler(os.Stdout, opts)
	return &DefaultLogger{
		logger: slog.New(handler),
		level:  level,
	}
}

func toSlogLevel(level LogLevel) slog.Level {
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

// Debug logs a debug message
func (l *DefaultLogger) Debug(msg string, args ...any) {
	if l.level <= LogLevelDebug {
		l.logger.Debug(msg, parseArgs(args)...)
	}
}

// Info logs an info message
func (l *DefaultLogger) Info(msg string, args ...any) {
	if l.level <= LogLevelInfo {
		l.logger.Info(msg, parseArgs(args)...)
	}
}

// Warn logs a warning message
func (l *DefaultLogger) Warn(msg string, args ...any) {
	if l.level <= LogLevelWarn {
		l.logger.Warn(msg, parseArgs(args)...)
	}
}

// Error logs an error message
func (l *DefaultLogger) Error(msg string, args ...any) {
	if l.level <= LogLevelError {
		l.logger.Error(msg, parseArgs(args)...)
	}
}

// SetLevel sets the logging level
func (l *DefaultLogger) SetLevel(level LogLevel) {
	l.level = level
	opts := &slog.HandlerOptions{
		Level: toSlogLevel(level),
	}
	handler := slog.NewTextHandler(os.Stdout, opts)
	l.logger = slog.New(handler)
}

// parseArgs converts args to slog attributes
func parseArgs(args []any) []any {
	if len(args) == 0 {
		return nil
	}

	// If first arg is a string starting with %, treat it as fmt format
	if len(args) > 0 {
		if format, ok := args[0].(string); ok && strings.HasPrefix(format, "%") {
			if len(args) > 1 {
				formatted := fmt.Sprintf(format, args[1:]...)
				return []any{slog.String("message", formatted)}
			}
		}
	}

	// Otherwise, return as is for slog to handle
	return args
}
