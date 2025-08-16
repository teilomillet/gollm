// Package logging provides logging utilities for the GoLLM library.
package logging

import (
	"fmt"
	"sync"
)

// MockLogger is a logger implementation for testing
type MockLogger struct {
	mu       sync.Mutex
	Messages []LogMessage
	level    LogLevel
}

// LogMessage represents a logged message
type LogMessage struct {
	Level   string
	Message string
	Args    []any
}

// NewMockLogger creates a new mock logger
func NewMockLogger() *MockLogger {
	return &MockLogger{
		Messages: []LogMessage{},
		level:    LogLevelDebug,
	}
}

// Debug logs a debug message
func (m *MockLogger) Debug(msg string, args ...any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.level <= LogLevelDebug {
		m.Messages = append(m.Messages, LogMessage{
			Level:   "DEBUG",
			Message: msg,
			Args:    args,
		})
	}
}

// Info logs an info message
func (m *MockLogger) Info(msg string, args ...any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.level <= LogLevelInfo {
		m.Messages = append(m.Messages, LogMessage{
			Level:   "INFO",
			Message: msg,
			Args:    args,
		})
	}
}

// Warn logs a warning message
func (m *MockLogger) Warn(msg string, args ...any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.level <= LogLevelWarn {
		m.Messages = append(m.Messages, LogMessage{
			Level:   "WARN",
			Message: msg,
			Args:    args,
		})
	}
}

// Error logs an error message
func (m *MockLogger) Error(msg string, args ...any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.level <= LogLevelError {
		m.Messages = append(m.Messages, LogMessage{
			Level:   "ERROR",
			Message: msg,
			Args:    args,
		})
	}
}

// SetLevel sets the logging level
func (m *MockLogger) SetLevel(level LogLevel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.level = level
}

// GetMessages returns all logged messages
func (m *MockLogger) GetMessages() []LogMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]LogMessage{}, m.Messages...)
}

// Clear clears all logged messages
func (m *MockLogger) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Messages = []LogMessage{}
}

// HasMessage checks if a message with the given text was logged
func (m *MockLogger) HasMessage(text string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, msg := range m.Messages {
		if msg.Message == text {
			return true
		}
	}
	return false
}

// String returns a string representation of all messages
func (m *MockLogger) String() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var result string
	for _, msg := range m.Messages {
		result += fmt.Sprintf("[%s] %s %v\n", msg.Level, msg.Message, msg.Args)
	}
	return result
}
