// Package debug provides debugging utilities for the GoLLM library.
package debug

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

// Manager handles debug output
type Manager struct {
	OutputDir string
	Enabled   bool
}

// NewDebugManager creates a new debug manager
func NewDebugManager(enabled bool, outputDir string) *Manager {
	return &Manager{
		Enabled:   enabled,
		OutputDir: outputDir,
	}
}

// LogRequestResponse logs request and response data for debugging
func (dm *Manager) LogRequestResponse(provider string, request, response any) {
	if !dm.Enabled {
		return
	}

	// Create debug output directory if it doesn't exist
	if err := os.MkdirAll(dm.OutputDir, 0o750); err != nil {
		slog.Warn("Failed to create debug output directory", slog.Any("error", err))
		return
	}

	// Create filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(dm.OutputDir, fmt.Sprintf("%s_%s.json", provider, timestamp))

	// Prepare debug data
	debugData := map[string]any{
		"timestamp": time.Now().Format(time.RFC3339),
		"provider":  provider,
		"request":   request,
		"response":  response,
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(debugData, "", "  ")
	if err != nil {
		slog.Warn("Failed to marshal debug data", slog.Any("error", err))
		return
	}

	// Write to file
	if err := os.WriteFile(filename, data, 0o600); err != nil {
		slog.Warn("Failed to write debug file", slog.Any("error", err))
		return
	}

	slog.Info("Debug data written", slog.String("file", filename))
}

// LogError logs error information for debugging
func (dm *Manager) LogError(provider string, err error, context map[string]any) {
	if !dm.Enabled {
		return
	}

	// Create debug output directory if it doesn't exist
	if err := os.MkdirAll(dm.OutputDir, 0o750); err != nil {
		slog.Warn("Failed to create debug output directory", slog.Any("error", err))
		return
	}

	// Create filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(dm.OutputDir, fmt.Sprintf("%s_error_%s.json", provider, timestamp))

	// Prepare error data
	errorData := map[string]any{
		"timestamp": time.Now().Format(time.RFC3339),
		"provider":  provider,
		"error":     err.Error(),
		"context":   context,
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(errorData, "", "  ")
	if err != nil {
		slog.Warn("Failed to marshal error data", slog.Any("error", err))
		return
	}

	// Write to file
	if err := os.WriteFile(filename, data, 0o600); err != nil {
		slog.Warn("Failed to write error file", slog.Any("error", err))
		return
	}

	slog.Info("Error data written", slog.String("file", filename))
}

// LogPrompt logs a prompt for debugging
func (dm *Manager) LogPrompt(category, prompt string) {
	if !dm.Enabled {
		return
	}
	slog.Debug("Prompt logged", slog.String("category", category), slog.String("prompt", prompt))
}

// LogResponse logs a response for debugging
func (dm *Manager) LogResponse(category, response string) {
	if !dm.Enabled {
		return
	}
	slog.Debug("Response logged", slog.String("category", category), slog.String("response", response))
}
