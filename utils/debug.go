// Package utils provides utility functions for the GoLLM library.
package utils

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// DebugOptions contains configuration for debug output.
type DebugOptions struct {
	Enabled      bool
	OutputDir    string
	Verbose      bool
	SaveToFile   bool
	LogPrompts   bool
	LogResponses bool
}

// DebugManager handles debug output and logging for the optimization process.
type DebugManager struct {
	options   DebugOptions
	logger    Logger
	outputDir string
}

// NewDebugManager creates a new debug manager with the given options.
func NewDebugManager(options DebugOptions) *DebugManager {
	outputDir := options.OutputDir
	if outputDir == "" {
		outputDir = filepath.Join(".", "debug_output")
	}

	// Create output directory if saving to file is enabled
	if options.SaveToFile && options.Enabled {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			log.Printf("Warning: failed to create debug output directory: %v", err)
		}
	}

	return &DebugManager{
		options:   options,
		logger:    NewLogger(LogLevelDebug),
		outputDir: outputDir,
	}
}

// Log logs a debug message if debugging is enabled.
func (dm *DebugManager) Log(format string, args ...any) {
	if !dm.options.Enabled {
		return
	}

	message := fmt.Sprintf(format, args...)
	dm.logger.Debug(message)

	if dm.options.SaveToFile {
		dm.saveToFile("debug.log", message)
	}
}

// SaveIteration saves iteration data to a file if debugging is enabled.
func (dm *DebugManager) SaveIteration(iteration int, data any) {
	if !dm.options.Enabled || !dm.options.SaveToFile {
		return
	}

	filename := fmt.Sprintf("iteration_%d_%s.json", iteration, time.Now().Format("20060102_150405"))
	dm.saveToFile(filename, fmt.Sprintf("%+v", data))
}

// SavePrompt saves a prompt to a file if debugging is enabled.
func (dm *DebugManager) SavePrompt(name string, prompt string) {
	if !dm.options.Enabled || !dm.options.SaveToFile {
		return
	}

	filename := fmt.Sprintf("%s_%s.txt", name, time.Now().Format("20060102_150405"))
	dm.saveToFile(filename, prompt)
}

// saveToFile saves content to a file in the debug output directory.
func (dm *DebugManager) saveToFile(filename string, content string) {
	filepath := filepath.Join(dm.outputDir, filename)
	file, err := os.OpenFile(filepath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		dm.logger.Error("Failed to open file for debug output", "error", err, "file", filepath)
		return
	}
	defer file.Close()

	timestamp := time.Now().Format("2006-01-02 15:04:05")
	if _, err := fmt.Fprintf(file, "[%s] %s\n", timestamp, content); err != nil {
		dm.logger.Error("Failed to write debug output", "error", err, "file", filepath)
	}
}

// IsEnabled returns whether debugging is enabled.
func (dm *DebugManager) IsEnabled() bool {
	return dm.options.Enabled
}

// SetEnabled enables or disables debugging.
func (dm *DebugManager) SetEnabled(enabled bool) {
	dm.options.Enabled = enabled
}

// LogPrompt logs a prompt if prompt logging is enabled.
func (dm *DebugManager) LogPrompt(name string, prompt string) {
	if !dm.options.Enabled || !dm.options.LogPrompts {
		return
	}

	dm.Log("Prompt [%s]: %s", name, prompt)
	if dm.options.SaveToFile {
		dm.SavePrompt(name, prompt)
	}
}

// LogResponse logs a response if response logging is enabled.
func (dm *DebugManager) LogResponse(name string, response string) {
	if !dm.options.Enabled || !dm.options.LogResponses {
		return
	}

	dm.Log("Response [%s]: %s", name, response)
	if dm.options.SaveToFile {
		filename := fmt.Sprintf("response_%s_%s.txt", name, time.Now().Format("20060102_150405"))
		dm.saveToFile(filename, response)
	}
}
