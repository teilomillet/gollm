// Package fileutils provides file operations for the GoLLM library.
package llm

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
)

// ReadExamplesFromFile reads examples from a file (JSON or text)
func ReadExamplesFromFile(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			slog.Error("Failed to close file", slog.String("filename", filename), slog.Any("error", err))
		}
	}()

	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	data := make([]byte, stat.Size())
	_, err = file.Read(data)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	if strings.HasSuffix(filename, ".json") {
		return readJSONExamples(data)
	}

	return readTextExamples(data), nil
}

// readJSONExamples parses JSON file content into examples
func readJSONExamples(data []byte) ([]string, error) {
	var examples []string
	if err := json.Unmarshal(data, &examples); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}
	return examples, nil
}

// readTextExamples splits text file content into examples by newlines
func readTextExamples(data []byte) []string {
	content := string(data)
	lines := strings.Split(content, "\n")

	var examples []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			examples = append(examples, trimmed)
		}
	}
	return examples
}
