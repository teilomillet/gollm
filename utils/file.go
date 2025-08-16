// Package utils provides utility functions for the GoLLM library.
package utils

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ReadExamplesFromFile reads example strings from a file.
// Supports .txt files (one example per line) and .jsonl files (JSON Lines format).
func ReadExamplesFromFile(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var examples []string
	scanner := bufio.NewScanner(file)

	if strings.HasSuffix(filename, ".jsonl") {
		// Read JSON Lines format
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			var data map[string]any
			if err := json.Unmarshal([]byte(line), &data); err != nil {
				return nil, fmt.Errorf("failed to parse JSON line: %w", err)
			}
			// Try to extract example text from common fields
			if text, ok := data["text"].(string); ok {
				examples = append(examples, text)
			} else if example, ok := data["example"].(string); ok {
				examples = append(examples, example)
			} else {
				// If no specific field, use the entire JSON as string
				examples = append(examples, line)
			}
		}
	} else {
		// Read plain text format (one example per line)
		for scanner.Scan() {
			line := scanner.Text()
			if line != "" {
				examples = append(examples, line)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return examples, nil
}
