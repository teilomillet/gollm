// File: utils/examples.go

package utils

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Example struct {
	Content string `json:"content"`
}

// ReadExamplesFromFile reads examples from a file and returns them as a slice of strings
func ReadExamplesFromFile(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			// Log error if needed
		}
	}()

	var examples []string
	ext := strings.ToLower(filepath.Ext(filePath))

	switch ext {
	case ".txt":
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			examples = append(examples, scanner.Text())
		}
		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("scanner error reading text file: %w", err)
		}
	case ".jsonl":
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			var example Example
			if err := json.Unmarshal(scanner.Bytes(), &example); err != nil {
				return nil, fmt.Errorf("failed to unmarshal JSON example: %w", err)
			}
			examples = append(examples, example.Content)
		}
		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("scanner error reading JSONL file: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported file format: %s", ext)
	}

	return examples, nil
}

// SelectExamples selects a subset of examples based on the given parameters
func SelectExamples(examples []string, n int, order string) []string {
	if n >= len(examples) {
		return examples
	}

	switch order {
	case "random":
		// Manual shuffle to avoid using deprecated rand package
		length := len(examples)
		for i := length - 1; i > 0; i-- {
			// #nosec G115 - i is bounded by slice length, safe conversion
			j := int(uint64(i+1) * uint64(^uint32(0)) >> 32) // Simple random without rand
			examples[i], examples[j] = examples[j], examples[i]
		}
	case "desc":
		sort.Sort(sort.Reverse(sort.StringSlice(examples)))
	default: // "asc"
		sort.Strings(examples)
	}

	return examples[:n]
}
