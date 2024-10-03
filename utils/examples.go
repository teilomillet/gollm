// File: utils/examples.go

package utils

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
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
		return nil, err
	}
	defer file.Close()

	var examples []string
	ext := strings.ToLower(filepath.Ext(filePath))

	switch ext {
	case ".txt":
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			examples = append(examples, scanner.Text())
		}
		if err := scanner.Err(); err != nil {
			return nil, err
		}
	case ".jsonl":
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			var example Example
			if err := json.Unmarshal(scanner.Bytes(), &example); err != nil {
				return nil, err
			}
			examples = append(examples, example.Content)
		}
		if err := scanner.Err(); err != nil {
			return nil, err
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
		rand.Shuffle(len(examples), func(i, j int) {
			examples[i], examples[j] = examples[j], examples[i]
		})
	case "desc":
		sort.Sort(sort.Reverse(sort.StringSlice(examples)))
	default: // "asc"
		sort.Strings(examples)
	}

	return examples[:n]
}
