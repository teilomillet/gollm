// File: tools/compare.go

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

type ComparisonResult[T any] struct {
	Provider string
	Model    string
	Response string
	Data     T
	Error    error
	Attempts int
}

func debugLog(config *config.Config, format string, args ...interface{}) {
	if config.LogLevel == utils.LogLevelDebug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

func cleanResponse(response string) string {
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}
	return strings.TrimSpace(response)
}

type ValidateFunc[T any] func(T) error

func CompareModels[T any](ctx context.Context, prompt string, validateFunc ValidateFunc[T], configs ...*config.Config) ([]ComparisonResult[T], error) {
	results := make([]ComparisonResult[T], len(configs))
	remainingConfigs := make([]*config.Config, len(configs))
	copy(remainingConfigs, configs)

	registry := providers.NewProviderRegistry()
	logger := utils.NewLogger(utils.LogLevelDebug)

	for attempt := 1; attempt <= 3; attempt++ {
		if len(remainingConfigs) == 0 {
			break
		}

		newRemainingConfigs := []*config.Config{}

		for _, config := range remainingConfigs {
			debugLog(config, "Attempting generation for %s %s (Attempt %d)", config.Provider, config.Model, attempt)

			llmInstance, err := llm.NewLLM(config, logger, registry)
			if err != nil {
				return nil, fmt.Errorf("failed to create LLM for %s: %w", config.Provider, err)
			}

			response, _, err := llmInstance.Generate(ctx, prompt)

			index := findConfigIndex(configs, config)
			results[index].Provider = config.Provider
			results[index].Model = config.Model
			results[index].Response = response
			results[index].Error = err
			results[index].Attempts = attempt

			if err != nil {
				debugLog(config, "Error generating response: %v", err)
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			debugLog(config, "Raw response received: %s", response)

			cleanedResponse := cleanResponse(response)
			debugLog(config, "Cleaned response: %s", cleanedResponse)

			results[index].Response = cleanedResponse

			var data T
			if err := json.Unmarshal([]byte(cleanedResponse), &data); err != nil {
				debugLog(config, "Invalid JSON: %v", err)
				results[index].Error = fmt.Errorf("invalid JSON: %w", err)
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			if err := validateFunc(data); err != nil {
				debugLog(config, "Validation failed: %v", err)
				results[index].Error = fmt.Errorf("validation failed: %w", err)
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			results[index].Data = data
			debugLog(config, "Valid response received for %s %s", config.Provider, config.Model)
		}

		remainingConfigs = newRemainingConfigs
	}

	return results, nil
}

func findConfigIndex(configs []*config.Config, target *config.Config) int {
	for i, config := range configs {
		if config.Provider == target.Provider && config.Model == target.Model {
			return i
		}
	}
	return -1
}

func AnalyzeComparisonResults[T any](results []ComparisonResult[T]) string {
	var analysis strings.Builder

	for _, result := range results {
		analysis.WriteString(strings.Repeat("-", 40) + "\n")
		analysis.WriteString(fmt.Sprintf("Provider: %s, Model: %s\n", result.Provider, result.Model))
		analysis.WriteString(fmt.Sprintf("Attempts: %d\n", result.Attempts))
		if result.Error != nil {
			analysis.WriteString(fmt.Sprintf("Error: %v\n", result.Error))
		} else {
			prettyJSON, err := json.MarshalIndent(result.Data, "", "  ")
			if err != nil {
				analysis.WriteString(fmt.Sprintf("Error prettifying JSON: %v\n", err))
				analysis.WriteString(fmt.Sprintf("Raw response: %s\n", result.Response))
			} else {
				analysis.WriteString(fmt.Sprintf("Response: %s\n", string(prettyJSON)))
			}
		}
		analysis.WriteString(strings.Repeat("-", 40) + "\n")
	}

	return analysis.String()
}

