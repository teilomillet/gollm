package main_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/assess"
)

// cleanJSONResponse removes markdown code block delimiters and trims whitespace
func cleanJSONResponse(response string) string {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	return strings.TrimSpace(response)
}

// validateTestMetrics validates test metrics with lenient timing
func validateTestMetrics(t *testing.T, test *assess.TestRunner) {
	metrics := test.GetBatchMetrics()
	if metrics == nil {
		return
	}

	for provider, latency := range metrics.BatchTiming.ProviderLatency {
		t.Run(provider+"_metrics", func(t *testing.T) {
			assert.Greater(t, latency, time.Duration(0), "Should have response times")
			// Only log errors instead of failing
			if len(metrics.Errors[provider]) > 0 {
				t.Logf("Provider %s had errors: %v", provider, metrics.Errors[provider])
			}
			// Only fail if response time is extremely high
			if latency > 120*time.Second {
				t.Errorf("Response time too high: %v", latency)
			}
		})
	}
}

// extractJSONFromText attempts to extract a JSON object from a text that may contain other content
func extractJSONFromText(text string) (string, error) {
	// Find the start of the JSON object
	start := strings.Index(text, "{")
	if start == -1 {
		return "", errors.New("no JSON object found")
	}

	// Find the matching closing brace
	depth := 1
	end := -1
outer:
	for i := start + 1; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i + 1
				break outer
			}
		}
	}

	if end == -1 {
		return "", errors.New("no matching closing brace found")
	}

	jsonStr := text[start:end]
	return cleanJSONResponse(jsonStr), nil
}

func TestJSONHandlingExamples(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping JSON handling test in short mode")
	}

	test := assess.NewTest(t).
		WithProvider("openai", "gpt-4o-mini").
		WithConfig(&gollm.Config{
			MaxRetries: 2,
			RetryDelay: time.Second * 10,
			MaxTokens:  150,
			LogLevel:   gollm.LogLevelInfo,
		})

	// Add all test cases using helper functions
	addSimpleJSONTest(test)
	addSchemaValidationTest(test)
	addComplexJSONTest(test)
	addMixedFormatTest(test)

	ctx := context.Background()
	test.Run(ctx)

	validateTestMetrics(t, test)
}
