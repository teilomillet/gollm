package main_test

import (
	"context"
	"encoding/json"
	"fmt"
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

// extractJSONFromText attempts to extract a JSON object from a text that may contain other content
func extractJSONFromText(text string) (string, error) {
	// Find the start of the JSON object
	start := strings.Index(text, "{")
	if start == -1 {
		return "", fmt.Errorf("no JSON object found")
	}

	// Find the matching closing brace
	depth := 1
	end := -1
	for i := start + 1; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i + 1
				break
			}
		}
		if end != -1 {
			break
		}
	}

	if end == -1 {
		return "", fmt.Errorf("no matching closing brace found")
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
			MaxRetries: 3,
			RetryDelay: time.Second * 2,
			MaxTokens:  200,
			LogLevel:   gollm.LogLevelInfo,
		})

	// Example 1: Simple JSON Output
	simplePrompt := gollm.NewPrompt("List three colors",
		gollm.WithOutput("JSON array of colors"),
	)
	test.AddCase("simple_json", simplePrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 100).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var colors []string
			if err := json.Unmarshal([]byte(cleanResponse), &colors); err != nil {
				return fmt.Errorf("invalid JSON array: %v", err)
			}
			if len(colors) != 3 {
				return fmt.Errorf("expected 3 colors, got %d", len(colors))
			}
			return nil
		})

	// Example 2: Structured JSON with Schema Validation
	colorSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"colors": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"name": map[string]interface{}{"type": "string"},
						"hex":  map[string]interface{}{"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
					},
					"required": []string{"name", "hex"},
				},
			},
		},
		"required": []string{"colors"},
	}

	schemaPrompt := gollm.NewPrompt("List three colors with their hex codes")
	test.AddCase("schema_validation", schemaPrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 150).
		ExpectSchema(colorSchema).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var result map[string]interface{}
			if err := json.Unmarshal([]byte(cleanResponse), &result); err != nil {
				return fmt.Errorf("invalid JSON: %v", err)
			}
			colors, ok := result["colors"].([]interface{})
			if !ok {
				return fmt.Errorf("colors field not found or invalid")
			}
			if len(colors) != 3 {
				return fmt.Errorf("expected 3 colors, got %d", len(colors))
			}
			return nil
		})

	// Example 3: Complex Nested JSON Structure
	userSchema := `{
		"type": "object",
		"properties": {
			"user": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"age": {"type": "integer", "minimum": 0},
					"preferences": {
						"type": "object",
						"properties": {
							"favoriteColors": {
								"type": "array",
								"items": {"type": "string"}
							},
							"settings": {
								"type": "object",
								"properties": {
									"darkMode": {"type": "boolean"},
									"notifications": {"type": "boolean"}
								},
								"required": ["darkMode", "notifications"]
							}
						},
						"required": ["favoriteColors", "settings"]
					}
				},
				"required": ["name", "age", "preferences"]
			}
		},
		"required": ["user"]
	}`

	userPrompt := gollm.NewPrompt("Generate a user profile with preferences")
	test.AddCase("complex_json", userPrompt.String()).
		WithTimeout(45*time.Second).
		WithOption("max_tokens", 200).
		ExpectSchema(userSchema).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var result map[string]interface{}
			if err := json.Unmarshal([]byte(cleanResponse), &result); err != nil {
				return fmt.Errorf("invalid JSON: %v", err)
			}

			user, ok := result["user"].(map[string]interface{})
			if !ok {
				return fmt.Errorf("user object not found or invalid")
			}

			required := []string{"name", "age", "preferences"}
			for _, field := range required {
				if _, exists := user[field]; !exists {
					return fmt.Errorf("required field '%s' missing", field)
				}
			}
			return nil
		})

	// Example 4: Mixed Format Response
	fmt.Println("\nExample 4: Mixed Format Response")
	fmt.Println("This example shows how to request specific sections in JSON format")

	mixedPrompt := gollm.NewPrompt("Analyze the color red",
		gollm.WithDirectives(
			"Provide a general description",
			"List color psychology effects",
			"Include common RGB and HEX values",
		),
		gollm.WithOutput(`Response should include a JSON object for technical details:
{
    "technical": {
        "rgb": [R, G, B],
        "hex": "string",
        "hsl": [H, S, L]
    }
}`),
	)
	test.AddCase("mixed_format", mixedPrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 200).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}

			// Check for required sections
			if !strings.Contains(strings.ToLower(response), "description") {
				return fmt.Errorf("missing general description section")
			}
			if !strings.Contains(strings.ToLower(response), "psychology") {
				return fmt.Errorf("missing color psychology section")
			}

			// Look for technical details in any format
			hasRGB := strings.Contains(strings.ToLower(response), "rgb") ||
				strings.Contains(response, "[255") ||
				strings.Contains(response, "255,") ||
				strings.Contains(response, "(255") ||
				strings.Contains(response, "255, 0, 0")
			hasHex := strings.Contains(strings.ToLower(response), "hex") ||
				strings.Contains(response, "#") ||
				strings.Contains(response, "FF0000") ||
				strings.Contains(response, "ff0000")

			// Only check for RGB and HEX as they are explicitly requested in directives
			var missingFormats []string
			if !hasRGB {
				missingFormats = append(missingFormats, "RGB")
			}
			if !hasHex {
				missingFormats = append(missingFormats, "HEX")
			}
			// HSL is optional as it's only in the template, not in directives

			if len(missingFormats) > 0 {
				return fmt.Errorf("missing color formats: %s", strings.Join(missingFormats, ", "))
			}

			return nil
		})

	ctx := context.Background()
	test.Run(ctx)

	// Verify metrics
	metrics := test.GetBatchMetrics()
	if metrics != nil {
		for provider, latency := range metrics.BatchTiming.ProviderLatency {
			t.Run(provider+"_metrics", func(t *testing.T) {
				assert.Greater(t, latency, time.Duration(0), "Should have response times")
				assert.Empty(t, metrics.Errors[provider], "Should have no errors")
				if latency > 120*time.Second {
					t.Errorf("Response time too high: %v", latency)
				}
			})
		}
	}
}
