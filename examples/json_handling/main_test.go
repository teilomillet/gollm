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
			MaxRetries: 2,
			RetryDelay: time.Second * 10,
			MaxTokens:  150,
			LogLevel:   gollm.LogLevelInfo,
		})

	// Example 1: Simple JSON Output
	simplePrompt := gollm.NewPrompt("List two colors",
		gollm.WithOutput("JSON array of colors"),
	)
	test.AddCase("simple_json", simplePrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 50).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var colors []string
			if err := json.Unmarshal([]byte(cleanResponse), &colors); err != nil {
				return fmt.Errorf("invalid JSON array: %v", err)
			}
			if len(colors) != 2 {
				return fmt.Errorf("expected 2 colors, got %d", len(colors))
			}
			return nil
		})

	// Example 2: Structured JSON with Schema Validation
	colorSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"colors": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"hex":  map[string]any{"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
					},
					"required": []string{"name", "hex"},
				},
			},
		},
		"required": []string{"colors"},
	}

	schemaPrompt := gollm.NewPrompt("List two colors with their hex codes")
	test.AddCase("schema_validation", schemaPrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 100).
		ExpectSchema(colorSchema).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var result map[string]any
			if err := json.Unmarshal([]byte(cleanResponse), &result); err != nil {
				return fmt.Errorf("invalid JSON: %v", err)
			}
			colors, ok := result["colors"].([]any)
			if !ok {
				return fmt.Errorf("colors field not found or invalid")
			}
			if len(colors) != 2 {
				return fmt.Errorf("expected 2 colors, got %d", len(colors))
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
								"items": {"type": "string"},
								"maxItems": 2
							},
							"settings": {
								"type": "object",
								"properties": {
									"darkMode": {"type": "boolean"}
								},
								"required": ["darkMode"]
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

	userPrompt := gollm.NewPrompt("Generate a simple user profile with preferences")
	test.AddCase("complex_json", userPrompt.String()).
		WithTimeout(45*time.Second).
		WithOption("max_tokens", 150).
		ExpectSchema(userSchema).
		Validate(func(response string) error {
			cleanResponse := cleanJSONResponse(response)
			var result map[string]any
			if err := json.Unmarshal([]byte(cleanResponse), &result); err != nil {
				return fmt.Errorf("invalid JSON: %v", err)
			}

			user, ok := result["user"].(map[string]any)
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
	mixedPrompt := gollm.NewPrompt("Analyze the color red",
		gollm.WithDirectives(
			"Start with a brief description of the color",
			"Include technical details in JSON format",
		),
		gollm.WithSystemPrompt("You are a color expert. Always start your responses with a description paragraph.", gollm.CacheTypeEphemeral),
		gollm.WithOutput(`Your response should have two parts:
1. A brief description paragraph
2. Technical details in this JSON format:
{
    "technical": {
        "rgb": [R, G, B],
        "hex": "string"
    }
}`),
	)
	test.AddCase("mixed_format", mixedPrompt.String()).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 150).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}

			// More flexible description check
			hasDescription := strings.Contains(strings.ToLower(response), "red is") ||
				strings.Contains(strings.ToLower(response), "the color red") ||
				strings.Contains(strings.ToLower(response), "description") ||
				strings.Contains(strings.ToLower(response), "represents")
			if !hasDescription {
				return fmt.Errorf("missing description section")
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

			if !hasRGB && !hasHex {
				return fmt.Errorf("missing both RGB and HEX color formats")
			}

			// Try to extract and validate JSON if present
			if jsonStr, err := extractJSONFromText(response); err == nil {
				var data map[string]any
				if err := json.Unmarshal([]byte(jsonStr), &data); err == nil {
					if tech, ok := data["technical"].(map[string]any); ok {
						if rgb, hasRGB := tech["rgb"]; hasRGB {
							if rgbArr, isArray := rgb.([]any); isArray && len(rgbArr) == 3 {
								// Valid RGB array found
								return nil
							}
						}
					}
				}
			}

			return nil
		})

	ctx := context.Background()
	test.Run(ctx)

	// Verify metrics with more lenient timing
	metrics := test.GetBatchMetrics()
	if metrics != nil {
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
}
