package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/assess"
)

func TestCustomConfigExample(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping custom config test in short mode")
	}

	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: OPENAI_API_KEY not set")
	}

	test := assess.NewTest(t).
		WithProvider("openai", "gpt-4o-mini").
		WithConfig(&gollm.Config{
			Temperature: 0.7,
			MaxTokens:   150,
			MaxRetries:  3,
			RetryDelay:  time.Second * 2,
		})

	// Test custom configuration setup
	t.Run("custom_config_setup", func(t *testing.T) {
		config := gollm.NewConfig()
		gollm.SetProvider("openai")(config)
		gollm.SetModel("gpt-4o-mini")(config)
		gollm.SetTemperature(0.7)(config)
		gollm.SetMaxTokens(150)(config)
		gollm.SetTimeout(30 * time.Second)(config)
		gollm.SetMaxRetries(3)(config)
		gollm.SetRetryDelay(2 * time.Second)(config)
		gollm.SetLogLevel(gollm.LogLevelInfo)(config)
		gollm.SetAPIKey(apiKey)(config)

		assert.Equal(t, "openai", config.Provider)
		assert.Equal(t, "gpt-4o-mini", config.Model)
		assert.Equal(t, float64(0.7), config.Temperature)
		assert.Equal(t, 150, config.MaxTokens)
		assert.Equal(t, 30*time.Second, config.Timeout)
		assert.Equal(t, 3, config.MaxRetries)
		assert.Equal(t, 2*time.Second, config.RetryDelay)
		assert.Equal(t, gollm.LogLevelInfo, config.LogLevel)
	})

	// Test prompt template with analysis
	t.Run("prompt_template_analysis", func(t *testing.T) {
		analysisPrompt := gollm.NewPromptTemplate(
			"CustomAnalysis",
			"Analyze a given topic",
			"Analyze the following topic: {{.Topic}}",
			gollm.WithPromptOptions(
				gollm.WithDirectives(
					"Consider technological, economic, and social implications",
					"Provide at least one potential positive and one potential negative outcome",
					"Conclude with a balanced summary",
				),
				gollm.WithOutput("Analysis:"),
			),
		)

		// Test template execution
		prompt, err := analysisPrompt.Execute(map[string]any{
			"Topic": "The widespread adoption of artificial intelligence",
		})
		assert.NoError(t, err, "Should execute template without error")
		assert.Contains(t, prompt.String(), "artificial intelligence", "Prompt should contain the topic")
		assert.Contains(t, prompt.String(), "Analysis:", "Prompt should contain the output prefix")
	})

	// Test topic analysis with validation
	topics := []string{
		"The widespread adoption of artificial intelligence",
		"The implementation of a four-day work week",
		"The transition to renewable energy sources",
	}

	for _, topic := range topics {
		testName := fmt.Sprintf("topic_analysis_%s", strings.ReplaceAll(topic, " ", "_"))
		test.AddCase(testName, fmt.Sprintf(`Analyze the following topic: %s

Please structure your response in exactly this format:

1. IMPLICATIONS:
   - Technological implications: [your analysis]
   - Economic implications: [your analysis]
   - Social implications: [your analysis]

2. OUTCOMES:
   - Positive outcomes: [list at least one for each implication]
   - Negative outcomes: [list at least one for each implication]

3. SUMMARY:
   [Write a balanced summary of all points discussed above]

Your response MUST include all these sections with the exact headings shown.`, topic)).
			WithTimeout(45*time.Second).
			WithOption("max_tokens", 500).
			Validate(func(response string) error {
				// Check for implications section
				if !strings.Contains(strings.ToLower(response), "implications:") {
					return fmt.Errorf("response should contain implications analysis")
				}

				// Check for outcomes section with both positive and negative
				hasPositive := strings.Contains(strings.ToLower(response), "positive outcomes:")
				hasNegative := strings.Contains(strings.ToLower(response), "negative outcomes:")
				if !hasPositive || !hasNegative {
					return fmt.Errorf("response should contain both positive and negative outcomes")
				}

				// Check for summary section
				if !strings.Contains(strings.ToLower(response), "summary:") {
					return fmt.Errorf("response should contain a summary")
				}
				return nil
			})
	}

	ctx := context.Background()
	test.Run(ctx)
}

func TestCustomConfigErrorHandling(t *testing.T) {
	// Test invalid configuration
	t.Run("invalid_config", func(t *testing.T) {
		_, err := gollm.NewLLM(
			gollm.SetProvider("invalid"),
			gollm.SetModel("invalid"),
		)
		assert.Error(t, err, "Should fail with invalid provider")
	})

	// Test invalid prompt template
	t.Run("invalid_template", func(t *testing.T) {
		// Create a template with invalid syntax that will fail during parsing
		invalidTemplate := gollm.NewPromptTemplate(
			"test",
			"test",
			"{{.Topic", // Invalid template syntax - missing closing brace
		)
		_, err := invalidTemplate.Execute(map[string]any{
			"Topic": "test",
		})
		assert.Error(t, err, "Should fail with invalid template syntax")
	})
}
