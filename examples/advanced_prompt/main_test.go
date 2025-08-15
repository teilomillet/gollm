package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/assess"
	"github.com/teilomillet/gollm/presets"
)

func TestAdvancedPromptExample(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping advanced prompt test in short mode")
	}

	// Create client with normal configuration for the main test
	llm := initLLMClient(t, &gollm.Config{
		Temperature: 0.7,
		MaxTokens:   1000,
		Timeout:     45 * time.Second, // Longer timeout for normal operation
		MaxRetries:  3,
		RetryDelay:  time.Second,
	})

	test := assess.NewTest(t).
		WithProvider("openai", "gpt-4o-mini").
		WithConfig(&gollm.Config{
			Temperature: 0.7,
			MaxTokens:   1000,
			MaxRetries:  3,
			RetryDelay:  time.Second,
		})

	// Define reusable analysis directives
	balancedAnalysisDirectives := []string{
		"Consider technological, economic, social, and ethical implications",
		"Provide at least one potential positive and one potential negative outcome for each perspective",
		"Ensure the analysis is balanced and objective",
	}

	// Test balanced analysis template
	balancedAnalysisTemplate := gollm.NewPromptTemplate(
		"BalancedAnalysis",
		"Analyze a topic from multiple perspectives",
		`Analyze the following topic from multiple perspectives: {{.Topic}}

Please structure your response as a JSON object with the following format:
{
  "perspectives": [
    {
      "name": "Perspective Name",
      "implications": [
        "Positive implication",
        "Negative implication"
      ]
    }
  ],
  "summary": "A brief, balanced summary of the analysis"
}`,
		gollm.WithPromptOptions(
			gollm.WithDirectives(balancedAnalysisDirectives...),
			gollm.WithMaxLength(500),
		),
	)

	// Test multiple topics analysis
	topics := []string{
		"The impact of artificial intelligence on job markets",
		"The role of social media in modern democracy",
	}

	for _, topic := range topics {
		testName := "analyze_" + strings.ReplaceAll(topic, " ", "_")

		// Execute the template for this topic
		prompt, err := balancedAnalysisTemplate.Execute(map[string]any{
			"Topic": topic,
		})
		assert.NoError(t, err)

		test.AddCase(testName, prompt.String()).
			WithTimeout(45*time.Second).
			WithOption("max_tokens", 1000).
			Validate(func(response string) error {
				// Clean and parse JSON response
				cleanedJSON := cleanJSONResponse(response)
				var result AnalysisResult
				err := json.Unmarshal([]byte(cleanedJSON), &result)
				if err != nil {
					return fmt.Errorf("failed to parse JSON: %w, response: %s", err, cleanedJSON)
				}

				// Validate structure
				if len(result.Perspectives) == 0 {
					return errors.New("no perspectives found in response")
				}
				if result.Summary == "" {
					return errors.New("no summary found in response")
				}

				// Validate each perspective
				for _, p := range result.Perspectives {
					if p.Name == "" {
						return errors.New("perspective name is empty")
					}
					if len(p.Implications) < 2 { // At least one positive and one negative
						return fmt.Errorf("perspective %s has fewer than 2 implications", p.Name)
					}
				}

				// Test additional features on the response using our initialized client
				// Test summarization
				summary, err := presets.Summarize(context.Background(), llm, cleanedJSON, gollm.WithMaxLength(50))
				if err != nil {
					return fmt.Errorf("summarization failed: %w", err)
				}
				if summary == "" {
					return errors.New("empty summary response")
				}

				// Test chain of thought
				// keyPoints, err := presets.ChainOfThought(context.Background(), llm,
				// 	fmt.Sprintf("Extract 3 key points from this analysis:\n%s", cleanedJSON))
				// if err != nil {
				// 	return fmt.Errorf("chain of thought failed: %v", err)
				// }
				// if keyPoints == "" {
				// 	return fmt.Errorf("empty key points response")
				// }

				return nil
			})
	}

	ctx := context.Background()
	test.Run(ctx)
}

// Test the JSON response cleaning function
func TestCleanJSONResponse(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "clean json",
			input:    `{"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "json with markdown",
			input:    "```json\n{\"key\": \"value\"}\n```",
			expected: `{"key": "value"}`,
		},
		{
			name:     "json with whitespace",
			input:    "\n  {\"key\": \"value\"}  \n",
			expected: `{"key": "value"}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := cleanJSONResponse(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// initLLMClient creates a new LLM client with the given configuration
func initLLMClient(t *testing.T, config *gollm.Config) gollm.LLM {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetTemperature(config.Temperature),
		gollm.SetMaxTokens(config.MaxTokens),
		gollm.SetTimeout(config.Timeout),
		gollm.SetMaxRetries(config.MaxRetries),
		gollm.SetRetryDelay(config.RetryDelay),
	)
	if err != nil {
		t.Fatalf("Failed to create LLM: %v", err)
	}
	return llm
}

// TestQuestionAnswerRetryMechanism tests the retry mechanism when using QuestionAnswer
func TestQuestionAnswerRetryMechanism(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping question answer retry test in short mode")
	}

	// Create client with retry configuration to demonstrate retries
	llm := initLLMClient(t, &gollm.Config{
		Temperature: 0.7,
		MaxTokens:   1000,
		Timeout:     5 * time.Second, // Short timeout to force retries
		MaxRetries:  3,               // Will attempt 4 times total (initial + 3 retries)
		RetryDelay:  time.Second,     // Wait between retries
	})

	// Use a complex prompt that's likely to take longer than the timeout
	_, err := presets.QuestionAnswer(
		context.Background(),
		llm,
		"Please provide a detailed analysis of the impact of quantum computing on cryptography, including current limitations, potential breakthroughs, and implications for cybersecurity.",
	)

	// We expect the error to show multiple retry attempts
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "context deadline exceeded") ||
			strings.Contains(errStr, "failed to generate after") {
			t.Logf("Got expected error showing retry attempts: %v", err)
		} else {
			t.Errorf("Got unexpected error type: %v", err)
		}
	} else {
		t.Log("Request succeeded (unexpected in this test)")
	}
}
