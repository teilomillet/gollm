package basic_usage_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/guiperry/gollm_cerebras"
	"github.com/guiperry/gollm_cerebras/assess"
)

func TestBasicUsageExample(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping basic usage test in short mode")
	}

	test := assess.NewTest(t).
		WithProvider("openai", "gpt-4o-mini").
		WithConfig(&gollm.Config{
			MaxRetries: 3,
			RetryDelay: time.Second * 2,
		})

	// Test retry mechanism
	test.AddCase("retry_mechanism", "This is a test prompt to verify retry mechanism").
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 50).
		Validate(func(response string) error {
			metrics := test.GetBatchMetrics()
			if metrics != nil && len(metrics.Errors) > 0 {
				// If we have errors but still got a response, it means retries worked
				if response != "" {
					return nil
				}
				return fmt.Errorf("retry mechanism failed to recover from errors")
			}
			return nil
		})

	// Test basic prompt
	test.AddCase("basic_prompt", "Explain the concept of 'recursion' in programming.").
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 150).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}
			return nil
		})

	// Test advanced prompt with longer timeout and higher token limit
	test.AddCase("advanced_prompt", `Compare functional and object-oriented programming paradigms.

Please structure your response to include:
1. Key differences
2. Code examples
3. Strengths and weaknesses`).
		WithTimeout(60*time.Second).
		WithOption("max_tokens", 500).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}
			return nil
		})

	// Test context prompt
	test.AddCase("context_prompt", `Given this context:
The Internet of Things (IoT) refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data.

Task: Write a concise summary of the main points.`).
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 100).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}
			return nil
		})

	// Test Chain of Thought
	test.AddCase("chain_of_thought", `Explain the process of photosynthesis step by step.
Break down your reasoning into clear steps.`).
		WithTimeout(45*time.Second).
		WithOption("max_tokens", 300).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}
			if len(response) < 100 {
				return fmt.Errorf("response too short for a step-by-step explanation")
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
				assert.Empty(t, metrics.Errors[provider], "Should have no errors")
				// Only fail if response time is extremely high
				if latency > 120*time.Second {
					t.Errorf("Response time too high: %v", latency)
				}
			})
		}
	}
}

func TestJSONSchemaValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping JSON schema validation test in short mode")
	}

	// Create a new LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
	)
	assert.NoError(t, err, "Should create LLM instance")

	// Test JSON Schema generation
	schema, err := llm.GetPromptJSONSchema()
	assert.NoError(t, err, "Should get JSON schema")
	assert.NotEmpty(t, schema, "JSON schema should not be empty")

	// Test valid prompt
	validPrompt := gollm.NewPrompt("Test prompt")
	err = validPrompt.Validate()
	assert.NoError(t, err, "Valid prompt should pass validation")

	// Test invalid prompt
	invalidPrompt := gollm.NewPrompt("")
	err = invalidPrompt.Validate()
	assert.Error(t, err, "Empty prompt should fail validation")
}

func TestCustomConfigExample(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping custom config test in short mode")
	}

	test := assess.NewTest(t).
		WithProvider("openai", "gpt-4o-mini")

	// Test custom configuration
	test.AddCase("custom_config", `Analyze the impact of artificial intelligence.

Please structure your response to include:
1. Technological implications
2. Economic implications
3. Social implications
4. A balanced conclusion`).
		WithTimeout(45*time.Second).
		WithOption("max_tokens", 300).
		WithOption("temperature", 0.7).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
			}
			return nil
		})

	// Test configuration validation
	test.AddCase("config_validation", "Test configuration validation").
		WithTimeout(30*time.Second).
		WithOption("temperature", 0.7).
		WithOption("max_tokens", 150).
		Validate(func(response string) error {
			if response == "" {
				return fmt.Errorf("empty response")
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
				assert.Empty(t, metrics.Errors[provider], "Should have no errors")
				// Only fail if response time is extremely high
				if latency > 120*time.Second {
					t.Errorf("Response time too high: %v", latency)
				}
			})
		}
	}
}
