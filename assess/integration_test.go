package assess

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
)

// TestProviderIntegration demonstrates comprehensive testing across providers
func TestProviderIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Create test runner with multiple providers
	test := NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  2,
			BatchTimeout: 5 * time.Minute,
		})

	if !test.HasAvailableProviders() {
		t.Skip("No providers available: missing API keys")
	}

	// System prompt for consistent context
	systemPrompt := `You are an AI assistant participating in a technical discussion about software development, 
	focusing on topics like system design, performance optimization, and best practices. 
	When discussing distributed systems, always mention the CAP theorem components (consistency, availability, partition tolerance) explicitly.
	Please provide detailed, technical responses.`

	// Test basic response validation
	t.Run("basic_validation", func(t *testing.T) {
		test.AddCase("technical_question", "What are the key principles of distributed systems? Please explain the CAP theorem.").
			WithSystemPrompt(systemPrompt).
			WithTimeout(30 * time.Second).
			Validate(ExpectMatches(`(?i)consistent|consistency`)).
			Validate(ExpectMatches(`(?i)availab(le|ility)`)).
			Validate(ExpectMatches(`(?i)partition.{0,20}toleran(t|ce)`))

		ctx := context.Background()
		test.RunBatch(ctx)
	})
}

// SOLIDResponse represents the expected structure for SOLID principles
type SOLIDResponse struct {
	Principles []string `json:"principles" validate:"required,min=1"`
	Examples   []string `json:"examples" validate:"required,min=1"`
}

// TestJSONValidation specifically tests structured output capabilities
func TestJSONValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping JSON validation test in short mode")
	}

	test := NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  2,
			BatchTimeout: 5 * time.Minute,
		})

	if !test.HasAvailableProviders() {
		t.Skip("No providers available: missing API keys")
	}

	// Generate schema from struct
	schema, err := gollm.GenerateJSONSchema(SOLIDResponse{})
	if err != nil {
		t.Fatalf("Failed to generate JSON schema: %v", err)
	}

	// Test JSON response with schema validation
	test.AddCase("json_solid_principles",
		fmt.Sprintf(`Please provide the SOLID principles and examples.
		Format your response as a JSON object that adheres to this schema:
		%s
		
		The principles array should contain each SOLID principle as a simple string.
		The examples array should contain practical examples as simple one-line strings.
		Do not include objects or complex types in the arrays.`, string(schema))).
		WithTimeout(60*time.Second).
		WithOption("max_tokens", 2000).
		ExpectSchema(schema).
		Validate(func(response string) error {
			var data SOLIDResponse
			if err := json.Unmarshal([]byte(response), &data); err != nil {
				return fmt.Errorf("invalid JSON response: %v", err)
			}

			if err := gollm.Validate(&data); err != nil {
				return fmt.Errorf("validation failed: %v", err)
			}

			return nil
		})

	ctx := context.Background()
	test.RunBatch(ctx)
}

// TestProviderSpecificFeatures demonstrates provider-specific testing capabilities
func TestProviderSpecificFeatures(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping provider-specific tests in short mode")
	}

	// Test Anthropic-specific features
	t.Run("anthropic_features", func(t *testing.T) {
		test := NewTest(t).
			WithProvider("anthropic", "claude-3-5-haiku-latest")

		test.AddCase("anthropic_test", "Analyze this code for security vulnerabilities").
			WithSystemPrompt("You are a security expert").
			WithTimeout(45 * time.Second)

		ctx := context.Background()
		test.Run(ctx)
	})

	// Test OpenAI-specific features
	t.Run("openai_features", func(t *testing.T) {
		test := NewTest(t).
			WithProvider("openai", "gpt-4o-mini")

		test.AddCase("openai_test", "Explain this code's complexity").
			WithSystemPrompt("You are a performance optimization expert").
			WithTimeout(45 * time.Second)

		ctx := context.Background()
		test.Run(ctx)
	})
}

// Add batch integration test
func TestBatchIntegration(t *testing.T) {
	test := NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  2,
			BatchTimeout: 2 * time.Minute,
		})

	if !test.HasAvailableProviders() {
		t.Skip("No providers available: missing API keys")
	}

	// Test cases with different validation requirements
	testCases := []struct {
		name      string
		query     string
		validate  ValidationFunc
		expectErr bool
	}{
		{
			name:     "simple_math",
			query:    "What is 2+2?",
			validate: ExpectContains("4"),
		},
		{
			name:     "code_generation",
			query:    "Write a function to calculate fibonacci numbers in Python",
			validate: ExpectContains("def fibonacci"),
		},
		{
			name:  "pattern_matching",
			query: "List three prime numbers between 1 and 20",
			validate: func(response string) error {
				// Define prime numbers up to 20
				primes := []string{"2", "3", "5", "7", "11", "13", "17", "19"}
				count := 0

				// Count how many prime numbers are in the response
				for _, prime := range primes {
					if strings.Contains(response, prime) {
						count++
					}
				}

				if count < 3 {
					return fmt.Errorf("expected at least 3 prime numbers, found %d", count)
				}
				return nil
			},
		},
	}

	// Add test cases with validations
	for _, tc := range testCases {
		test.AddCase(tc.name, tc.query).
			WithTimeout(30 * time.Second).
			Validate(tc.validate)
	}

	ctx := context.Background()
	test.RunBatch(ctx)

	metrics := test.GetBatchMetrics()

	// Verify batch execution completed
	assert.True(t, metrics.BatchTiming.TotalDuration > 0)
	assert.True(t, metrics.BatchTiming.EndTime.After(metrics.BatchTiming.StartTime))

	// Check provider latencies
	for provider, latency := range metrics.BatchTiming.ProviderLatency {
		t.Logf("Provider %s average latency: %v", provider, latency)
		assert.True(t, latency > 0)
	}

	// Verify error handling
	for provider, errs := range metrics.Errors {
		for _, err := range errs {
			t.Logf("Provider %s error: %v", provider, err)
		}
	}
}

// TestBatchCrossProvider tests cross-provider consistency
func TestBatchCrossProvider(t *testing.T) {
	test := NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  2,
			BatchTimeout: 2 * time.Minute,
		})

	if !test.HasAvailableProviders() {
		t.Skip("No providers available: missing API keys")
	}

	// System prompt to ensure consistent output format
	systemPrompt := `You are a helpful assistant. Always provide your answers in a clear, concise format.
For mathematical questions, just provide the numerical answer without explanation.
For yes/no questions, respond with only "yes" or "no".`

	// Test cases designed to compare responses across providers
	queries := []string{
		"What is 15 * 7?",
		"Is Paris the capital of France?",
		"What is the square root of 144?",
	}

	for _, query := range queries {
		test.AddCase("cross_provider_"+query[:10], query).
			WithSystemPrompt(systemPrompt).
			WithTimeout(20 * time.Second)
	}

	ctx := context.Background()
	test.RunBatch(ctx)

	metrics := test.GetBatchMetrics()

	// Compare provider performance
	t.Log("Provider Performance Comparison:")
	for provider, latency := range metrics.BatchTiming.ProviderLatency {
		t.Logf("%s average latency: %v", provider, latency)
	}

	// Log any errors
	for provider, errs := range metrics.Errors {
		for _, err := range errs {
			t.Logf("%s error: %v", provider, err)
		}
	}
}
