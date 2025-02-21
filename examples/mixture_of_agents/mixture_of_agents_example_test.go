package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mauza/gollm"
	"github.com/mauza/gollm/assess"
	"github.com/stretchr/testify/assert"
)

func TestMixtureOfAgents(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping mixture of agents test in short mode")
	}

	// Check for required API keys
	openaiKey := os.Getenv("OPENAI_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	if openaiKey == "" || anthropicKey == "" {
		t.Fatal("Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set for this test")
	}

	// Create test runner with multiple providers
	test := assess.NewTest(t).
		WithProviders(map[string]string{
			"openai":    "gpt-4o-mini",
			"anthropic": "claude-3-5-haiku-latest",
		}).
		WithConfig(&gollm.Config{
			MaxRetries: 3,
			RetryDelay: 2 * time.Second,
			LogLevel:   gollm.LogLevelInfo,
		}).
		WithBatchConfig(assess.BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  2,
			BatchTimeout: 5 * time.Minute,
		})

	// Test MOA configuration
	t.Run("moa_config", func(t *testing.T) {
		// Create base config with common settings
		baseConfig := &gollm.Config{
			MaxTokens:  1024,
			MaxRetries: 3,
			RetryDelay: 2 * time.Second,
			LogLevel:   gollm.LogLevelInfo,
			APIKeys: map[string]string{
				"openai":    openaiKey,
				"anthropic": anthropicKey,
			},
		}

		// Configure MOA models
		moaConfig := gollm.MOAConfig{
			Iterations: 2,
			Models: []gollm.ConfigOption{
				// Group all options for the first model together
				func(cfg *gollm.Config) {
					gollm.SetProvider("openai")(cfg)
					gollm.SetModel("gpt-4o-mini")(cfg)
					gollm.SetAPIKey(openaiKey)(cfg)
					gollm.SetMaxTokens(baseConfig.MaxTokens)(cfg)
					gollm.SetMaxRetries(baseConfig.MaxRetries)(cfg)
					gollm.SetRetryDelay(baseConfig.RetryDelay)(cfg)
					gollm.SetLogLevel(baseConfig.LogLevel)(cfg)
					gollm.SetTimeout(60 * time.Second)(cfg)
					gollm.SetExtraHeaders(map[string]string{})(cfg)
				},
			},
			MaxParallel:  2,
			AgentTimeout: 60 * time.Second,
		}

		// Configure aggregator with complete configuration
		aggregatorOpts := []gollm.ConfigOption{
			func(cfg *gollm.Config) {
				gollm.SetProvider("anthropic")(cfg)
				gollm.SetModel("claude-3-5-haiku-latest")(cfg)
				gollm.SetAPIKey(anthropicKey)(cfg)
				gollm.SetTemperature(0.7)(cfg)
				gollm.SetMaxTokens(baseConfig.MaxTokens)(cfg)
				gollm.SetTimeout(90 * time.Second)(cfg)
				gollm.SetMaxRetries(baseConfig.MaxRetries)(cfg)
				gollm.SetRetryDelay(baseConfig.RetryDelay)(cfg)
				gollm.SetLogLevel(baseConfig.LogLevel)(cfg)
				gollm.SetExtraHeaders(map[string]string{})(cfg)
			},
		}

		// Create a new MOA instance with both configurations
		moa, err := gollm.NewMOA(moaConfig, aggregatorOpts...)
		if !assert.NoError(t, err, "Should create MOA instance") {
			t.FailNow()
		}
		if !assert.NotNil(t, moa, "MOA instance should not be nil") {
			t.FailNow()
		}
		assert.Equal(t, 2, moa.Config.Iterations, "Should have correct number of iterations")
		assert.Equal(t, 2, moa.Config.MaxParallel, "Should have correct parallel limit")
		assert.Equal(t, 60*time.Second, moa.Config.AgentTimeout, "Should have correct timeout")
	})

	// Test invalid MOA configuration
	t.Run("invalid_moa_config", func(t *testing.T) {
		invalidConfig := gollm.MOAConfig{
			Iterations:   2,
			Models:       []gollm.ConfigOption{}, // Empty models list
			MaxParallel:  2,
			AgentTimeout: 30 * time.Second,
		}

		_, err := gollm.NewMOA(invalidConfig)
		if !assert.Error(t, err, "Should fail with empty models list") {
			t.FailNow()
		}
		assert.Contains(t, err.Error(), "at least one model must be specified", "Error should indicate empty models list")
	})

	// Test MOA response generation using assess framework
	test.AddCase("moa_generation", "Explain the concept of quantum entanglement and its potential applications in computing.").
		WithTimeout(90 * time.Second). // Increased timeout
		WithSystemPrompt("You are a quantum computing expert. Provide detailed technical explanations.").
		Validate(func(response string) error {
			if !strings.Contains(strings.ToLower(response), "quantum") {
				return fmt.Errorf("response should contain quantum-related content")
			}
			if !strings.Contains(strings.ToLower(response), "entanglement") {
				return fmt.Errorf("response should contain entanglement-related content")
			}
			if len(response) < 200 {
				return fmt.Errorf("response too short, expected at least 200 characters")
			}
			return nil
		})

	// Test MOA with timeout using assess framework
	test.AddCase("moa_timeout", "Explain the concept of quantum entanglement.").
		WithTimeout(50 * time.Millisecond). // Slightly longer but still short enough to timeout
		Validate(func(response string) error {
			return fmt.Errorf("should have timed out")
		})

	// Test cross-provider consistency
	test.AddCase("cross_provider_test", "What is 2+2? Respond with just the number.").
		WithTimeout(45 * time.Second).
		WithSystemPrompt("You are a math tutor. Provide only the numerical answer without explanation.").
		Validate(func(response string) error {
			cleaned := strings.TrimSpace(response)
			if !strings.Contains(cleaned, "4") {
				return fmt.Errorf("expected response to contain '4', got: %s", cleaned)
			}
			return nil
		})

	// Run all test cases
	ctx := context.Background()
	test.RunBatch(ctx)

	// Verify metrics
	metrics := test.GetBatchMetrics()
	if metrics != nil {
		// Check for errors
		if len(metrics.Errors) > 0 {
			for provider, errs := range metrics.Errors {
				for _, err := range errs {
					t.Logf("Provider %s error: %v", provider, err)
				}
			}
		}

		// Check response times
		for provider, latency := range metrics.BatchTiming.ProviderLatency {
			t.Logf("Provider %s average latency: %v", provider, latency)
			assert.True(t, latency > 0, "Should have non-zero latency")
			if latency > 120*time.Second {
				t.Errorf("Response time too high for %s: %v", provider, latency)
			}
		}

		// Log batch execution summary
		t.Logf("Total batch execution time: %v", metrics.BatchTiming.TotalDuration)
		t.Logf("Average concurrent operations: %.2f", metrics.ConcurrencyStats.AverageConcurrent)
		t.Logf("Worker utilization: %.2f%%", metrics.ConcurrencyStats.WorkerUtilization*100)
	}
}
