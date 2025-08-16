package assess

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Helper function to calculate average duration of first attempts
func averageFirstAttempts(queryTimes map[string][]time.Duration) time.Duration {
	var total time.Duration
	count := 0
	for _, times := range queryTimes {
		if len(times) > 0 {
			total += times[0]
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / time.Duration(count)
}

func TestBasicInteraction(t *testing.T) {
	test := NewTest(t).
		WithProvider("anthropic", "claude-3-5-haiku-latest")

	test.AddCase("basic_response", "What's 2+2?").
		WithTimeout(30 * time.Second)

	ctx := context.Background()
	test.Run(ctx)

	// Verify metrics
	for provider, times := range test.metrics.ResponseTimes {
		t.Run(provider+" metrics", func(t *testing.T) {
			// Check average response time
			avg := averageFirstAttempts(map[string][]time.Duration{
				"basic_response": times,
			})
			assert.Less(t, avg, 2*time.Second, "Average response time should be reasonable")
			assert.Empty(t, test.metrics.Errors[provider], "Should have no errors")
		})
	}
}

func TestMultiProviderInteraction(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping multi-provider test in short mode")
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

	// Test cross-provider interaction
	test.AddCase("cross_provider", "What is the capital of France?").
		WithTimeout(30*time.Second).
		WithOption("max_tokens", 100)

	ctx := context.Background()
	test.RunBatch(ctx)

	// Check metrics with more realistic thresholds
	metrics := test.GetBatchMetrics()
	for provider, latency := range metrics.BatchTiming.ProviderLatency {
		t.Run(provider+"_metrics", func(t *testing.T) {
			// Allow up to 10 seconds average response time
			// This is more realistic for API calls that may have network latency
			assert.Less(t, latency, 10*time.Second,
				"Average response time should be reasonable")

			// Log the actual latency for monitoring
			t.Logf("%s average latency: %v", provider, latency)
		})
	}
}
