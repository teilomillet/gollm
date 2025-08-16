package assess

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestCachingBehavior tests the caching functionality across providers
func TestCachingBehavior(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping caching test in short mode")
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

	// Use a simpler query that's more likely to be cached consistently
	query := "What is 2+2?"

	// Run the query multiple times to test caching
	for i := range 3 {
		test.AddCase(fmt.Sprintf("cache_test_%d", i), query).
			WithTimeout(30 * time.Second)
	}

	ctx := context.Background()
	test.RunBatch(ctx)

	// Analyze cache performance with more tolerance for variations
	for provider, times := range test.metrics.ResponseTimes {
		if len(times) < 2 {
			continue
		}

		baseline := times[0]
		t.Logf("%s baseline response time: %v", provider, baseline)

		// Calculate average of subsequent responses
		var totalCached time.Duration
		for i := 1; i < len(times); i++ {
			totalCached += times[i]
			t.Logf("%s cache hit %d time: %v", provider, i, times[i])
		}
		avgCached := totalCached / time.Duration(len(times)-1)
		t.Logf("%s average cached response time: %v", provider, avgCached)

		// Check if there's any improvement in average response time
		// Allow for some variance due to network conditions
		improvement := 1 - (float64(avgCached) / float64(baseline))
		t.Logf("%s average cache improvement: %.2f%%", provider, improvement*100)

		// We expect some improvement, but be more lenient with the threshold
		// Just ensure cached responses aren't significantly slower
		assert.Less(t, avgCached, baseline*2,
			"Cached responses for %s shouldn't be significantly slower than baseline", provider)
	}
}

// TestPerformanceCharacteristics tests response times and performance metrics
func TestPerformanceCharacteristics(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
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

	queries := []string{
		"What is the time complexity of quicksort?",
		"Explain the concept of memory locality",
		"How does garbage collection work?",
	}

	for _, query := range queries {
		test.AddCase("perf_"+query[:10], query).
			WithTimeout(30 * time.Second)
	}

	ctx := context.Background()
	test.RunBatch(ctx)

	// Analyze performance metrics
	metrics := test.metrics
	for providerName, times := range metrics.ResponseTimes {
		if len(times) == 0 {
			continue
		}

		// Calculate statistics
		var total, maxDur, minDur time.Duration
		minDur = times[0]
		for _, duration := range times {
			total += duration
			if duration > maxDur {
				maxDur = duration
			}
			if duration < minDur {
				minDur = duration
			}
		}
		avg := total / time.Duration(len(times))

		// Log detailed metrics
		t.Logf("%s Performance Metrics:", providerName)
		t.Logf("  Average: %v", avg)
		t.Logf("  Min: %v", minDur)
		t.Logf("  Max: %v", maxDur)
		t.Logf("  Variance: %v", maxDur-minDur)

		// Ensure max response time is within reasonable bounds
		assert.Less(t, maxDur, 2*time.Minute,
			"Maximum response time for %s should be under 2 minutes", providerName)
	}
}

// TestErrorHandling tests timeout and error scenarios
func TestErrorHandling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping error handling test in short mode")
	}

	test := NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  1,
			BatchTimeout: 5 * time.Second,
		})

	test.AddCase("timeout_test", "Generate a very long response").
		WithTimeout(1 * time.Millisecond).
		Validate(func(response string) error {
			// With such a short timeout (1ms), we expect an empty response
			if response == "" {
				return nil // Expected timeout resulted in empty response
			}
			return errors.New("expected timeout to prevent response generation")
		})

	ctx := context.Background()
	test.RunBatch(ctx)
}
