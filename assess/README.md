# Gollm Testing Framework

A state-of-the-art testing framework for the `gollm` library, providing comprehensive testing capabilities for LLM interactions.

## Core Features

- Enterprise-grade testing infrastructure
- Real provider testing with response validation
- Performance metrics and caching analysis
- Multi-provider compatibility testing
- Behavioral testing with custom validations
- Comprehensive metrics collection
- Timeout and error handling
- System prompt support

## Quick Start

```go
func TestBasic(t *testing.T) {
    test := testing.NewTest(t).
        WithProvider("anthropic", "claude-3-5-haiku-latest")

    test.AddCase("simple_query", "What is 2+2?").
        WithTimeout(30 * time.Second).
        Validate(ExpectContains("4"))

    ctx := context.Background()
    test.Run(ctx)
}
```

## Advanced Usage

### Testing Multiple Providers

```go
test := testing.NewTest(t).
    WithProviders(map[string]string{
        "anthropic": "claude-3-5-haiku-latest",
        "openai":    "gpt-4o-mini",
    })
```

### Caching Behavior Tests

```go
func TestCaching(t *testing.T) {
    test := testing.NewTest(t).
        WithProvider("anthropic", "claude-3-5-haiku-latest")

    // Same query multiple times to test cache
    query := "What is the capital of France?"
    for i := 0; i < 3; i++ {
        test.AddCase(fmt.Sprintf("cache_test_%d", i), query).
            WithTimeout(30 * time.Second)
    }

    ctx := context.Background()
    test.Run(ctx)

    // Verify cache metrics
    metrics := test.GetMetrics()
    // First call should be slower than subsequent cached calls
    assert.Less(t, metrics.ResponseTimes["anthropic"][1], 
        metrics.ResponseTimes["anthropic"][0])
}
```

### Custom Validation Functions

```go
test.AddCase("complex_validation", "Generate a JSON response").
    Validate(ExpectContains("data")).
    Validate(ExpectMatches(`"status":\s*"success"`)).
    Validate(func(response string) error {
        // Custom validation logic
        return nil
    })
```

### System Prompts

```go
test.AddCase("with_system_prompt", "Analyze this code").
    WithSystemPrompt("You are an expert code reviewer").
    WithTimeout(45 * time.Second)
```

## Performance Testing

The framework automatically collects key metrics:

- Response times per provider
- Cache hit/miss rates
- Error rates and types
- Total test execution time

## Configuration

### Environment Variables

Required API keys:
- `ANTHROPIC_API_KEY` for Anthropic
- `OPENAI_API_KEY` for OpenAI

### Provider-Specific Headers

The framework automatically configures appropriate headers for each provider:
```go
// Headers are automatically set based on provider
headers := getDefaultHeaders("anthropic")
// Returns: {"anthropic-beta": "prompt-caching-2024-07-31"}
```

## Running Tests

Run all tests:
```bash
go test ./testing/...
```

Run specific test:
```bash
go test -run "TestCaching" ./testing/...
```

Skip provider-specific tests:
```bash
SKIP_OPENAI=1 go test ./testing/...
```

## Best Practices

1. Test real provider interactions for production readiness
2. Use appropriate timeouts for different types of queries
3. Implement comprehensive validation functions
4. Monitor cache performance in performance-critical scenarios
5. Use system prompts to ensure consistent behavior
6. Test both success and error scenarios
7. Verify response times and cache effectiveness
8. Include cross-provider compatibility tests

## Metrics and Monitoring

The framework provides detailed metrics:
```go
metrics := test.GetMetrics()
fmt.Printf("Average response time: %v\n", metrics.AverageResponseTime())
fmt.Printf("Cache hit rate: %v%%\n", metrics.CacheHitRate())
fmt.Printf("Error rate: %v%%\n", metrics.ErrorRate())
```

## Error Handling

The framework includes built-in error handling and retries:
- Automatic retries on transient failures
- Timeout handling
- Rate limit management
- Detailed error reporting 