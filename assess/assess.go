// Package testing provides testing utilities for the Gollm library.
package assess

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
	"golang.org/x/time/rate"
)

// BatchTestConfig configures batch test execution
type BatchTestConfig struct {
	EnableBatch  bool
	MaxParallel  int
	RateLimit    *rate.Limiter
	BatchTimeout time.Duration
}

// BatchMetrics tracks batch execution metrics
type BatchMetrics struct {
	BatchTiming struct {
		StartTime       time.Time
		EndTime         time.Time
		TotalDuration   time.Duration
		ProviderLatency map[string]time.Duration
	}
	ConcurrencyStats struct {
		MaxConcurrent     int
		AverageConcurrent float64
		WorkerUtilization float64
	}
	Errors map[string][]error
}

// TestProvider represents a provider configuration for testing
type TestProvider struct {
	Name    string
	Model   string
	Headers map[string]string
}

// TestCase represents a single test scenario
type TestCase struct {
	Name           string
	Input          string
	SystemPrompt   string
	ExpectedSchema interface{}
	Timeout        time.Duration
	Validations    []ValidationFunc
	options        map[string]interface{}
	directives     []string
	context        string
	maxLength      int
	examples       []string
	tools          []gollm.Tool
	toolChoice     string
	messages       []gollm.PromptMessage
	output         string
}

// ValidationFunc is a function type for custom validations
type ValidationFunc func(response string) error

// TestRunner manages test execution across providers
type TestRunner struct {
	t            *testing.T
	providers    []TestProvider
	cases        []*TestCase
	metrics      *TestMetrics
	config       *config.Config
	batchCfg     *BatchTestConfig
	batchMetrics *BatchMetrics
	mu           sync.Mutex
}

// TestMetrics tracks test execution metrics
type TestMetrics struct {
	ResponseTimes map[string][]time.Duration
	CacheHits     map[string]int
	CacheMisses   map[string]int
	Errors        map[string][]error
	StartTime     time.Time
}

func NewTest(t *testing.T) *TestRunner {
	return &TestRunner{
		t: t,
		metrics: &TestMetrics{
			ResponseTimes: make(map[string][]time.Duration),
			CacheHits:     make(map[string]int),
			CacheMisses:   make(map[string]int),
			Errors:        make(map[string][]error),
			StartTime:     time.Now(),
		},
	}
}

func (tr *TestRunner) WithProvider(name, model string) *TestRunner {
	tr.providers = append(tr.providers, TestProvider{
		Name:    name,
		Model:   model,
		Headers: getDefaultHeaders(name),
	})
	return tr
}

// HasAvailableProviders checks if any configured provider has an API key set.
// Use this to skip tests early when no providers are available.
func (tr *TestRunner) HasAvailableProviders() bool {
	for _, provider := range tr.providers {
		apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(provider.Name))
		if os.Getenv(apiKeyEnv) != "" {
			return true
		}
	}
	return false
}

func (tr *TestRunner) WithProviders(providers map[string]string) *TestRunner {
	for name, model := range providers {
		tr.WithProvider(name, model)
	}
	return tr
}

func (tr *TestRunner) WithConfig(cfg *config.Config) *TestRunner {
	tr.config = cfg
	return tr
}

func (tc *TestCase) WithSystemPrompt(prompt string) *TestCase {
	tc.SystemPrompt = prompt
	return tc
}

func (tc *TestCase) WithTimeout(timeout time.Duration) *TestCase {
	tc.Timeout = timeout
	return tc
}

func (tc *TestCase) ExpectSchema(schema interface{}) *TestCase {
	tc.ExpectedSchema = schema
	return tc
}

func (tc *TestCase) Validate(fn ValidationFunc) *TestCase {
	tc.Validations = append(tc.Validations, fn)
	return tc
}

func (tc *TestCase) WithOption(key string, value interface{}) *TestCase {
	if tc.options == nil {
		tc.options = make(map[string]interface{})
	}
	tc.options[key] = value
	return tc
}

func (tr *TestRunner) AddCase(name, input string) *TestCase {
	tc := &TestCase{
		Name:    name,
		Input:   input,
		Timeout: 30 * time.Second,
		options: make(map[string]interface{}),
	}
	tr.cases = append(tr.cases, tc)
	return tc
}

func (tr *TestRunner) WithBatchConfig(cfg BatchTestConfig) *TestRunner {
	if cfg.MaxParallel <= 0 {
		cfg.MaxParallel = 5 // default
	}
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = 10 * time.Minute // default
	}
	tr.batchCfg = &cfg
	tr.batchMetrics = &BatchMetrics{
		Errors: make(map[string][]error),
	}
	tr.batchMetrics.BatchTiming.ProviderLatency = make(map[string]time.Duration)
	return tr
}

func (tr *TestRunner) GetBatchMetrics() *BatchMetrics {
	return tr.batchMetrics
}

// Helper method to print error summary
func (tr *TestRunner) printErrorSummary() {
	tr.t.Log("\n=== Error Summary ===")

	if tr.batchMetrics != nil && len(tr.batchMetrics.Errors) > 0 {
		for provider, errors := range tr.batchMetrics.Errors {
			if len(errors) > 0 {
				tr.t.Logf("\nProvider: %s", provider)
				tr.t.Log("Errors:")
				for i, err := range errors {
					tr.t.Logf("  %d. %v", i+1, err)
				}
			}
		}
	} else {
		tr.t.Log("No errors recorded during test execution")
	}

	tr.t.Log("\n=== Performance Summary ===")
	for provider, latency := range tr.batchMetrics.BatchTiming.ProviderLatency {
		tr.t.Logf("Provider %s - Average Response Time: %v", provider, latency)
	}
	tr.t.Logf("Total Execution Time: %v", tr.batchMetrics.BatchTiming.TotalDuration)
	tr.t.Log("==================")
}

func (tr *TestRunner) RunBatch(ctx context.Context) {
	if tr.batchCfg == nil {
		tr.batchCfg = &BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  5,
			BatchTimeout: 10 * time.Minute,
		}
	}

	// Filter providers with available API keys
	var availableProviders []TestProvider
	for _, provider := range tr.providers {
		apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(provider.Name))
		if os.Getenv(apiKeyEnv) != "" {
			availableProviders = append(availableProviders, provider)
		} else {
			tr.t.Logf("Skipping provider %s: %s environment variable not set", provider.Name, apiKeyEnv)
		}
	}

	if len(availableProviders) == 0 {
		tr.t.Log("No providers available: missing API keys - skipping batch execution")
		return
	}

	ctx, cancel := context.WithTimeout(ctx, tr.batchCfg.BatchTimeout)
	defer cancel()

	tr.batchMetrics.BatchTiming.StartTime = time.Now()
	tr.t.Logf("Starting batch execution with %d providers and %d test cases", len(availableProviders), len(tr.cases))

	// Create worker pool
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, tr.batchCfg.MaxParallel)
	currentConcurrent := 0
	maxConcurrent := 0

	var concurrencyMu sync.Mutex

	// Create channels for real-time updates
	type testResult struct {
		provider string
		testCase string
		duration time.Duration
		err      error
		response string
	}
	results := make(chan testResult, len(availableProviders)*len(tr.cases))

	for _, provider := range availableProviders {
		client := tr.setupClient(tr.t, provider)
		tr.t.Logf("Initialized client for provider: %s", provider.Name)

		for _, tc := range tr.cases {
			wg.Add(1)
			go func(p TestProvider, testCase *TestCase) {
				defer wg.Done()

				// Acquire semaphore
				semaphore <- struct{}{}
				concurrencyMu.Lock()
				currentConcurrent++
				if currentConcurrent > maxConcurrent {
					maxConcurrent = currentConcurrent
				}
				concurrencyMu.Unlock()

				tr.t.Logf("Starting test case [%s] for provider [%s]", testCase.Name, p.Name)

				// Apply rate limiting if configured
				if tr.batchCfg.RateLimit != nil {
					if err := tr.batchCfg.RateLimit.Wait(ctx); err != nil {
						tr.recordError(p.Name, fmt.Errorf("rate limit wait failed: %w", err))
						tr.t.Logf("Rate limit wait error for provider %s: %v", p.Name, err)
					}
				}

				// Run the test case
				start := time.Now()
				var testErr error
				var response string

				// Create a sub-test for proper test organization
				tr.t.Run(fmt.Sprintf("%s/%s", p.Name, testCase.Name), func(t *testing.T) {
					response, testErr = tr.runBatchCase(ctx, t, client, p, testCase)
				})

				duration := time.Since(start)

				// Send result through channel
				results <- testResult{
					provider: p.Name,
					testCase: testCase.Name,
					duration: duration,
					err:      testErr,
					response: response,
				}

				// Update provider latency metrics
				tr.mu.Lock()
				if existing, ok := tr.batchMetrics.BatchTiming.ProviderLatency[p.Name]; ok {
					tr.batchMetrics.BatchTiming.ProviderLatency[p.Name] = (existing + duration) / 2
				} else {
					tr.batchMetrics.BatchTiming.ProviderLatency[p.Name] = duration
				}
				tr.mu.Unlock()

				// Release semaphore
				concurrencyMu.Lock()
				currentConcurrent--
				concurrencyMu.Unlock()
				<-semaphore

				tr.t.Logf("Completed test case [%s] for provider [%s] in %v", testCase.Name, p.Name, duration)
			}(provider, tc)
		}
	}

	// Start a goroutine to close results channel when all tests complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results as they come in
	completedTests := 0
	totalTests := len(availableProviders) * len(tr.cases)
	for result := range results {
		completedTests++
		if result.err != nil {
			tr.recordError(result.provider, result.err)
			tr.t.Logf("❌ [%s/%s] Failed: %v", result.provider, result.testCase, result.err)
		} else {
			tr.t.Logf("✓ [%s/%s] Completed in %v", result.provider, result.testCase, result.duration)
		}
		tr.t.Logf("Progress: %d/%d tests completed (%d%%)", completedTests, totalTests, (completedTests*100)/totalTests)
	}

	// Record final metrics
	tr.batchMetrics.BatchTiming.EndTime = time.Now()
	tr.batchMetrics.BatchTiming.TotalDuration = tr.batchMetrics.BatchTiming.EndTime.Sub(tr.batchMetrics.BatchTiming.StartTime)
	tr.batchMetrics.ConcurrencyStats.MaxConcurrent = maxConcurrent

	tr.t.Logf("Batch execution completed in %v", tr.batchMetrics.BatchTiming.TotalDuration)
	tr.t.Logf("Maximum concurrent tests: %d", maxConcurrent)

	// Print error summary at the end
	tr.printErrorSummary()
}

// Helper method to run a single batch test case
func (tr *TestRunner) runBatchCase(ctx context.Context, t *testing.T, client llm.LLM, provider TestProvider, tc *TestCase) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, tc.Timeout)
	defer cancel()

	prompt := client.NewPrompt(tc.Input)
	if tc.SystemPrompt != "" {
		prompt.SystemPrompt = tc.SystemPrompt
	}

	// Apply test case options
	if tc.options != nil {
		for key, value := range tc.options {
			client.SetOption(key, value)
		}
	}

	var response string
	var err error

	if tc.ExpectedSchema != nil {
		// Use JSON schema validation for all providers
		response, err = client.GenerateWithSchema(ctx, prompt, tc.ExpectedSchema)
	} else {
		response, err = client.Generate(ctx, prompt)
	}

	if err != nil {
		return "", err
	}

	// Run validations
	for _, validate := range tc.Validations {
		if err := validate(response); err != nil {
			t.Errorf("Validation failed: %v", err)
		}
	}

	return response, nil
}

func (tr *TestRunner) setupClient(t *testing.T, provider TestProvider) llm.LLM {
	// Get API key from environment
	apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(provider.Name))
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		t.Skipf("Skipping tests for %s: %s environment variable not set", provider.Name, apiKeyEnv)
	}

	// Create options with provider-specific settings
	opts := []gollm.ConfigOption{
		gollm.SetProvider(provider.Name),
		gollm.SetModel(provider.Model),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second * 2),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetEnableCaching(true),
		gollm.SetTimeout(30 * time.Second),
	}

	// Add provider-specific settings
	switch provider.Name {
	case "anthropic":
		opts = append(opts,
			gollm.SetMaxTokens(1000),
		)
	case "openai":
		opts = append(opts,
			gollm.SetMaxTokens(500),
		)
	}

	// Add any provider-specific headers
	if provider.Headers != nil {
		opts = append(opts, gollm.SetExtraHeaders(provider.Headers))
	}

	// Create LLM client
	client, err := gollm.NewLLM(opts...)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Note: Response format for JSON schema validation is now set in runBatchCase
	// when ExpectedSchema is present, not here in setupClient

	return client
}

func (tr *TestRunner) runCase(ctx context.Context, t *testing.T, client llm.LLM, provider TestProvider, tc *TestCase) {
	ctx, cancel := context.WithTimeout(ctx, tc.Timeout)
	defer cancel()

	// Create prompt with all options
	promptOpts := []gollm.PromptOption{}

	if tc.SystemPrompt != "" {
		promptOpts = append(promptOpts, gollm.WithSystemPrompt(tc.SystemPrompt, gollm.CacheTypeEphemeral))
	}

	if len(tc.directives) > 0 {
		for _, directive := range tc.directives {
			promptOpts = append(promptOpts, gollm.WithDirectives(directive))
		}
	}

	if tc.context != "" {
		promptOpts = append(promptOpts, gollm.WithContext(tc.context))
	}

	if tc.maxLength > 0 {
		promptOpts = append(promptOpts, gollm.WithMaxLength(tc.maxLength))
	}

	if len(tc.examples) > 0 {
		for _, example := range tc.examples {
			promptOpts = append(promptOpts, gollm.WithExamples(example))
		}
	}

	if len(tc.tools) > 0 {
		promptOpts = append(promptOpts, gollm.WithTools(tc.tools))
	}

	if tc.toolChoice != "" {
		promptOpts = append(promptOpts, gollm.WithToolChoice(tc.toolChoice))
	}

	if len(tc.messages) > 0 {
		promptOpts = append(promptOpts, gollm.WithMessages(tc.messages))
	}

	if tc.output != "" {
		promptOpts = append(promptOpts, gollm.WithOutput(tc.output))
	}

	prompt := gollm.NewPrompt(tc.Input, promptOpts...)

	// Apply test case options
	if tc.options != nil {
		for key, value := range tc.options {
			client.SetOption(key, value)
		}
	}

	start := time.Now()
	var response string
	var err error

	if tc.ExpectedSchema != nil {
		response, err = client.GenerateWithSchema(ctx, prompt, tc.ExpectedSchema)
	} else {
		response, err = client.Generate(ctx, prompt)
	}

	duration := time.Since(start)
	tr.metrics.ResponseTimes[provider.Name] = append(tr.metrics.ResponseTimes[provider.Name], duration)

	if err != nil {
		tr.metrics.Errors[provider.Name] = append(tr.metrics.Errors[provider.Name], err)
		t.Errorf("Generation failed: %v", err)
		return
	}

	// Run validations
	for _, validate := range tc.Validations {
		if err := validate(response); err != nil {
			t.Errorf("Validation failed: %v", err)
		}
	}
}

func getDefaultHeaders(provider string) map[string]string {
	headers := make(map[string]string)

	switch provider {
	case "anthropic":
		headers["anthropic-beta"] = "prompt-caching-2024-07-31"
	case "openai":
		headers["Content-Type"] = "application/json"
	}

	return headers
}

// Common validation functions
func ExpectContains(substr string) ValidationFunc {
	return func(response string) error {
		if !strings.Contains(response, substr) {
			return fmt.Errorf("expected response to contain %q", substr)
		}
		return nil
	}
}

func ExpectMatches(pattern string) ValidationFunc {
	return func(response string) error {
		matched, err := regexp.MatchString(pattern, response)
		if err != nil {
			return fmt.Errorf("invalid pattern %q: %v", pattern, err)
		}
		if !matched {
			return fmt.Errorf("expected response to match pattern %q", pattern)
		}
		return nil
	}
}

func (tr *TestRunner) Run(ctx context.Context) {
	for _, provider := range tr.providers {
		tr.t.Run(provider.Name, func(t *testing.T) {
			client := tr.setupClient(t, provider)

			for _, tc := range tr.cases {
				t.Run(tc.Name, func(t *testing.T) {
					tr.runCase(ctx, t, client, provider, tc)
				})
			}
		})
	}
}

func (tr *TestRunner) recordError(provider string, err error) {
	tr.mu.Lock()
	defer tr.mu.Unlock()

	if tr.batchMetrics != nil {
		tr.batchMetrics.Errors[provider] = append(tr.batchMetrics.Errors[provider], err)
	}
	if tr.metrics != nil {
		tr.metrics.Errors[provider] = append(tr.metrics.Errors[provider], err)
	}
}

// WithDirectives adds directives to the test case
func (tc *TestCase) WithDirectives(directives []string) *TestCase {
	tc.directives = directives
	tc.options["directives"] = directives
	return tc
}

// WithContext adds context to the test case
func (tc *TestCase) WithContext(context string) *TestCase {
	tc.context = context
	tc.options["context"] = context
	return tc
}

// WithMaxLength sets the maximum length for the response
func (tc *TestCase) WithMaxLength(length int) *TestCase {
	tc.maxLength = length
	tc.options["max_length"] = length
	return tc
}

// WithExamples adds examples to the test case
func (tc *TestCase) WithExamples(examples []string) *TestCase {
	tc.examples = examples
	tc.options["examples"] = examples
	return tc
}

// WithTools configures available tools for the test case
func (tc *TestCase) WithTools(tools []gollm.Tool) *TestCase {
	tc.tools = tools
	tc.options["tools"] = tools
	return tc
}

// WithToolChoice specifies how tools should be selected
func (tc *TestCase) WithToolChoice(choice string) *TestCase {
	tc.toolChoice = choice
	tc.options["tool_choice"] = choice
	return tc
}

// WithMessages adds multiple messages to the test case
func (tc *TestCase) WithMessages(messages []gollm.PromptMessage) *TestCase {
	tc.messages = messages
	tc.options["messages"] = messages
	return tc
}

// WithOutput configures the expected output format
func (tc *TestCase) WithOutput(output string) *TestCase {
	tc.output = output
	tc.options["output"] = output
	return tc
}
