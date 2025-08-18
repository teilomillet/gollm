// Package assess provides testing utilities for the Gollm library.
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

	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"golang.org/x/time/rate"

	"github.com/weave-labs/gollm"
	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/models"
	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/providers"
)

// Constants for default values
const (
	DefaultTestTimeout     = 30 * time.Second
	DefaultBatchTimeout    = 10 * time.Minute
	DefaultMaxParallel     = 5
	DefaultMaxRetries      = 3
	DefaultRetryDelay      = 2 * time.Second
	AverageWeight          = 2
	PercentageMultiplier   = 100
	DefaultAnthropicTokens = 1000
	DefaultOpenAITokens    = 500
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
	Name               string
	Input              string
	SystemPrompt       string
	ExpectedSchema     *jsonschema.Schema
	ExpectedSchemaType any // Used for JSON schema validation
	Timeout            time.Duration
	Validations        []ValidationFunc
	options            map[string]any
	directives         []string
	context            string
	maxLength          int
	examples           []string
	tools              []models.Tool
	toolChoice         string
	messages           []gollm.PromptMessage
	output             string
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

// NewTest creates a new test runner instance
func NewTest(t *testing.T) *TestRunner {
	t.Helper()
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

// WithProvider adds a provider to test against.
func (tr *TestRunner) WithProvider(name, model string) *TestRunner {
	tr.providers = append(tr.providers, TestProvider{
		Name:    name,
		Model:   model,
		Headers: getDefaultHeaders(name),
	})
	return tr
}

// WithProviders configures the test runner with multiple providers
func (tr *TestRunner) WithProviders(providerList map[string]string) *TestRunner {
	for name, model := range providerList {
		tr.WithProvider(name, model)
	}
	return tr
}

// WithConfig sets the configuration for the test runner
func (tr *TestRunner) WithConfig(cfg *config.Config) *TestRunner {
	tr.config = cfg
	return tr
}

// WithSystemPrompt sets the system prompt for the test case
func (tc *TestCase) WithSystemPrompt(prompt string) *TestCase {
	tc.SystemPrompt = prompt
	return tc
}

// WithTimeout sets the timeout duration for the test case
func (tc *TestCase) WithTimeout(timeout time.Duration) *TestCase {
	tc.Timeout = timeout
	return tc
}

// SetExpectedSchema sets the expected schema for validation
func SetExpectedSchema[T any](tc *TestCase) *TestCase {
	schema, err := jsonschema.For[T]()
	if err != nil {
		panic(fmt.Errorf("failed to get schema for type %T: %w", tc, err))
	}

	tc.ExpectedSchema = schema
	tc.ExpectedSchemaType = *new(T)

	return tc
}

// Validate adds a validation function to the test case
func (tc *TestCase) Validate(fn ValidationFunc) *TestCase {
	tc.Validations = append(tc.Validations, fn)
	return tc
}

// WithOption sets a test case option
func (tc *TestCase) WithOption(key string, value any) *TestCase {
	if tc.options == nil {
		tc.options = make(map[string]any)
	}
	tc.options[key] = value
	return tc
}

// AddCase adds a new test case to the runner
func (tr *TestRunner) AddCase(name, input string) *TestCase {
	tc := &TestCase{
		Name:    name,
		Input:   input,
		Timeout: DefaultTestTimeout,
		options: make(map[string]any),
	}
	tr.cases = append(tr.cases, tc)
	return tc
}

// WithBatchConfig configures batch test execution settings
func (tr *TestRunner) WithBatchConfig(cfg BatchTestConfig) *TestRunner {
	if cfg.MaxParallel <= 0 {
		cfg.MaxParallel = DefaultMaxParallel // default
	}
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = DefaultBatchTimeout // default
	}
	tr.batchCfg = &cfg
	tr.batchMetrics = &BatchMetrics{
		Errors: make(map[string][]error),
	}
	tr.batchMetrics.BatchTiming.ProviderLatency = make(map[string]time.Duration)
	return tr
}

// GetBatchMetrics returns the batch execution metrics
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

// testResult represents the result of a single test execution
type testResult struct {
	provider string
	testCase string
	duration time.Duration
	err      error
	response string
}

// workerState tracks concurrency state across workers
type workerState struct {
	current int
	max     int
	mu      sync.Mutex
}

// RunBatch executes all test cases in parallel batches
func (tr *TestRunner) RunBatch(ctx context.Context) {
	if tr.batchCfg == nil {
		tr.batchCfg = &BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  DefaultMaxParallel,
			BatchTimeout: DefaultBatchTimeout,
		}
	}

	ctx, cancel := context.WithTimeout(ctx, tr.batchCfg.BatchTimeout)
	defer cancel()

	tr.batchMetrics.BatchTiming.StartTime = time.Now()
	tr.t.Logf("Starting batch execution with %d providers and %d test cases", len(tr.providers), len(tr.cases))

	// Create worker pool
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, tr.batchCfg.MaxParallel)
	workerState := &workerState{}
	results := make(chan testResult, len(tr.providers)*len(tr.cases))

	// Launch test workers
	tr.launchTestWorkers(ctx, &wg, semaphore, workerState, results)

	// Start a goroutine to close results channel when all tests complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results
	tr.processTestResults(results)

	// Finalize metrics
	tr.finalizeBatchMetrics(workerState.max)
}

// launchTestWorkers starts test execution workers
func (tr *TestRunner) launchTestWorkers(
	ctx context.Context,
	wg *sync.WaitGroup,
	semaphore chan struct{},
	state *workerState,
	results chan testResult,
) {
	for _, provider := range tr.providers {
		client := tr.setupClient(provider)
		tr.t.Logf("Initialized client for provider: %s", provider.Name)

		for _, tc := range tr.cases {
			wg.Add(1)
			go tr.runTestWorker(ctx, wg, semaphore, state, results, provider, tc, client)
		}
	}
}

// runTestWorker executes a single test in a worker goroutine
func (tr *TestRunner) runTestWorker(
	ctx context.Context,
	wg *sync.WaitGroup,
	semaphore chan struct{},
	state *workerState,
	results chan testResult,
	p TestProvider,
	testCase *TestCase,
	client llm.LLM,
) {
	defer wg.Done()

	// Acquire semaphore
	semaphore <- struct{}{}
	state.mu.Lock()
	state.current++
	if state.current > state.max {
		state.max = state.current
	}
	state.mu.Unlock()

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
		response, testErr = tr.runBatchCase(ctx, t, client, testCase)
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
		tr.batchMetrics.BatchTiming.ProviderLatency[p.Name] = (existing + duration) / AverageWeight
	} else {
		tr.batchMetrics.BatchTiming.ProviderLatency[p.Name] = duration
	}
	tr.mu.Unlock()

	// Release semaphore
	state.mu.Lock()
	state.current--
	state.mu.Unlock()
	<-semaphore

	tr.t.Logf("Completed test case [%s] for provider [%s] in %v", testCase.Name, p.Name, duration)
}

// processTestResults handles incoming test results
func (tr *TestRunner) processTestResults(results chan testResult) {
	completedTests := 0
	totalTests := len(tr.providers) * len(tr.cases)
	for result := range results {
		completedTests++
		if result.err != nil {
			tr.recordError(result.provider, result.err)
			tr.t.Logf("❌ [%s/%s] Failed: %v", result.provider, result.testCase, result.err)
		} else {
			tr.t.Logf("✓ [%s/%s] Completed in %v", result.provider, result.testCase, result.duration)
		}
		tr.t.Logf(
			"Progress: %d/%d tests completed (%d%%)",
			completedTests,
			totalTests,
			(completedTests*PercentageMultiplier)/totalTests,
		)
	}
}

// finalizeBatchMetrics records final execution metrics
func (tr *TestRunner) finalizeBatchMetrics(maxConcurrent int) {
	tr.batchMetrics.BatchTiming.EndTime = time.Now()
	tr.batchMetrics.BatchTiming.TotalDuration = tr.batchMetrics.BatchTiming.EndTime.Sub(
		tr.batchMetrics.BatchTiming.StartTime,
	)
	tr.batchMetrics.ConcurrencyStats.MaxConcurrent = maxConcurrent

	tr.t.Logf("Batch execution completed in %v", tr.batchMetrics.BatchTiming.TotalDuration)
	tr.t.Logf("Maximum concurrent tests: %d", maxConcurrent)

	// Print error summary at the end
	tr.printErrorSummary()
}

// Helper method to run a single batch test case
func (tr *TestRunner) runBatchCase(ctx context.Context, t *testing.T, client llm.LLM, tc *TestCase) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(ctx, tc.Timeout)
	defer cancel()

	prompt := llm.NewPrompt(tc.Input)
	if tc.SystemPrompt != "" {
		prompt.SystemPrompt = tc.SystemPrompt
	}

	// Apply test case options
	if tc.options != nil {
		for key, value := range tc.options {
			client.SetOption(key, value)
		}
	}

	var response *providers.Response
	var err error

	response, err = client.Generate(ctx, prompt)

	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	// Run validations
	for _, validate := range tc.Validations {
		if err := validate(response.AsText()); err != nil {
			t.Errorf("Validation failed: %v", err)
		}
	}

	return response.AsText(), nil
}

func (tr *TestRunner) setupClient(provider TestProvider) llm.LLM {
	apiKeyEnv := strings.ToUpper(provider.Name) + "_API_KEY"
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		tr.t.Skipf("Skipping tests for %s: %s environment variable not set", provider.Name, apiKeyEnv)
	}

	// Create options with provider-specific settings
	opts := []gollm.ConfigOption{
		gollm.SetProvider(provider.Name),
		gollm.SetModel(provider.Model),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxRetries(DefaultMaxRetries),
		gollm.SetRetryDelay(DefaultRetryDelay),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetEnableCaching(true),
		gollm.SetTimeout(DefaultTestTimeout),
	}

	// Add provider-specific settings
	switch provider.Name {
	case "anthropic":
		opts = append(opts,
			gollm.SetMaxTokens(DefaultAnthropicTokens),
		)
	case "openai":
		opts = append(opts,
			gollm.SetMaxTokens(DefaultOpenAITokens),
		)
	}

	// Add any provider-specific headers
	if provider.Headers != nil {
		opts = append(opts, gollm.SetExtraHeaders(provider.Headers))
	}

	// Create LLM client
	client, err := gollm.NewLLM(opts...)
	if err != nil {
		tr.t.Fatalf("Failed to create client: %v", err)
	}

	// Note: Response format for JSON schema validation is now set in runBatchCase
	// when ExpectedSchema is present, not here in setupClient

	return client
}

// buildPromptOptions creates prompt options from test case configuration
func (tc *TestCase) buildPromptOptions() []gollm.PromptOption {
	var promptOpts []gollm.PromptOption

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

	return promptOpts
}

func (tr *TestRunner) runCase(ctx context.Context, t *testing.T, client llm.LLM, provider TestProvider, tc *TestCase) {
	t.Helper()
	ctx, cancel := context.WithTimeout(ctx, tc.Timeout)
	defer cancel()

	// Create prompt with all options
	promptOpts := tc.buildPromptOptions()
	prompt := gollm.NewPrompt(tc.Input, promptOpts...)

	// Apply test case options
	if tc.options != nil {
		for key, value := range tc.options {
			client.SetOption(key, value)
		}
	}

	start := time.Now()
	var response *providers.Response
	var err error

	response, err = client.Generate(ctx, prompt)

	duration := time.Since(start)
	tr.metrics.ResponseTimes[provider.Name] = append(tr.metrics.ResponseTimes[provider.Name], duration)

	if err != nil {
		tr.metrics.Errors[provider.Name] = append(tr.metrics.Errors[provider.Name], err)
		t.Errorf("Generation failed: %v", err)
		return
	}

	// Run validations
	for _, validate := range tc.Validations {
		if err := validate(response.AsText()); err != nil {
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

// ExpectContains Common validation functions
func ExpectContains(substr string) ValidationFunc {
	return func(response string) error {
		if !strings.Contains(response, substr) {
			return fmt.Errorf("expected response to contain %q", substr)
		}
		return nil
	}
}

// ExpectMatches creates a validation function that checks if response matches a regex pattern
func ExpectMatches(pattern string) ValidationFunc {
	return func(response string) error {
		matched, err := regexp.MatchString(pattern, response)
		if err != nil {
			return fmt.Errorf("invalid pattern %q: %w", pattern, err)
		}
		if !matched {
			return fmt.Errorf("expected response to match pattern %q", pattern)
		}
		return nil
	}
}

// Run executes all test cases sequentially
func (tr *TestRunner) Run(ctx context.Context) {
	for _, provider := range tr.providers {
		tr.t.Run(provider.Name, func(t *testing.T) {
			client := tr.setupClient(provider)

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
func (tc *TestCase) WithContext(ctx string) *TestCase {
	tc.context = ctx
	tc.options["context"] = ctx
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
func (tc *TestCase) WithTools(tools []models.Tool) *TestCase {
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
