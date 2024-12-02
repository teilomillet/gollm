package providers

import (
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// MockProvider implements the Provider interface for testing purposes.
type MockProvider struct {
	endpoint     string
	model        string
	extraHeaders map[string]string
	options      map[string]interface{}
	logger       utils.Logger
	// Mock response configuration
	responseText  string
	shouldError   bool
	errorMsg      string
	responses     []string // Queue of preset responses
	currentIndex  int      // Current position in response queue
	loopResponses bool     // Whether to loop through responses or error when exhausted
}

// NewMockProvider creates a new mock provider instance for testing.
func NewMockProvider(endpoint, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &MockProvider{
		endpoint:     endpoint,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
		responseText: "This is a mock response",
		responses:    make([]string, 0),
	}
}

// SetMockResponse configures the mock response text
func (p *MockProvider) SetMockResponse(response string) {
	p.responseText = response
}

// SetMockError configures the mock to return an error
func (p *MockProvider) SetMockError(shouldError bool, errorMsg string) {
	p.shouldError = shouldError
	p.errorMsg = errorMsg
}

func (p *MockProvider) SetLogger(logger utils.Logger)             { p.logger = logger }
func (p *MockProvider) Name() string                              { return "mock" }
func (p *MockProvider) Endpoint() string                          { return p.endpoint }
func (p *MockProvider) SetOption(key string, value interface{})   { p.options[key] = value }
func (p *MockProvider) SupportsJSONSchema() bool                  { return true }
func (p *MockProvider) SetExtraHeaders(headers map[string]string) { p.extraHeaders = headers }

func (p *MockProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type": "application/json",
	}
	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

func (p *MockProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	if p.shouldError {
		return nil, fmt.Errorf(p.errorMsg)
	}

	requestBody := map[string]interface{}{
		"model":  p.model,
		"prompt": prompt,
	}
	for k, v := range options {
		requestBody[k] = v
	}
	return json.Marshal(requestBody)
}

func (p *MockProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	return p.PrepareRequest(prompt, options)
}

// SetResponses configures a list of responses to be returned in sequence
func (p *MockProvider) SetResponses(responses []string, loop bool) {
	p.responses = responses
	p.currentIndex = 0
	p.loopResponses = loop
}

// getNextResponse returns the next response from the queue
func (p *MockProvider) getNextResponse() (string, error) {
	if len(p.responses) == 0 {
		return p.responseText, nil // Fall back to default response
	}

	if p.currentIndex >= len(p.responses) {
		if p.loopResponses {
			p.currentIndex = 0 // Reset to start
		} else {
			return "", fmt.Errorf("mock responses exhausted")
		}
	}

	response := p.responses[p.currentIndex]
	p.currentIndex++
	return response, nil
}

func (p *MockProvider) ParseResponse(body []byte) (string, error) {
	if p.shouldError {
		return "", fmt.Errorf(p.errorMsg)
	}
	return p.getNextResponse()
}

func (p *MockProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	return nil, nil
}

func (p *MockProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}
