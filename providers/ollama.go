// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// OllamaProvider implements the Provider interface for Ollama's API.
// It enables interaction with locally hosted language models through Ollama,
// supporting various open-source models like Llama, Mistral, and others.
type OllamaProvider struct {
	// endpoint is the base URL for the Ollama API
	endpoint     string                 // Ollama API endpoint URL
	// model is the name of the model to use (e.g., "llama2", "mistral")
	model        string                 // Model identifier (e.g., "llama2", "mistral")
	// extraHeaders are additional HTTP headers for requests
	extraHeaders map[string]string      // Additional HTTP headers
	// options are model-specific options for the provider
	options      map[string]interface{} // Model-specific options
	// logger is the logger instance for this provider
	logger       utils.Logger           // Logger instance
}

// NewOllamaProvider creates a new Ollama provider instance.
// It initializes the provider with the specified endpoint URL and model name.
//
// Parameters:
//   - endpoint: The Ollama API endpoint URL (e.g., "http://localhost:11434")
//   - model: The model to use (e.g., "llama2", "mistral")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Ollama Provider instance
func NewOllamaProvider(endpoint, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OllamaProvider{
		endpoint:     endpoint,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Ollama provider.
// This is used for debugging and monitoring API interactions.
func (p *OllamaProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// Name returns the identifier for this provider ("ollama").
func (p *OllamaProvider) Name() string {
	return "ollama"
}

// Endpoint returns the configured Ollama API endpoint URL.
// This is typically "http://localhost:11434/api/generate".
func (p *OllamaProvider) Endpoint() string {
	return p.endpoint + "/api/generate"
}

// SetOption sets a model-specific option for the Ollama provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - num_predict: Maximum number of tokens to generate
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
//   - stop: Custom stop sequences
func (p *OllamaProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	if p.logger != nil {
		p.logger.Debug("Setting option for Ollama", "key", key, "value", value)
	}
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature and other generation parameters.
func (p *OllamaProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("num_predict", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	if config.OllamaEndpoint != "" {
		p.SetEndpoint(config.OllamaEndpoint)
	}
	p.SetOption("top_p", config.TopP)
	p.SetOption("min_p", config.MinP)
	p.SetOption("repeat_penalty", config.RepeatPenalty)
	p.SetOption("repeat_last_n", config.RepeatLastN)
	p.SetOption("mirostat", config.Mirostat)
	p.SetOption("mirostat_eta", config.MirostatEta)
	p.SetOption("mirostat_tau", config.MirostatTau)
	p.SetOption("tfs_z", config.TfsZ)
}

// SupportsJSONSchema indicates whether this provider supports JSON schema validation.
// Currently, Ollama does not natively support JSON schema validation.
func (p *OllamaProvider) SupportsJSONSchema() bool {
	return false
}

// Headers returns the HTTP headers required for Ollama API requests.
// This includes content type and any custom headers.
func (p *OllamaProvider) Headers() map[string]string {
	return map[string]string{
		"Content-Type": "application/json",
	}
}

// PrepareRequest creates the request body for an Ollama API call.
// It formats the prompt and options according to Ollama's API requirements.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OllamaProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  p.model,
		"prompt": prompt,
	}

	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// PrepareRequestWithSchema creates a request with JSON schema validation.
// Since Ollama doesn't support schema validation natively, this falls back to
// standard request preparation.
func (p *OllamaProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	// Ollama doesn't support JSON schema validation natively
	// We'll just use the regular PrepareRequest method
	return p.PrepareRequest(prompt, options)
}

// ParseResponse extracts the generated text from the Ollama API response.
// It handles Ollama's streaming response format and concatenates the results.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *OllamaProvider) ParseResponse(body []byte) (string, error) {
	var fullResponse strings.Builder
	decoder := json.NewDecoder(bytes.NewReader(body))

	for decoder.More() {
		var response struct {
			Model    string `json:"model"`
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := decoder.Decode(&response); err != nil {
			return "", fmt.Errorf("error parsing Ollama response: %w", err)
		}
		fullResponse.WriteString(response.Response)
		if response.Done {
			break
		}
	}

	return fullResponse.String(), nil
}

// HandleFunctionCalls processes function calling capabilities.
// Since Ollama doesn't support function calling natively, this returns nil.
func (p *OllamaProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	// Ollama doesn't support function calling natively, so we return nil
	return nil, nil
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *OllamaProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SetEndpoint configures the base URL for the Ollama API.
// By default, this points to the local Ollama instance.
func (p *OllamaProvider) SetEndpoint(endpoint string) {
	p.endpoint = endpoint
}

// Generate sends a completion request to the Ollama API and returns the generated text.
// It handles the full request lifecycle including context management and error handling.
//
// Parameters:
//   - ctx: Context for request cancellation and timeouts
//   - prompt: The input text to generate from
//
// Returns:
//   - Generated text
//   - Original prompt
//   - Any error encountered
func (p *OllamaProvider) Generate(ctx context.Context, prompt string) (string, string, error) {
	reqBody, err := p.PrepareRequest(prompt, p.options)
	if err != nil {
		return "", "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", "", err
	}

	for k, v := range p.Headers() {
		req.Header.Set(k, v)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}

	result, err := p.ParseResponse(body)
	if err != nil {
		return "", "", err
	}

	return result, prompt, nil
}

// SetDebugLevel sets the logging level for the provider.
// This controls the verbosity of debug output.
func (p *OllamaProvider) SetDebugLevel(level utils.LogLevel) {
	if p.logger != nil {
		p.logger.SetLevel(level)
	}
}
