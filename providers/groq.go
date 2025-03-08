// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// GroqProvider implements the Provider interface for Groq's API.
// It supports Groq's optimized language models and provides access to their
// high-performance inference capabilities.
type GroqProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "llama2-70b", "mixtral-8x7b")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewGroqProvider creates a new Groq provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Groq API key for authentication
//   - model: The model to use (e.g., "llama2-70b", "mixtral-8x7b")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Groq Provider instance
func NewGroqProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &GroqProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Groq provider.
// This is used for debugging and monitoring API interactions.
func (p *GroqProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// Name returns the identifier for this provider ("groq").
func (p *GroqProvider) Name() string {
	return "groq"
}

// Endpoint returns the Groq API endpoint URL.
// This is "https://api.groq.com/openai/v1/chat/completions".
func (p *GroqProvider) Endpoint() string {
	return "https://api.groq.com/openai/v1/chat/completions"
}

// SetOption sets a model-specific option for the Groq provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
func (p *GroqProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *GroqProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

// SupportsJSONSchema indicates whether this provider supports JSON schema validation.
// Currently, Groq does not natively support JSON schema validation.
func (p *GroqProvider) SupportsJSONSchema() bool {
	return false
}

// Headers returns the HTTP headers required for Groq API requests.
// This includes the authorization token and content type headers.
func (p *GroqProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

// PrepareRequest creates the request body for a Groq API call.
// It formats the prompt and options according to Groq's API requirements.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *GroqProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	// First, add the default options
	for k, v := range p.options {
		requestBody[k] = v
	}

	// Then, add any additional options (which may override defaults)
	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// PrepareRequestWithSchema creates a request with JSON schema validation.
// Since Groq doesn't support schema validation natively, this falls back to
// standard request preparation.
func (p *GroqProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type":   "json_schema",
			"schema": schema,
		},
	}

	// Add any additional options
	for k, v := range options {
		requestBody[k] = v
	}

	// Add strict option if provided
	if strict, ok := options["strict"].(bool); ok && strict {
		requestBody["response_format"].(map[string]interface{})["strict"] = true
	}

	return json.Marshal(requestBody)
}

// ParseResponse extracts the generated text from the Groq API response.
// It handles Groq's response format and extracts the content.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *GroqProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	err := json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Message.Content, nil
}

// HandleFunctionCalls processes function calling capabilities.
// Since Groq doesn't support function calling natively, this returns nil.
func (p *GroqProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		return nil, nil // No function calls found
	}

	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *GroqProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SupportsStreaming returns whether the provider supports streaming responses
func (p *GroqProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest prepares a request body for streaming
func (p *GroqProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	options["stream"] = true
	return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *GroqProvider) ParseStreamResponse(chunk []byte) (string, error) {
	var response struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(chunk, &response); err != nil {
		return "", err
	}
	if len(response.Choices) == 0 {
		return "", nil
	}
	return response.Choices[0].Delta.Content, nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string.
func (p *GroqProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Add system prompt if present
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Convert structured messages to Groq format (OpenAI compatible)
	for _, msg := range messages {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	// Add other options from provider and request
	for k, v := range p.options {
		if k != "messages" {
			request[k] = v
		}
	}
	for k, v := range options {
		if k != "messages" && k != "system_prompt" && k != "structured_messages" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}
