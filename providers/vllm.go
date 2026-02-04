// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// VLLMProvider implements the Provider interface for vLLM's OpenAI-compatible API.
// It supports local models served via vLLM without requiring API key authentication.
type VLLMProvider struct {
	baseURL      string                 // Base URL for vLLM server (e.g., "http://192.168.1.17:8000/v1")
	model        string                 // Model identifier (e.g., "eduMind-6.7b")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewVLLMProvider creates a new vLLM provider instance.
// It initializes the provider with the given base URL, model, and optional headers.
//
// Parameters:
//   - baseURL: vLLM server base URL (e.g., "http://192.168.1.17:8000/v1")
//   - model: The model to use (e.g., "eduMind-6.7b")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured vLLM Provider instance
func NewVLLMProvider(baseURL, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &VLLMProvider{
		baseURL:      baseURL,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the vLLM provider.
func (p *VLLMProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the vLLM provider.
func (p *VLLMProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *VLLMProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}

// Name returns "vllm" as the provider identifier.
func (p *VLLMProvider) Name() string {
	return "vllm"
}

// Endpoint returns the vLLM API endpoint URL.
// It normalizes the base URL to ensure proper formatting.
func (p *VLLMProvider) Endpoint() string {
	baseURL := p.baseURL

	// Remove trailing slash if present
	if len(baseURL) > 0 && baseURL[len(baseURL)-1] == '/' {
		baseURL = baseURL[:len(baseURL)-1]
	}

	// Ensure base URL ends with /v1 if not already present
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = baseURL + "/v1"
	}

	return baseURL + "/chat/completions"
}

// SupportsJSONSchema indicates that vLLM supports JSON schema validation
func (p *VLLMProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for vLLM API requests.
// Note: vLLM does NOT require Authorization header for local models
func (p *VLLMProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type": "application/json",
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for a vLLM API call.
func (p *VLLMProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Add user message
	request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
		"role":    "user",
		"content": prompt,
	})

	// Merge options
	for k, v := range p.options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	for k, v := range options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
func (p *VLLMProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	// vLLM supports OpenAI-compatible JSON schema
	var schemaObj interface{}
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema bytes: %w", err)
		}
	case map[string]interface{}:
		schemaObj = s
	default:
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
		}
	}

	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type": "json_object",
		},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append([]map[string]interface{}{
			{"role": "system", "content": systemPrompt},
		}, request["messages"].([]map[string]interface{})...)
	}

	// Merge options
	for k, v := range p.options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	for k, v := range options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}

// ParseResponse extracts the generated text from the vLLM API response.
func (p *VLLMProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Message.Content, nil
}

// HandleFunctionCalls processes function calling in the response.
func (p *VLLMProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	// vLLM may support function calling depending on the model
	return nil, fmt.Errorf("function calling not implemented for vLLM")
}

// SetExtraHeaders configures additional HTTP headers for API requests.
func (p *VLLMProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming indicates whether streaming is supported
func (p *VLLMProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *VLLMProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	}

	// Merge options
	for k, v := range p.options {
		if k != "stream" {
			requestBody[k] = v
		}
	}

	for k, v := range options {
		if k != "stream" {
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *VLLMProvider) ParseStreamResponse(chunk []byte) (string, error) {
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return "", io.EOF
	}

	var response struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(chunk, &response); err != nil {
		return "", fmt.Errorf("malformed response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	if response.Choices[0].FinishReason != "" {
		return "", io.EOF
	}

	return response.Choices[0].Delta.Content, nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
func (p *VLLMProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Convert MemoryMessage objects to vLLM messages format
	for _, msg := range messages {
		message := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		request["messages"] = append(request["messages"].([]map[string]interface{}), message)
	}

	// Merge options
	for k, v := range p.options {
		if k != "system_prompt" && k != "structured_messages" {
			request[k] = v
		}
	}

	for k, v := range options {
		if k != "system_prompt" && k != "structured_messages" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}
