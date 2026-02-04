// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// LambdaProvider implements the Provider interface for Lambda Labs' API.
// Lambda Labs provides access to various open-source models through an OpenAI-compatible API.
type LambdaProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "hermes-3-llama-3.1-405b-fp8")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewLambdaProvider creates a new Lambda Labs provider instance.
//
// Parameters:
//   - apiKey: Lambda Labs API key for authentication
//   - model: The model to use (e.g., "hermes-3-llama-3.1-405b-fp8", "llama3.1-70b-instruct-fp8")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Lambda Labs Provider instance
func NewLambdaProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &LambdaProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Lambda Labs provider.
func (p *LambdaProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// Name returns "lambda" as the provider identifier.
func (p *LambdaProvider) Name() string {
	return "lambda"
}

// Endpoint returns the Lambda Labs API endpoint URL.
func (p *LambdaProvider) Endpoint() string {
	return "https://api.lambdalabs.com/v1/chat/completions"
}

// SupportsJSONSchema indicates that Lambda Labs supports JSON schema validation
// through the OpenAI-compatible API.
func (p *LambdaProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for Lambda Labs API requests.
func (p *LambdaProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}
	for key, value := range p.extraHeaders {
		headers[key] = value
	}
	return headers
}

// SetOption sets a specific option for the Lambda Labs provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - frequency_penalty: Repetition reduction
//   - presence_penalty: Topic steering
func (p *LambdaProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *LambdaProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

// PrepareRequest creates the request body for a Lambda Labs API call.
func (p *LambdaProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
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

	// Add provider options
	for k, v := range p.options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	// Add request-specific options
	for k, v := range options {
		if k != "system_prompt" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
func (p *LambdaProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   "structured_response",
				"schema": schema,
				"strict": true,
			},
		},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append([]map[string]interface{}{
			{"role": "system", "content": systemPrompt},
		}, request["messages"].([]map[string]interface{})...)
	}

	// Add options
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

// ParseResponse extracts the generated text from the Lambda Labs API response.
func (p *LambdaProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if response.Error != nil && response.Error.Message != "" {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Message.Content, nil
}

// HandleFunctionCalls processes function calling capabilities.
func (p *LambdaProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}
	if len(functionCalls) == 0 {
		return nil, nil
	}
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
func (p *LambdaProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SupportsStreaming indicates whether streaming is supported.
func (p *LambdaProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls.
func (p *LambdaProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	options["stream"] = true
	return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse processes a single chunk from a streaming response.
func (p *LambdaProvider) ParseStreamResponse(chunk []byte) (string, error) {
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

// PrepareRequestWithMessages creates a request body using structured message objects.
func (p *LambdaProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
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

	// Convert MemoryMessage objects to Lambda Labs format (OpenAI-compatible)
	for _, msg := range messages {
		message := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		request["messages"] = append(request["messages"].([]map[string]interface{}), message)
	}

	// Add options
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
