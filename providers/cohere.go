package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

const (
	cohereKeyText               = "text"
	cohereKeyUser               = "user"
	cohereKeyStructuredMessages = "structured_messages"
)

// CohereProvider implements the Provider interface for Cohere's API.
// It supports Cohere's language models and provides access to their capabilities,
// including chat completion and structured output
type CohereProvider struct {
	logger       utils.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewCohereProvider creates a new Cohere provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Cohere API key for authentication
//   - model: The model to use (e.g., "command-r-plus-08-2024", "command-r-plus-04-2024")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Cohere Provider instance
func NewCohereProvider(apiKey, model string, extraHeaders map[string]string) *CohereProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	return &CohereProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Cohere provider.
// This is used for debugging and monitoring API interactions.
func (p *CohereProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Cohere provider.
// Support options include:
//   - temperature: Controls randomness
//   - max_tokens: Maximum tokens in the response
//   - p: Total probability mass (0.01 to 0.99)
//   - k: Top k most likely tokens are considered
//   - strict_tools: If set to true, follow tool definition strictly
func (p *CohereProvider) SetOption(key string, value any) {
	p.options[key] = value
	if p.logger != nil {
		p.logger.Debug("Setting option for Cohere", "key", key, "value", value)
	}
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *CohereProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	p.SetOption("stream", false)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
}

// Name returns "cohere" as the provider identifier.
func (p *CohereProvider) Name() string {
	return "cohere"
}

// Endpoint returns the base URL for the Cohere API.
// This is "https://api.cohere.com/v2/chat".
func (p *CohereProvider) Endpoint() string {
	return "https://api.cohere.com/v2/chat"
}

// SupportsJSONSchema indicates that Cohere supports structured output
// through its system prompts and response formatting capabilities.
func (p *CohereProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for Cohere API requests.
// This includes:
//   - Content-type: application/json
//   - Authorization: Bearer token using the API key
//   - Any additional headers specified via SetExtraHeaders
func (p *CohereProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

// PrepareRequest creates the request body for a Cohere API call.
// It handles:
//   - Message formatting
//   - System prompts
//   - Response formatting
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *CohereProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
	requestBody := map[string]any{
		"model": p.model,
		"messages": []map[string]any{
			{"role": cohereKeyUser, "content": prompt},
		},
	}

	// First, add default options
	for k, v := range p.options {
		requestBody[k] = v
	}

	// Then, add any additional options (which may override defaults)
	for k, v := range options {
		requestBody[k] = v
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// PrepareRequestWithSchema creates a request that includes structured output formatting.
// This uses Cohere's system prompts to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *CohereProvider) PrepareRequestWithSchema(prompt string, options map[string]any, schema any) ([]byte, error) {
	requestBody := map[string]any{
		"model": p.model,
		"messages": []map[string]any{
			{"role": cohereKeyUser, "content": prompt},
		},
		"response_format": map[string]any{
			"type":        "json_object",
			"json_schema": schema,
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

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseResponse extracts the generated text from the Cohere API response.
// It handles various response formats and error cases
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *CohereProvider) ParseResponse(body []byte) (*Response, error) {
	var response struct {
		Message struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Message.Content) == 0 {
		return nil, errors.New("empty response from API")
	}

	var finalResponse strings.Builder

	for _, content := range response.Message.Content {
		if content.Type == cohereKeyText {
			finalResponse.WriteString(content.Text)
			p.logger.Debug("Text content: %s", content.Text)
		}
	}

	for _, toolCall := range response.Message.ToolCalls {
		// Parse arguments as raw JSON to preserve the exact format
		var args any
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("error parsing function arguments: %w", err)
		}

		functionCall, err := FormatFunctionCall(toolCall.Function.Name, args)
		if err != nil {
			return nil, fmt.Errorf("error formatting function call: %w", err)
		}
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(functionCall)
	}

	p.logger.Debug("Final response: %s", finalResponse.String())
	return &Response{Content: Text{Value: finalResponse.String()}}, nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Cohere's response formatting capabilities.
func (p *CohereProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		return nil, nil // No function calls found
	}

	data, err := json.Marshal(functionCalls)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal function calls: %w", err)
	}
	return data, nil
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *CohereProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming returns whether the provider supports streaming responses
func (p *CohereProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest prepares a request body for streaming
func (p *CohereProvider) PrepareStreamRequest(prompt string, options map[string]any) ([]byte, error) {
	options["stream"] = true
	return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *CohereProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	var response struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(chunk, &response); err != nil {
		return nil, fmt.Errorf("malformed response: %w", err)
	}
	if response.Text == "" {
		return nil, errors.New("skip resp")
	}
	return &Response{Content: Text{Value: response.Text}}, nil
}

// PrepareRequestWithMessages creates a request using structured message objects.
func (p *CohereProvider) PrepareRequestWithMessages(
	messages []types.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	// Cohere uses a chat history format
	chatHistory := []map[string]any{}
	var userMessage string

	// Process messages and build chat history
	for i, msg := range messages {
		if i == len(messages)-1 && msg.Role == cohereKeyUser {
			// Last user message goes in the message field
			userMessage = msg.Content
		} else {
			// Previous messages go into chat history
			chatHistory = append(chatHistory, map[string]any{
				"role":    msg.Role,
				"message": msg.Content,
			})
		}
	}

	// Build request
	request := map[string]any{
		"model":        p.model,
		"message":      userMessage,
		"chat_history": chatHistory,
	}

	// Add other options
	for k, v := range p.options {
		if k != "message" && k != "chat_history" {
			request[k] = v
		}
	}
	for k, v := range options {
		if k != "message" && k != "chat_history" && k != "system_prompt" && k != cohereKeyStructuredMessages {
			request[k] = v
		}
	}

	// Add system prompt if present
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["preamble"] = systemPrompt
	}

	data, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	return data, nil
}
