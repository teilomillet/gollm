package providers

import (
	"encoding/json"
	"fmt"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
	"strings"
)

// CohereProvider implements the Provider interface for Cohere's API.
// It supports Cohere's language models and provides access to their capabilities,
// including chat completion and structured output
type CohereProvider struct {
	apiKey       string            // API key for authentication
	model        string            // Model identifier (e.g., "command-r-plus-08-2024", "command-r-plus-04-2024")
	extraHeaders map[string]string // Additional HTTP headers
	options      map[string]any    // Model-specific options
	logger       utils.Logger      // Logger instance
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
func NewCohereProvider(apiKey, model string, extraHeaders map[string]string) Provider {
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
func (p *CohereProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	p.SetOption("stream", false)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
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
			{"role": "user", "content": prompt},
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

	return json.Marshal(requestBody)
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
			{"role": "user", "content": prompt},
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

	return json.Marshal(requestBody)
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
func (p *CohereProvider) ParseResponse(body []byte) (string, error) {
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
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Message.Content) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	var finalResponse strings.Builder

	for _, content := range response.Message.Content {
		switch content.Type {
		case "text":
			finalResponse.WriteString(content.Text)
			p.logger.Debug("Text content: %s", content.Text)
		}
	}

	for _, toolCall := range response.Message.ToolCalls {
		finalResponse.WriteString(
			fmt.Sprintf("<function_call>{\"name\": \"%s\", \"arguments\": %s}</function_call>",
				toolCall.Function.Name, toolCall.Function.Arguments))
	}

	p.logger.Debug("Final response: %s", finalResponse.String())
	return finalResponse.String(), nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Cohere's response formatting capabilities.
func (p *CohereProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
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

	if len(response.Message.ToolCalls) == 0 {
		return nil, fmt.Errorf("no tool calls found in response")
	}

	toolCalls := response.Message.ToolCalls
	result := make([]map[string]any, len(toolCalls))
	for i, call := range toolCalls {
		var args map[string]any
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("error parsing arguments: %w", err)
		}
		result[i] = map[string]any{
			"name":      call.Function.Name,
			"arguments": args,
		}
	}

	return json.Marshal(result)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *CohereProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}
