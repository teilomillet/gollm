// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// AnthropicProvider implements the Provider interface for Anthropic's Claude API.
// It supports Claude models and provides access to Anthropic's language model capabilities,
// including structured output and system prompts.
type AnthropicProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "claude-3-opus", "claude-3-sonnet")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewAnthropicProvider creates a new Anthropic provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Anthropic API key for authentication
//   - model: The model to use (e.g., "claude-3-opus", "claude-3-sonnet")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Anthropic Provider instance
func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo), // Default logger
	}

	// Copy the provided extraHeaders
	for k, v := range extraHeaders {
		provider.extraHeaders[k] = v
	}

	// Add the caching header if it's not already present
	if _, exists := provider.extraHeaders["anthropic-beta"]; !exists {
		provider.extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	return provider
}

// SetLogger configures the logger for the Anthropic provider.
// This is used for debugging and monitoring API interactions.
func (p *AnthropicProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Anthropic provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
//   - stop_sequences: Custom stop sequences
func (p *AnthropicProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *AnthropicProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

// Name returns "anthropic" as the provider identifier.
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// Endpoint returns the Anthropic API endpoint URL.
// For API version 2024-02-15, this is "https://api.anthropic.com/v1/messages".
func (p *AnthropicProvider) Endpoint() string {
	return "https://api.anthropic.com/v1/messages"
}

// SupportsJSONSchema indicates that Anthropic supports structured output
// through its system prompts and response formatting capabilities.
func (p *AnthropicProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for Anthropic API requests.
// This includes:
//   - x-api-key: API key for authentication
//   - anthropic-version: API version identifier
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *AnthropicProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "prompt-caching-2024-07-31",
	}
	return headers
}

// PrepareRequest creates the request body for an Anthropic API call.
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
func (p *AnthropicProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": p.options["max_tokens"],
		"system":     []map[string]interface{}{},
		"messages":   []map[string]interface{}{},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"]; ok {
		if sp, ok := systemPrompt.(string); ok && sp != "" {
			parts := splitSystemPrompt(sp, 3)
			for i, part := range parts {
				systemMessage := map[string]interface{}{
					"type": "text",
					"text": part,
				}
				if i > 0 {
					systemMessage["cache_control"] = map[string]string{"type": "ephemeral"}
				}
				requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
			}
		}
	}

	// Handle user message with potential caching
	userMessage := map[string]interface{}{
		"role": "user",
		"content": []map[string]interface{}{
			{
				"type": "text",
				"text": prompt,
			},
		},
	}

	// Add cache_control only if caching is enabled
	if caching, ok := options["enable_caching"].(bool); ok && caching {
		userMessage["content"].([]map[string]interface{})[0]["cache_control"] = map[string]string{"type": "ephemeral"}
	}

	requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), userMessage)

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		anthropicTools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			anthropicTools[i] = map[string]interface{}{
				"name":         tool.Function.Name,
				"description":  tool.Function.Description,
				"input_schema": tool.Function.Parameters,
			}
		}
		requestBody["tools"] = anthropicTools
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		requestBody["tool_choice"] = map[string]interface{}{
			"type": toolChoice,
		}
	}

	// Add other options
	for k, v := range options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" && k != "enable_caching" {
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
}

// Helper function to split the system prompt into a maximum of n parts
func splitSystemPrompt(prompt string, n int) []string {
	if n <= 1 {
		return []string{prompt}
	}

	// Split the prompt into paragraphs
	paragraphs := strings.Split(prompt, "\n\n")

	if len(paragraphs) <= n {
		return paragraphs
	}

	// If we have more paragraphs than allowed parts, we need to combine some
	result := make([]string, n)
	paragraphsPerPart := len(paragraphs) / n
	extraParagraphs := len(paragraphs) % n

	currentIndex := 0
	for i := 0; i < n; i++ {
		end := currentIndex + paragraphsPerPart
		if i < extraParagraphs {
			end++
		}
		result[i] = strings.Join(paragraphs[currentIndex:end], "\n\n")
		currentIndex = end
	}

	return result
}

// PrepareRequestWithSchema creates a request that includes structured output formatting.
// This uses Anthropic's system prompts to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	// Create a system message that enforces the JSON schema
	systemMsg := fmt.Sprintf("You must respond with a JSON object that strictly adheres to this schema:\n%s\nDo not include any explanatory text, only output valid JSON.", string(schemaJSON))

	requestBody := map[string]interface{}{
		"model":  p.model,
		"system": systemMsg,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	// Add any additional options
	for k, v := range options {
		if k != "system_prompt" { // Skip system_prompt as we're using it for schema
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
}

// ParseResponse extracts the generated text from the Anthropic API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *AnthropicProvider) ParseResponse(body []byte) (string, error) {
	p.logger.Debug("Raw API response: %s", string(body))

	var response struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type  string          `json:"type"`
			Text  string          `json:"text,omitempty"`
			ID    string          `json:"id,omitempty"`
			Name  string          `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		StopReason string  `json:"stop_reason"`
		StopSeq    *string `json:"stop_sequence"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}
	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	var finalResponse strings.Builder
	var functionCalls []string

	for _, content := range response.Content {
		switch content.Type {
		case "text":
			finalResponse.WriteString(content.Text)
			p.logger.Debug("Text content: %s", content.Text)
		case "tool_use":
			functionCall, err := json.Marshal(map[string]interface{}{
				"name":      content.Name,
				"arguments": content.Input,
			})
			if err != nil {
				return "", fmt.Errorf("error marshaling function call: %w", err)
			}
			functionCalls = append(functionCalls, string(functionCall))
			p.logger.Debug("Function call detected: %s", string(functionCall))
		}
	}

	// If there are function calls, append them to the response
	if len(functionCalls) > 0 {
		for _, call := range functionCalls {
			finalResponse.WriteString("\n<function_call>")
			finalResponse.WriteString(call)
			finalResponse.WriteString("</function_call>")
			p.logger.Debug("Appending function call to response")
		}
	}

	p.logger.Debug("Final response: %s", finalResponse.String())
	return finalResponse.String(), nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Anthropic's response formatting capabilities.
func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls from response")
	var response string
	err := json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %w", err)
	}

	p.logger.Debug("Response body: %s", response)

	// Extract function calls from the response
	var functionCalls []map[string]interface{}
	functionCallRegex := regexp.MustCompile(`<function_call>(.*?)</function_call>`)
	matches := functionCallRegex.FindAllStringSubmatch(response, -1)

	for _, match := range matches {
		if len(match) > 1 {
			var functionCall map[string]interface{}
			err := json.Unmarshal([]byte(match[1]), &functionCall)
			if err != nil {
				return nil, fmt.Errorf("error unmarshaling function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Extracted function call: %v", functionCall)
		}
	}

	if len(functionCalls) == 0 {
		p.logger.Debug("No function calls found in the response")
		return nil, nil // No function calls
	}

	p.logger.Debug("Function calls to handle: %v", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *AnthropicProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}
