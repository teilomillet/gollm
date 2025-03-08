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
	systemPrompt := ""
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		systemPrompt = sp
	}

	// If we have tools, add tool usage instructions to the system prompt
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

		// Add tool usage instructions to system prompt
		if len(tools) > 1 {
			toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
			if systemPrompt != "" {
				systemPrompt = toolUsagePrompt + "\n\n" + systemPrompt
			} else {
				systemPrompt = toolUsagePrompt
			}
		}

		// Only set tool_choice when tools are provided
		if toolChoice, ok := options["tool_choice"].(string); ok {
			requestBody["tool_choice"] = map[string]interface{}{
				"type": toolChoice,
			}
		} else {
			// Default to auto for tool choice when tools are provided
			requestBody["tool_choice"] = map[string]interface{}{
				"type": "auto",
			}
		}
	}

	// Add system prompt if we have one
	if systemPrompt != "" {
		parts := splitSystemPrompt(systemPrompt, 3)
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

	p.logger.Debug("Number of content blocks: %d", len(response.Content))
	p.logger.Debug("Stop reason: %s", response.StopReason)

	var finalResponse strings.Builder
	var functionCalls []string
	var pendingText strings.Builder
	var lastType string

	// First pass: collect all function calls and text
	for i, content := range response.Content {
		p.logger.Debug("Processing content block %d: type=%s", i, content.Type)

		switch content.Type {
		case "text":
			// If we have pending text and this is also text, add a space
			if lastType == "text" && pendingText.Len() > 0 {
				pendingText.WriteString(" ")
			}
			pendingText.WriteString(content.Text)
			p.logger.Debug("Added text content: %s", content.Text)

		case "tool_use", "tool_calls":
			// If we have any pending text, add it to the final response
			if pendingText.Len() > 0 {
				if finalResponse.Len() > 0 {
					finalResponse.WriteString("\n")
				}
				finalResponse.WriteString(pendingText.String())
				pendingText.Reset()
			}

			// Parse input as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(content.Input, &args); err != nil {
				p.logger.Debug("Error parsing tool input: %v, raw input: %s", err, string(content.Input))
				return "", fmt.Errorf("error parsing tool input: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(content.Name, args)
			if err != nil {
				p.logger.Debug("Error formatting function call: %v", err)
				return "", fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Added function call: %s", functionCall)
		}
		lastType = content.Type
	}

	// Add any remaining pending text
	if pendingText.Len() > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(pendingText.String())
	}

	p.logger.Debug("Number of function calls collected: %d", len(functionCalls))
	for i, call := range functionCalls {
		p.logger.Debug("Function call %d: %s", i, call)
	}

	// Add all function calls at the end
	if len(functionCalls) > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(strings.Join(functionCalls, "\n"))
	}

	result := finalResponse.String()
	p.logger.Debug("Final response: %s", result)
	return result, nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Anthropic's response formatting capabilities.
func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls from response")
	response := string(body)

	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		p.logger.Debug("No function calls found in the response")
		return nil, nil
	}

	p.logger.Debug("Function calls to handle: %v", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *AnthropicProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SupportsStreaming indicates whether streaming is supported
func (p *AnthropicProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *AnthropicProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  p.model,
		"stream": true,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"max_tokens": 1024, // Default max tokens
	}

	// Add system prompt if present
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		requestBody["system"] = systemPrompt
		delete(options, "system_prompt")
	}

	// Add max tokens if present
	if maxTokens, ok := options["max_tokens"].(int); ok {
		requestBody["max_tokens"] = maxTokens
		delete(options, "max_tokens")
	}

	// Add temperature if present
	if temperature, ok := options["temperature"].(float64); ok {
		requestBody["temperature"] = temperature
		delete(options, "temperature")
	}

	// Add other options
	for k, v := range options {
		if k != "stream" { // Don't override stream setting
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *AnthropicProvider) ParseStreamResponse(chunk []byte) (string, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	// Check for [DONE] marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return "", io.EOF
	}

	// Parse the event
	var event struct {
		Type  string `json:"type"`
		Index int    `json:"index"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
	}

	if err := json.Unmarshal(chunk, &event); err != nil {
		return "", fmt.Errorf("malformed event: %w", err)
	}

	// Handle different event types
	switch event.Type {
	case "content_block_delta":
		if event.Delta.Type == "text_delta" {
			if event.Delta.Text == "" {
				return "", fmt.Errorf("skip token")
			}
			return event.Delta.Text, nil
		}
		return "", fmt.Errorf("skip token")
	case "message_stop":
		return "", io.EOF
	default:
		return "", fmt.Errorf("skip token")
	}
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching and
// better preserves conversation structure for the Claude API.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": p.options["max_tokens"],
		"system":     []map[string]interface{}{},
		"messages":   []map[string]interface{}{},
	}

	// Extract system prompt if present in options
	systemPrompt := ""
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		systemPrompt = sp
	}

	// Handle system prompt
	if systemPrompt != "" {
		parts := splitSystemPrompt(systemPrompt, 3)
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

	// Process tools if present
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

		// Add tool usage instructions to system prompt if needed
		if len(tools) > 1 {
			toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
			// This is separate from the existing system messages
			systemMessage := map[string]interface{}{
				"type": "text",
				"text": toolUsagePrompt,
			}
			requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
		}

		// Only set tool_choice when tools are provided
		if toolChoice, ok := options["tool_choice"].(string); ok {
			requestBody["tool_choice"] = map[string]interface{}{
				"type": toolChoice,
			}
		} else {
			// Default to auto for tool choice when tools are provided
			requestBody["tool_choice"] = map[string]interface{}{
				"type": "auto",
			}
		}
	}

	// Convert MemoryMessage objects to Anthropic messages
	for _, msg := range messages {
		content := []map[string]interface{}{
			{
				"type": "text",
				"text": msg.Content,
			},
		}

		// Add cache_control if specified
		if msg.CacheControl != "" {
			content[0]["cache_control"] = map[string]string{"type": msg.CacheControl}
		} else if caching, ok := options["enable_caching"].(bool); ok && caching {
			// Add default caching if enabled globally
			content[0]["cache_control"] = map[string]string{"type": "ephemeral"}
		}

		message := map[string]interface{}{
			"role":    msg.Role,
			"content": content,
		}

		requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), message)
	}

	// Add other options
	for k, v := range options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" && k != "enable_caching" && k != "structured_messages" {
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
}
