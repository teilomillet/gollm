// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// Common parameter keys
const (
	keyMaxTokens = "max_tokens"
	keyStream    = "stream"
	keyText      = "text"
)

// Anthropic-specific parameter keys
const (
	anthropicKeySystemPrompt       = KeySystemPrompt
	anthropicKeyTools              = KeyTools
	anthropicKeyToolChoice         = KeyToolChoice
	anthropicKeyEnableCaching      = "enable_caching"
	anthropicKeyStructuredMessages = KeyStructuredMessages
)

// AnthropicProvider implements the Provider interface for Anthropic's Claude API.
// It supports Claude models and provides access to Anthropic's language model capabilities,
// including structured output and system prompts.
type AnthropicProvider struct {
	logger       utils.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
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
func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) *AnthropicProvider {
	provider := &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]any),
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
func (p *AnthropicProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *AnthropicProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption(keyMaxTokens, cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
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
func (p *AnthropicProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	// Process system prompt and tools
	systemPrompt := p.extractSystemPrompt(options)
	systemPrompt = p.handleToolsForRequest(requestBody, systemPrompt, options)
	p.addSystemPromptToRequestBody(requestBody, systemPrompt)

	// Handle user message
	userMessage := p.createUserMessage(prompt, options)
	p.addMessageToRequestBody(requestBody, userMessage)

	// Add other options
	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// initializeRequestBody creates the base request structure
func (p *AnthropicProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		"model":      p.model,
		keyMaxTokens: p.options[keyMaxTokens],
		"system":     []map[string]any{},
		"messages":   []map[string]any{},
	}
}

// extractSystemPrompt gets the system prompt from options
func (p *AnthropicProvider) extractSystemPrompt(options map[string]any) string {
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		return sp
	}
	return ""
}

// handleToolsForRequest processes tools and updates system prompt if needed
func (p *AnthropicProvider) handleToolsForRequest(
	requestBody map[string]any,
	systemPrompt string,
	options map[string]any,
) string {
	tools, ok := options[anthropicKeyTools].([]types.Tool)
	if !ok || len(tools) == 0 {
		return systemPrompt
	}
	return p.processTools(tools, requestBody, systemPrompt, options)
}

// addSystemPromptToRequestBody adds the system prompt to the request
func (p *AnthropicProvider) addSystemPromptToRequestBody(requestBody map[string]any, systemPrompt string) {
	if systemPrompt == "" {
		return
	}

	parts := splitSystemPrompt(systemPrompt, AnthropicSystemPromptMaxParts)
	for i, part := range parts {
		systemMessage := map[string]any{
			"type": "text",
			"text": part,
		}
		if i > 0 {
			systemMessage["cache_control"] = map[string]string{"type": "ephemeral"}
		}
		if systemArray, ok := requestBody["system"].([]map[string]any); ok {
			requestBody["system"] = append(systemArray, systemMessage)
		}
	}
}

// createUserMessage creates a user message with optional caching
func (p *AnthropicProvider) createUserMessage(prompt string, options map[string]any) map[string]any {
	userMessage := map[string]any{
		"role": "user",
		"content": []map[string]any{
			{
				"type": "text",
				"text": prompt,
			},
		},
	}

	// Add cache_control only if caching is enabled
	if caching, ok := options["enable_caching"].(bool); ok && caching {
		if contentArray, ok := userMessage["content"].([]map[string]any); ok && len(contentArray) > 0 {
			contentArray[0]["cache_control"] = map[string]string{"type": "ephemeral"}
		}
	}

	return userMessage
}

// addMessageToRequestBody adds a message to the request body
func (p *AnthropicProvider) addMessageToRequestBody(requestBody map[string]any, message map[string]any) {
	if messagesArray, ok := requestBody["messages"].([]map[string]any); ok {
		requestBody["messages"] = append(messagesArray, message)
	}
}

// addRemainingOptions adds non-handled options to the request
func (p *AnthropicProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	for k, v := range options {
		if p.isHandledAnthropicOption(k) {
			continue
		}
		requestBody[k] = v
	}
}

// isHandledAnthropicOption checks if an option is already handled
func (p *AnthropicProvider) isHandledAnthropicOption(key string) bool {
	return key == anthropicKeySystemPrompt ||
		key == keyMaxTokens ||
		key == anthropicKeyTools ||
		key == anthropicKeyToolChoice ||
		key == anthropicKeyEnableCaching
}

// processTools handles tool configuration and updates system prompt
func (p *AnthropicProvider) processTools(
	tools []types.Tool,
	requestBody map[string]any,
	systemPrompt string,
	options map[string]any,
) string {
	anthropicTools := make([]map[string]any, len(tools))
	for i, tool := range tools {
		anthropicTools[i] = map[string]any{
			"name":         tool.Function.Name,
			"description":  tool.Function.Description,
			"input_schema": tool.Function.Parameters,
		}
	}
	requestBody[anthropicKeyTools] = anthropicTools

	// Add tool usage instructions to system prompt for multiple tools
	if len(tools) > 1 {
		toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
		if systemPrompt != "" {
			systemPrompt = toolUsagePrompt + "\n\n" + systemPrompt
		} else {
			systemPrompt = toolUsagePrompt
		}
	}

	// Set tool choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		requestBody["tool_choice"] = map[string]any{
			"type": toolChoice,
		}
	} else {
		// Default to auto for tool choice when tools are provided
		requestBody["tool_choice"] = map[string]any{
			"type": "auto",
		}
	}

	return systemPrompt
}

// processToolsForMessages handles tool configuration for message-based requests
func (p *AnthropicProvider) processToolsForMessages(
	tools []types.Tool,
	requestBody map[string]any,
	options map[string]any,
) {
	anthropicTools := make([]map[string]any, len(tools))
	for i, tool := range tools {
		anthropicTools[i] = map[string]any{
			"name":         tool.Function.Name,
			"description":  tool.Function.Description,
			"input_schema": tool.Function.Parameters,
		}
	}
	requestBody[anthropicKeyTools] = anthropicTools

	// Add tool usage instructions to system prompt if needed
	if len(tools) > 1 {
		toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
		// This is separate from the existing system messages
		systemMessage := map[string]any{
			"type": "text",
			"text": toolUsagePrompt,
		}
		if systemArray, ok := requestBody["system"].([]map[string]any); ok {
			requestBody["system"] = append(systemArray, systemMessage)
		}
	}

	// Set tool choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		requestBody["tool_choice"] = map[string]any{
			"type": toolChoice,
		}
	} else {
		// Default to auto for tool choice when tools are provided
		requestBody["tool_choice"] = map[string]any{
			"type": "auto",
		}
	}
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
	for i := range n {
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
func (p *AnthropicProvider) PrepareRequestWithSchema(
	prompt string,
	options map[string]any,
	schema any,
) ([]byte, error) {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	// Create a system message that enforces the JSON schema
	systemMsg := fmt.Sprintf(
		"You must respond with a JSON object that strictly adheres to this schema:\n%s\nDo not include any explanatory text, only output valid JSON.",
		string(schemaJSON),
	)

	requestBody := map[string]any{
		"model":  p.model,
		"system": systemMsg,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	// Add any additional options
	for k, v := range options {
		if k != KeySystemPrompt { // Skip system_prompt as we're using it for schema
			requestBody[k] = v
		}
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// processAnthropicContent processes the content blocks from Anthropic response
func (p *AnthropicProvider) processAnthropicContent(contents []anthropicContent) (string, error) {
	var finalResponse strings.Builder
	var functionCalls []string
	var pendingText strings.Builder
	var lastType string

	// First pass: collect all function calls and text
	for i, content := range contents {
		p.logger.Debug("Processing content block %d: type=%s", i, content.Type)

		switch content.Type {
		case "text":
			p.processTextContent(&pendingText, content.Text, lastType)
			p.logger.Debug("Added text content: %s", content.Text)

		case "tool_use", "tool_calls":
			// Transfer pending text to final response
			p.transferPendingText(&finalResponse, &pendingText)

			// Process function call
			functionCall, err := p.processFunctionCall(&content)
			if err != nil {
				return "", err
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Added function call: %s", functionCall)
		}
		lastType = content.Type
	}

	// Add any remaining pending text
	p.transferPendingText(&finalResponse, &pendingText)

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

	return finalResponse.String(), nil
}

// processTextContent handles text content blocks
func (p *AnthropicProvider) processTextContent(pendingText *strings.Builder, text string, lastType string) {
	// If we have pending text and this is also text, add a space
	if lastType == "text" && pendingText.Len() > 0 {
		pendingText.WriteString(" ")
	}
	pendingText.WriteString(text)
}

// transferPendingText transfers pending text to final response
func (p *AnthropicProvider) transferPendingText(finalResponse, pendingText *strings.Builder) {
	if pendingText.Len() > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(pendingText.String())
		pendingText.Reset()
	}
}

// processFunctionCall processes a function call content block
func (p *AnthropicProvider) processFunctionCall(content *anthropicContent) (string, error) {
	// Parse input as raw JSON to preserve the exact format
	var args any
	if err := json.Unmarshal(content.Input, &args); err != nil {
		p.logger.Debug("Error parsing tool input: %v, raw input: %s", err, string(content.Input))
		return "", fmt.Errorf("error parsing tool input: %w", err)
	}

	functionCall, err := FormatFunctionCall(content.Name, args)
	if err != nil {
		p.logger.Debug("Error formatting function call: %v", err)
		return "", fmt.Errorf("error formatting function call: %w", err)
	}

	return functionCall, nil
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
func (p *AnthropicProvider) ParseResponse(body []byte) (*Response, error) {
	p.logger.Debug("Raw API anthropicResponse: %s", string(body))

	anthropicResponse := anthropicResponse{}

	if err := json.Unmarshal(body, &anthropicResponse); err != nil {
		return nil, fmt.Errorf("error parsing anthropicResponse: %w", err)
	}
	if len(anthropicResponse.Content) == 0 {
		return nil, errors.New("empty anthropicResponse from LLM")
	}

	p.logger.Debug("Number of content blocks: %d", len(anthropicResponse.Content))
	p.logger.Debug("Stop reason: %s", anthropicResponse.StopReason)

	// Process content blocks
	result, err := p.processAnthropicContent(anthropicResponse.Content)
	if err != nil {
		return nil, err
	}

	p.logger.Debug("Final anthropicResponse: %s", result)

	response := &Response{
		Content: Text{result},
		Usage: NewUsage(
			anthropicResponse.Usage.InputTokens,
			anthropicResponse.Usage.CacheCreationInputTokens,
			anthropicResponse.Usage.OutputTokens,
			anthropicResponse.Usage.CacheReadInputTokens,
		),
	}

	return response, nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Anthropic's response formatting capabilities.
func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls from response")
	response := string(body)

	functionCalls, err := ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		p.logger.Debug("No function calls found in the response")
		return nil, nil
	}

	p.logger.Debug("Function calls to handle: %v", functionCalls)
	data, err := json.Marshal(functionCalls)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal function calls: %w", err)
	}
	return data, nil
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
func (p *AnthropicProvider) PrepareStreamRequest(prompt string, options map[string]any) ([]byte, error) {
	requestBody := map[string]any{
		"model":   p.model,
		keyStream: true,
		"messages": []map[string]any{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"max_tokens": AnthropicDefaultMaxTokens, // Default max tokens
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
		if k != keyStream { // Don't override stream setting
			requestBody[k] = v
		}
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseStreamResponse processes a single SSE JSON "data:" payload from Anthropic Messages streaming.
// It returns either a text Content token, a Usage-only token, io.EOF for message_stop, or "skip token".
func (p *AnthropicProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return nil, errors.New("empty chunk")
	}
	// [DONE] guard (if your decoder ever passes this through)
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return nil, io.EOF
	}

	var ev anthropicEvent
	if err := json.Unmarshal(chunk, &ev); err != nil {
		return nil, fmt.Errorf("malformed event: %w", err)
	}

	switch ev.Type {
	case "content_block_delta":
		// Only emit text deltas as tokens
		if ev.Delta != nil && ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
			return &Response{
				Content: Text{Value: ev.Delta.Text},
			}, nil
		}
		return nil, errors.New("skip token")

	case "message_start":
		// Usage may be present on the embedded message
		if ev.Message != nil && ev.Message.Usage != nil {
			return &Response{
				Usage: NewUsage(
					ev.Message.Usage.InputTokens,
					ev.Message.Usage.CacheCreationInputTokens,
					ev.Message.Usage.OutputTokens,
					ev.Message.Usage.CacheReadInputTokens,
				),
			}, nil
		}
		return nil, errors.New("skip token")

	case "message_delta":
		// Usage may be present at the top level; counts are cumulative
		if ev.Usage != nil {
			return &Response{
				Usage: NewUsage(
					ev.Usage.InputTokens,
					ev.Usage.CacheCreationInputTokens,
					ev.Usage.OutputTokens,
					ev.Usage.CacheReadInputTokens,
				),
			}, nil
		}
		return nil, errors.New("skip token")

	case "message_stop":
		return nil, io.EOF

	// Ignore pings, starts/stops of blocks, tool JSON partials, thinking/signature, etc.
	default:
		return nil, errors.New("skip token")
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
func (p *AnthropicProvider) PrepareRequestWithMessages(
	messages []types.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	// Handle system prompt
	systemPrompt := p.extractSystemPrompt(options)
	p.addSystemPromptToRequestBody(requestBody, systemPrompt)

	// Process tools if present
	p.handleToolsForMessagesRequest(requestBody, options)

	// Convert and add messages
	p.addMemoryMessagesToRequestBody(requestBody, messages, options)

	// Add other options
	p.addRemainingMessagesOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// handleToolsForMessagesRequest processes tools for message-based requests
func (p *AnthropicProvider) handleToolsForMessagesRequest(requestBody map[string]any, options map[string]any) {
	tools, ok := options[anthropicKeyTools].([]types.Tool)
	if !ok || len(tools) == 0 {
		return
	}
	p.processToolsForMessages(tools, requestBody, options)
}

// addMemoryMessagesToRequestBody converts and adds memory messages to the request
func (p *AnthropicProvider) addMemoryMessagesToRequestBody(
	requestBody map[string]any,
	messages []types.MemoryMessage,
	options map[string]any,
) {
	for _, msg := range messages {
		message := p.convertMemoryMessage(msg, options)
		p.addMessageToRequestBody(requestBody, message)
	}
}

// convertMemoryMessage converts a MemoryMessage to Anthropic format
func (p *AnthropicProvider) convertMemoryMessage(msg types.MemoryMessage, options map[string]any) map[string]any {
	content := []map[string]any{
		{
			"type": "text",
			"text": msg.Content,
		},
	}

	// Add cache_control if specified
	p.addCacheControlToContent(content, msg.CacheControl, options)

	return map[string]any{
		"role":    msg.Role,
		"content": content,
	}
}

// addCacheControlToContent adds cache control to message content
func (p *AnthropicProvider) addCacheControlToContent(
	content []map[string]any,
	cacheControl string,
	options map[string]any,
) {
	if len(content) == 0 {
		return
	}

	if cacheControl != "" {
		content[0]["cache_control"] = map[string]string{"type": cacheControl}
	} else if caching, ok := options["enable_caching"].(bool); ok && caching {
		// Add default caching if enabled globally
		content[0]["cache_control"] = map[string]string{"type": "ephemeral"}
	}
}

// addRemainingMessagesOptions adds non-handled options for message requests
func (p *AnthropicProvider) addRemainingMessagesOptions(requestBody map[string]any, options map[string]any) {
	for k, v := range options {
		if p.isHandledMessagesOption(k) {
			continue
		}
		requestBody[k] = v
	}
}

// isHandledMessagesOption checks if an option is already handled for messages
func (p *AnthropicProvider) isHandledMessagesOption(key string) bool {
	return key == anthropicKeySystemPrompt ||
		key == keyMaxTokens ||
		key == anthropicKeyTools ||
		key == anthropicKeyToolChoice ||
		key == anthropicKeyEnableCaching ||
		key == anthropicKeyStructuredMessages
}

// anthropicResponse represents the structure of a response from the Anthropic API.
type anthropicResponse struct {
	StopSeq    *string            `json:"stop_sequence"`
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Model      string             `json:"model"`
	StopReason string             `json:"stop_reason"`
	Content    []anthropicContent `json:"content"`
	Usage      anthropicUsage     `json:"usage"`
}

// anthropicContent represents a single content block in an Anthropic response.
type anthropicContent struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicEvent struct {
	Index   *int              `json:"index,omitempty"`
	Delta   *anthropicDelta   `json:"delta,omitempty"`
	Usage   *anthropicUsage   `json:"usage,omitempty"`
	Message *anthropicMessage `json:"message,omitempty"`
	Type    string            `json:"type"`
}

type anthropicMessage struct {
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        *anthropicUsage `json:"usage,omitempty"`
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []any           `json:"content"`
}

type anthropicDelta struct {
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
	Type         string  `json:"type,omitempty"`
	Text         string  `json:"text,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
	Thinking     string  `json:"thinking,omitempty"`
	Signature    string  `json:"signature,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int64 `json:"input_tokens,omitempty"`
	OutputTokens             int64 `json:"output_tokens,omitempty"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens,omitempty"`
}
