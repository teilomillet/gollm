// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/weave-labs/gollm/internal/models"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Common parameter keys
const (
	deepSeekKeyMaxTokens          = "max_tokens"
	deepSeekKeyStream             = "stream"
	deepSeekKeyModel              = "model"
	deepSeekKeySystemPrompt       = "system_prompt"
	deepSeekKeyTools              = "tools"
	deepSeekKeyToolChoice         = "tool_choice"
	deepSeekKeyStructuredMessages = "structured_messages"
	deepSeekKeyMessages           = "messages"
	deepSeekKeyResponseFormat     = "response_format"
	deepSeekKeyTemperature        = "temperature"
	deepSeekKeyTopP               = "top_p"
	deepSeekKeySeed               = "seed"
	deepSeekKeyStop               = "stop"
)

// DeepSeekProvider implements the Provider interface for DeepSeek's API.
// It supports DeepSeek models and provides access to their capabilities,
// including chat completion and structured output using OpenAI-compatible API.
type DeepSeekProvider struct {
	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewDeepSeekProvider creates a new DeepSeek provider instance.
// It initializes the provider with the given API key, model, and optional headers.
func NewDeepSeekProvider(apiKey, model string, extraHeaders map[string]string) *DeepSeekProvider {
	provider := &DeepSeekProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}

	for k, v := range extraHeaders {
		provider.extraHeaders[k] = v
	}

	return provider
}

// SetLogger configures the logger for the DeepSeek provider.
// This is used for debugging and monitoring API interactions.
func (p *DeepSeekProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the DeepSeek provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - seed: Random seed for deterministic output
//   - stop: Stop sequences
func (p *DeepSeekProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *DeepSeekProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption(deepSeekKeyTemperature, cfg.Temperature)
	p.SetOption(deepSeekKeyMaxTokens, cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption(deepSeekKeySeed, *cfg.Seed)
	}
}

// Name returns "deepseek" as the provider identifier.
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// Endpoint returns the DeepSeek API endpoint URL.
// This is "https://api.deepseek.com/chat/completions".
func (p *DeepSeekProvider) Endpoint() string {
	return "https://api.deepseek.com/chat/completions"
}

// Headers returns the required HTTP headers for DeepSeek API requests.
// This includes:
//   - Content-Type: application/json
//   - Authorization: Bearer token using the API key
//   - Any additional headers specified via SetExtraHeaders
func (p *DeepSeekProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *DeepSeekProvider) SetExtraHeaders(extraHeaders map[string]string) {
	for k, v := range extraHeaders {
		p.extraHeaders[k] = v
	}
}

// PrepareRequest prepares a request payload for the DeepSeek API using the unified Request structure.
// It handles system prompts, messages, structured responses, and tool configurations.
func (p *DeepSeekProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	// Add messages
	p.addMessagesToRequestBody(requestBody, req.Messages, options)

	// Handle system prompt
	systemPrompt := p.extractSystemPromptFromRequest(req, options)
	if systemPrompt != "" {
		p.addSystemPromptToRequestBody(requestBody, systemPrompt)
	}

	// Add structured response support if schema is provided
	if req.ResponseSchema != nil {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	// Handle tools
	requestBody = p.handleToolsForRequest(requestBody, options)

	// Add remaining options
	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// PrepareStreamRequest prepares a streaming request payload for the DeepSeek API.
// It's similar to PrepareRequest but enables streaming mode.
func (p *DeepSeekProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	// Create a copy of options and enable streaming
	streamOptions := make(map[string]any)
	for k, v := range options {
		streamOptions[k] = v
	}
	streamOptions[deepSeekKeyStream] = true

	return p.PrepareRequest(req, streamOptions)
}

// ParseResponse parses the JSON response from the DeepSeek API into a Response struct.
// It extracts the content, role, and usage information from the API response.
func (p *DeepSeekProvider) ParseResponse(body []byte) (*Response, error) {
	var deepSeekResp deepSeekResponse
	if err := json.Unmarshal(body, &deepSeekResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal DeepSeek response: %w", err)
	}

	if len(deepSeekResp.Choices) == 0 {
		return nil, errors.New("no choices in DeepSeek response")
	}

	choice := deepSeekResp.Choices[0]
	content := Text{Value: choice.Message.Content}

	response := &Response{
		Role:    choice.Message.Role,
		Content: content,
	}

	// Add usage information if available
	if deepSeekResp.Usage != nil {
		response.Usage = &Usage{
			InputTokens:  int64(deepSeekResp.Usage.PromptTokens),
			OutputTokens: int64(deepSeekResp.Usage.CompletionTokens),
			TotalTokens:  int64(deepSeekResp.Usage.TotalTokens),
		}
	}

	// Handle tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		response.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
		for i, tc := range choice.Message.ToolCalls {
			response.ToolCalls[i] = ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}{
					Name:      tc.Function.Name,
					Arguments: json.RawMessage(tc.Function.Arguments),
				},
			}
		}
	}

	return response, nil
}

// ParseStreamResponse parses streaming response chunks from the DeepSeek API.
// It handles the server-sent events format and extracts delta content.
func (p *DeepSeekProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Handle server-sent events format
	lines := bytes.Split(chunk, []byte("\n"))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 || !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Remove "data: " prefix
		jsonData := bytes.TrimPrefix(line, []byte("data: "))

		// Check for end of stream
		if bytes.Equal(jsonData, []byte("[DONE]")) {
			continue
		}

		var streamResp deepSeekStreamResponse
		if err := json.Unmarshal(jsonData, &streamResp); err != nil {
			p.logger.Debug("Failed to parse streaming chunk", "error", err, "data", string(jsonData))
			continue
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		choice := streamResp.Choices[0]
		if choice.Delta.Content != "" {
			return &Response{
				Role:    "assistant",
				Content: Text{Value: choice.Delta.Content},
			}, nil
		}
	}

	return nil, io.EOF
}

// initializeRequestBody creates the base request body with model information.
func (p *DeepSeekProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		deepSeekKeyModel: p.model,
	}
}

// extractSystemPromptFromRequest extracts the system prompt from the request or options.
func (p *DeepSeekProvider) extractSystemPromptFromRequest(req *Request, options map[string]any) string {
	if req.SystemPrompt != "" {
		return req.SystemPrompt
	}
	if systemPrompt, exists := options[deepSeekKeySystemPrompt]; exists {
		if sp, ok := systemPrompt.(string); ok {
			return sp
		}
	}
	return ""
}

// handleToolsForRequest processes tools configuration for the request.
func (p *DeepSeekProvider) handleToolsForRequest(
	requestBody map[string]any,
	options map[string]any,
) map[string]any {
	if tools, exists := options[deepSeekKeyTools]; exists {
		if toolsSlice, ok := tools.([]models.Tool); ok && len(toolsSlice) > 0 {
			return p.processTools(toolsSlice, requestBody, options)
		}
	}
	return requestBody
}

// addSystemPromptToRequestBody adds the system prompt to the request messages.
func (p *DeepSeekProvider) addSystemPromptToRequestBody(requestBody map[string]any, systemPrompt string) {
	messages, exists := requestBody[deepSeekKeyMessages]
	if !exists {
		messages = []map[string]any{}
	}

	messagesList, ok := messages.([]map[string]any)
	if !ok {
		messagesList = []map[string]any{}
	}
	systemMessage := map[string]any{
		"role":    "system",
		"content": systemPrompt,
	}

	// Prepend system message
	messagesList = append([]map[string]any{systemMessage}, messagesList...)
	requestBody[deepSeekKeyMessages] = messagesList
}

// addStructuredResponseToRequest adds JSON schema for structured responses.
func (p *DeepSeekProvider) addStructuredResponseToRequest(requestBody map[string]any, schema any) {
	// DeepSeek supports OpenAI-compatible structured responses
	responseFormat := map[string]any{
		"type": "json_object",
	}

	// If schema is provided, add it to the response format
	if schema != nil {
		responseFormat["json_schema"] = schema
	}

	requestBody[deepSeekKeyResponseFormat] = responseFormat
}

// addMessagesToRequestBody converts and adds messages to the request body.
func (p *DeepSeekProvider) addMessagesToRequestBody(
	requestBody map[string]any,
	messages []Message,
	options map[string]any,
) {
	if len(messages) == 0 {
		return
	}

	convertedMessages := make([]map[string]any, 0, len(messages))
	for i := range messages {
		convertedMessage := p.convertMessageToDeepSeekFormat(&messages[i], options)
		convertedMessages = append(convertedMessages, convertedMessage)
	}

	requestBody[deepSeekKeyMessages] = convertedMessages
}

// convertMessageToDeepSeekFormat converts a unified Message to DeepSeek API format.
func (p *DeepSeekProvider) convertMessageToDeepSeekFormat(msg *Message, _ map[string]any) map[string]any {
	converted := map[string]any{
		"role":    msg.Role,
		"content": msg.Content,
	}

	// Add optional fields if present
	if msg.Name != "" {
		converted["name"] = msg.Name
	}

	if msg.ToolCallID != "" {
		converted["tool_call_id"] = msg.ToolCallID
	}

	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]map[string]any, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			toolCalls[i] = map[string]any{
				"id":   tc.ID,
				"type": tc.Type,
				"function": map[string]any{
					"name":      tc.Function.Name,
					"arguments": string(tc.Function.Arguments),
				},
			}
		}
		converted["tool_calls"] = toolCalls
	}

	return converted
}

// addRemainingOptions adds provider-specific options to the request body.
func (p *DeepSeekProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	// Merge provider options with request options, giving priority to request options
	mergedOptions := make(map[string]any)
	for k, v := range p.options {
		mergedOptions[k] = v
	}
	for k, v := range options {
		if !p.isGlobalOption(k) {
			mergedOptions[k] = v
		}
	}

	// Add DeepSeek-specific options
	for key, value := range mergedOptions {
		if p.isValidDeepSeekOption(key) {
			requestBody[key] = value
		}
	}
}

// isGlobalOption checks if a key is a global option that shouldn't be passed to the API.
func (p *DeepSeekProvider) isGlobalOption(key string) bool {
	globalKeys := []string{
		deepSeekKeySystemPrompt,
		deepSeekKeyTools,
		deepSeekKeyStructuredMessages,
		deepSeekKeyToolChoice,
	}
	for _, globalKey := range globalKeys {
		if key == globalKey {
			return true
		}
	}
	return false
}

// isValidDeepSeekOption checks if a key is a valid DeepSeek API option.
func (p *DeepSeekProvider) isValidDeepSeekOption(key string) bool {
	validKeys := []string{
		deepSeekKeyTemperature,
		deepSeekKeyMaxTokens,
		deepSeekKeyTopP,
		deepSeekKeySeed,
		deepSeekKeyStop,
		deepSeekKeyStream,
		deepSeekKeyResponseFormat,
	}
	for _, validKey := range validKeys {
		if key == validKey {
			return true
		}
	}
	return false
}

// processTools processes tool configurations for the request.
func (p *DeepSeekProvider) processTools(
	tools []models.Tool,
	requestBody map[string]any,
	options map[string]any,
) map[string]any {
	if len(tools) == 0 {
		return requestBody
	}

	// Convert tools to DeepSeek/OpenAI format
	convertedTools := make([]map[string]any, len(tools))
	for i, tool := range tools {
		convertedTools[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
				"parameters":  tool.Function.Parameters,
			},
		}
	}

	requestBody[deepSeekKeyTools] = convertedTools

	// Handle tool choice if specified
	if toolChoice, exists := options[deepSeekKeyToolChoice]; exists {
		requestBody[deepSeekKeyToolChoice] = toolChoice
	}

	return requestBody
}

// DeepSeek API response structures
type deepSeekResponse struct {
	Usage   *deepSeekUsage   `json:"usage,omitempty"`
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Model   string           `json:"model"`
	Choices []deepSeekChoice `json:"choices"`
	Created int64            `json:"created"`
}

type deepSeekChoice struct {
	FinishReason string          `json:"finish_reason"`
	Message      deepSeekMessage `json:"message"`
	Index        int             `json:"index"`
}

type deepSeekMessage struct {
	Role      string             `json:"role"`
	Content   string             `json:"content"`
	ToolCalls []deepSeekToolCall `json:"tool_calls,omitempty"`
}

type deepSeekToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function deepSeekFunction `json:"function"`
}

type deepSeekFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type deepSeekUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type deepSeekStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Model   string                 `json:"model"`
	Choices []deepSeekStreamChoice `json:"choices"`
	Created int64                  `json:"created"`
}

type deepSeekStreamChoice struct {
	FinishReason string              `json:"finish_reason"`
	Delta        deepSeekStreamDelta `json:"delta"`
	Index        int                 `json:"index"`
}

type deepSeekStreamDelta struct {
	Role      string             `json:"role,omitempty"`
	Content   string             `json:"content,omitempty"`
	ToolCalls []deepSeekToolCall `json:"tool_calls,omitempty"`
}

// SupportsStreaming returns false as DeepSeek models do not support streaming.
func (p *DeepSeekProvider) SupportsStreaming() bool {
	return false
}

// SupportsStructuredResponse returns false as DeepSeek models do not support structured output.
func (p *DeepSeekProvider) SupportsStructuredResponse() bool {
	return false
}

// SupportsFunctionCalling returns true as DeepSeek models support function calling.
func (p *DeepSeekProvider) SupportsFunctionCalling() bool {
	return true
}
