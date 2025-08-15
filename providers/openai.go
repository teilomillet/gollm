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

// OpenAIProvider implements the Provider interface for OpenAI's API.
// It supports GPT models and provides access to OpenAI's language model capabilities,
// including function calling, JSON mode, and structured output validation.
type OpenAIProvider struct {
	logger       utils.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewOpenAIProvider creates a new OpenAI provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: OpenAI API key for authentication
//   - model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured OpenAI Provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenAI provider.
// This is used for debugging and monitoring API interactions.
func (p *OpenAIProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// needsMaxCompletionTokens checks if the model requires max_completion_tokens instead of max_tokens
func (p *OpenAIProvider) needsMaxCompletionTokens() bool {
	// Check for models that start with "o"
	if strings.HasPrefix(p.model, "o") {
		return true
	}

	// Check for gpt-4o and similar models
	if strings.Contains(p.model, "4o") || strings.Contains(p.model, "-o") {
		return true
	}

	return false
}

// SetOption sets a specific option for the OpenAI provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response (automatically converted to max_completion_tokens for "o" models)
//   - top_p: Nucleus sampling parameter
//   - frequency_penalty: Repetition reduction
//   - presence_penalty: Topic steering
//   - seed: Deterministic sampling seed
func (p *OpenAIProvider) SetOption(key string, value any) {
	// Handle max_tokens conversion for "o" models
	switch key {
	case "max_tokens":
		if p.needsMaxCompletionTokens() {
			// For models requiring max_completion_tokens, use that instead
			key = "max_completion_tokens"
			// Delete max_tokens if it was previously set
			delete(p.options, "max_tokens")
		} else {
			// For models using max_tokens, make sure max_completion_tokens is not set
			delete(p.options, "max_completion_tokens")
		}
	case "max_completion_tokens":
		// If explicitly setting max_completion_tokens, remove max_tokens to avoid conflicts
		delete(p.options, "max_tokens")
	}

	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *OpenAIProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug(
		"Default options set",
		"temperature",
		config.Temperature,
		"max_tokens",
		config.MaxTokens,
		"seed",
		config.Seed,
	)
}

// Name returns "openai" as the provider identifier.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Endpoint returns the OpenAI API endpoint URL.
// For API version 1, this is "https://api.openai.com/v1/chat/completions".
func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

// SupportsJSONSchema indicates that OpenAI supports native JSON schema validation
// through its function calling and JSON mode capabilities.
func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for OpenAI API requests.
// This includes:
//   - Authorization: Bearer token using the API key
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for an OpenAI API call.
// It handles:
//   - Message formatting
//   - System messages
//   - Function/tool definitions
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
	request := map[string]any{
		"model":    p.model,
		"messages": []map[string]any{},
	}

	// Handle system prompt as developer message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]any), map[string]any{
			"role":    "developer",
			"content": systemPrompt,
		})
	}

	// Add user message
	request["messages"] = append(request["messages"].([]map[string]any), map[string]any{
		"role":    "user",
		"content": prompt,
	})

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		request["tool_choice"] = toolChoice
	}

	// Handle tools
	if tools, ok := options["tools"].([]types.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]any, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true, // Add this if you want strict mode
			}
		}
		request["tools"] = openAITools
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
// This uses OpenAI's function calling feature to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithSchema(prompt string, options map[string]any, schema any) ([]byte, error) {
	p.logger.Debug("Preparing request with schema", "prompt", prompt, "schema", schema)

	// First, ensure we have a proper object for the schema
	var schemaObj any
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema bytes: %w", err)
		}
	case map[string]any:
		schemaObj = s
	default:
		// Try to marshal and unmarshal to ensure we have a proper object
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
		}
	}

	// Clean the schema for OpenAI by removing unsupported validation rules
	cleanSchema := cleanSchemaForOpenAI(schemaObj)

	// Debug log the cleaned schema
	cleanSchemaJSON, _ := json.MarshalIndent(cleanSchema, "", "  ")
	p.logger.Debug("Cleaned schema for OpenAI", "schema", string(cleanSchemaJSON))

	request := map[string]any{
		"model": p.model,
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "structured_response",
				"schema": cleanSchema,
				"strict": true,
			},
		},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append([]map[string]any{
			{"role": "system", "content": systemPrompt},
		}, request["messages"].([]map[string]any)...)
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request with schema", "error", err)
		return nil, err
	}

	p.logger.Debug("Full request to OpenAI", "request", string(reqJSON))
	return reqJSON, nil
}

// cleanSchemaForOpenAI removes validation rules that OpenAI doesn't support
func cleanSchemaForOpenAI(schema any) any {
	if schemaMap, ok := schema.(map[string]any); ok {
		result := make(map[string]any)
		for k, v := range schemaMap {
			switch k {
			case "type", "properties", "required", "items":
				switch k {
				case "properties":
					props := make(map[string]any)
					if propsMap, ok := v.(map[string]any); ok {
						for propName, propSchema := range propsMap {
							props[propName] = cleanSchemaForOpenAI(propSchema)
						}
					}
					result[k] = props
				case "items":
					result[k] = cleanSchemaForOpenAI(v)
				default:
					result[k] = v
				}
			}
		}
		// Add additionalProperties: false at each object level
		if schemaMap["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	}
	return schema
}

// ParseResponse extracts the generated text from the OpenAI API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *OpenAIProvider) ParseResponse(body []byte) (*Response, error) {
	response := openAIResponse{}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New("empty response from API")
	}

	usage := &Usage{}

	if response.Usage != nil && response.Usage.PromptTokensDetails != nil {
		usage = NewUsage(
			response.Usage.PromptTokens,
			response.Usage.PromptTokensDetails.CacheTokens,
			response.Usage.CompletionTokens,
			0,
		)
	}

	message := response.Choices[0].Message
	if message.Content != "" {
		return &Response{
			Content: Text{message.Content},
			Usage:   usage}, nil
	}

	if len(message.ToolCalls) > 0 {
		var functionCalls []string
		for _, call := range message.ToolCalls {
			// Parse arguments as raw JSON to preserve the exact format
			var args any
			if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
				return nil, fmt.Errorf("error parsing function arguments: %w", err)
			}

			functionCall, err := FormatFunctionCall(call.Function.Name, args)
			if err != nil {
				return nil, fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
		}

		return &Response{
			Content: Text{strings.Join(functionCalls, "\n")},
			Usage:   usage}, nil
	}

	return nil, errors.New("no content or tool calls in response")
}

// HandleFunctionCalls processes function calling in the response.
// This supports OpenAI's function calling and JSON mode features.
func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		return nil, errors.New("no function calls found in response")
	}

	p.logger.Debug("Function calls to handle", "calls", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming indicates whether streaming is supported
func (p *OpenAIProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *OpenAIProvider) PrepareStreamRequest(prompt string, options map[string]any) ([]byte, error) {
	// Start with regular request preparation
	requestBody := map[string]any{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
		"stream_options": []map[string]bool{
			{"include_usage": true},
		},
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *OpenAIProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return nil, errors.New("empty chunk")
	}

	// Check for [DONE] marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return nil, io.EOF
	}

	// Parse the chunk
	response := openAIStreamResponse{}

	if err := json.Unmarshal(chunk, &response); err != nil {
		return nil, fmt.Errorf("malformed response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, errors.New("no choices in response")
	}

	// Handle finish reason
	if response.Choices[0].FinishReason != "" {
		return nil, io.EOF
	}

	// Skip role-only messages
	if response.Choices[0].Delta.Role != "" && response.Choices[0].Delta.Content == "" {
		return nil, errors.New("skip token")
	}

	usage := &Usage{}

	if response.Usage != nil && response.Usage.PromptTokensDetails != nil {
		usage = NewUsage(
			response.Usage.PromptTokens,
			response.Usage.PromptTokensDetails.CacheTokens,
			response.Usage.CompletionTokens,
			0,
		)
	}

	return &Response{
		Content: Text{
			response.Choices[0].Delta.Content,
		},
		Usage: usage,
	}, nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching and
// better preserves conversation structure for the OpenAI API.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithMessages(
	messages []types.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	request := map[string]any{
		"model":    p.model,
		"messages": []map[string]any{},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]any), map[string]any{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Convert MemoryMessage objects to OpenAI messages format
	for _, msg := range messages {
		message := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		// Add metadata if present
		if len(msg.Metadata) > 0 {
			for k, v := range msg.Metadata {
				message[k] = v
			}
		}

		request["messages"] = append(request["messages"].([]map[string]any), message)
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		request["tool_choice"] = toolChoice
	}

	// Handle tools
	if tools, ok := options["tools"].([]types.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]any, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true, // Add this if you want strict mode
			}
		}
		request["tools"] = openAITools
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "structured_messages" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "structured_messages" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}

type openAIResponse struct {
	Usage   *openAIUsage   `json:"usage"`
	Choices []openAIChoice `json:"choices"`
}

type openAIChoice struct {
	Message *openAIMessage `json:"message"`
}

type openAIMessage struct {
	Content   string           `json:"content"`
	ToolCalls []openAIToolCall `json:"tool_calls"`
}

type openAIToolCall struct {
	Function *openAIFunction `json:"function"`
	ID       string          `json:"id"`
	Type     string          `json:"type"`
}

type openAIFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}
type openAIUsage struct {
	PromptTokensDetails     *openAIPromptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *openAICompletionTokensDetails `json:"completion_tokens_details"`
	PromptTokens            int64                          `json:"prompt_tokens"`
	CompletionTokens        int64                          `json:"completion_tokens"`
	TotalTokens             int64                          `json:"total_tokens"`
}

type openAIPromptTokensDetails struct {
	CacheTokens int64 `json:"cache_tokens"`
	AudioTokens int64 `json:"audio_tokens"`
}

type openAICompletionTokensDetails struct {
	ReasoningTokens          int64 `json:"reasoning_tokens"`
	AudioTokens              int64 `json:"audio_tokens"`
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
}

type openAIStreamResponse struct {
	Usage   *openAIUsage         `json:"usage,omitempty"`
	Choices []openAIStreamChoice `json:"choices"`
}

type openAIStreamChoice struct {
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason string            `json:"finish_reason"`
}

type openAIStreamDelta struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
