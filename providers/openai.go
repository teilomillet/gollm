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

const (
	openAIKeyMaxTokens  = "max_tokens"
	openAIKeyToolChoice = "tool_choice"
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
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) *OpenAIProvider {
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
	case openAIKeyMaxTokens:
		if p.needsMaxCompletionTokens() {
			// For models requiring max_completion_tokens, use that instead
			key = "max_completion_tokens"
			// Delete max_tokens if it was previously set
			delete(p.options, openAIKeyMaxTokens)
		} else {
			// For models using max_tokens, make sure max_completion_tokens is not set
			delete(p.options, "max_completion_tokens")
		}
	case "max_completion_tokens":
		// If explicitly setting max_completion_tokens, remove max_tokens to avoid conflicts
		delete(p.options, openAIKeyMaxTokens)
	}

	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *OpenAIProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption(openAIKeyMaxTokens, cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
	p.logger.Debug(
		"Default options set",
		"temperature",
		cfg.Temperature,
		openAIKeyMaxTokens,
		cfg.MaxTokens,
		"seed",
		cfg.Seed,
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
	request := p.initializeOpenAIRequest()

	// Add system and user messages
	p.addSystemPromptAsMessage(request, options)
	p.addUserPromptMessage(request, prompt)

	// Handle tools and tool choice
	p.addToolConfiguration(request, options)

	// Merge and handle options
	mergedOptions := p.mergeProviderAndRequestOptions(options)
	p.handleTokenParameters(mergedOptions)
	p.applyOptionsToRequest(request, mergedOptions)

	data, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	return data, nil
}

// initializeOpenAIRequest creates the base request structure
func (p *OpenAIProvider) initializeOpenAIRequest() map[string]any {
	return map[string]any{
		"model":    p.model,
		"messages": []map[string]any{},
	}
}

// addSystemPromptAsMessage adds system prompt as developer message
func (p *OpenAIProvider) addSystemPromptAsMessage(request map[string]any, options map[string]any) {
	systemPrompt, ok := options["system_prompt"].(string)
	if !ok || systemPrompt == "" {
		return
	}

	if messages, ok := request["messages"].([]map[string]any); ok {
		request["messages"] = append(messages, map[string]any{
			"role":    "developer",
			"content": systemPrompt,
		})
	}
}

// addUserPromptMessage adds the user's prompt as a message
func (p *OpenAIProvider) addUserPromptMessage(request map[string]any, prompt string) {
	if messages, ok := request["messages"].([]map[string]any); ok {
		request["messages"] = append(messages, map[string]any{
			"role":    "user",
			"content": prompt,
		})
	}
}

// addToolConfiguration adds tools and tool choice to request
func (p *OpenAIProvider) addToolConfiguration(request map[string]any, options map[string]any) {
	// Handle tool_choice
	if toolChoice, ok := options[openAIKeyToolChoice].(string); ok {
		request[openAIKeyToolChoice] = toolChoice
	}

	// Handle tools
	tools, ok := options[KeyTools].([]types.Tool)
	if !ok || len(tools) == 0 {
		return
	}

	openAITools := p.convertToolsToOpenAIFormat(tools)
	request[KeyTools] = openAITools
}

// convertToolsToOpenAIFormat converts tools to OpenAI format
func (p *OpenAIProvider) convertToolsToOpenAIFormat(tools []types.Tool) []map[string]any {
	openAITools := make([]map[string]any, len(tools))
	for i, tool := range tools {
		openAITools[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
				"parameters":  tool.Function.Parameters,
			},
			"strict": true,
		}
	}
	return openAITools
}

// mergeProviderAndRequestOptions merges provider and request options
func (p *OpenAIProvider) mergeProviderAndRequestOptions(options map[string]any) map[string]any {
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if !p.isHandledOpenAIOption(k) {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if !p.isHandledOpenAIOption(k) {
			mergedOptions[k] = v
		}
	}

	return mergedOptions
}

// isHandledOpenAIOption checks if an option is already handled
func (p *OpenAIProvider) isHandledOpenAIOption(key string) bool {
	return key == KeyTools || key == openAIKeyToolChoice || key == KeySystemPrompt
}

// applyOptionsToRequest adds merged options to the request
func (p *OpenAIProvider) applyOptionsToRequest(request map[string]any, mergedOptions map[string]any) {
	for k, v := range mergedOptions {
		request[k] = v
	}
}

// parseSchema converts various schema formats to a map structure
func (p *OpenAIProvider) parseSchema(schema any) (any, error) {
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
	return schemaObj, nil
}

// buildSchemaRequest builds a request with JSON schema response format
func (p *OpenAIProvider) buildSchemaRequest(prompt string, cleanSchema any, options map[string]any) map[string]any {
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
		if messages, ok := request["messages"].([]map[string]any); ok {
			request["messages"] = append([]map[string]any{
				{"role": "system", "content": systemPrompt},
			}, messages...)
		}
	}

	return request
}

// mergeOptionsForSchema merges provider options with request options for schema requests
func (p *OpenAIProvider) mergeOptionsForSchema(options map[string]any) map[string]any {
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

	return mergedOptions
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

	// Parse and clean the schema
	schemaObj, err := p.parseSchema(schema)
	if err != nil {
		return nil, err
	}

	// Clean the schema for OpenAI by removing unsupported validation rules
	cleanSchema := cleanSchemaForOpenAI(schemaObj)

	// Debug log the cleaned schema
	if cleanSchemaJSON, err := json.MarshalIndent(cleanSchema, "", "  "); err == nil {
		p.logger.Debug("Cleaned schema for OpenAI", "schema", string(cleanSchemaJSON))
	}

	// Build the request with schema
	request := p.buildSchemaRequest(prompt, cleanSchema, options)

	// Merge and process options
	mergedOptions := p.mergeOptionsForSchema(options)
	p.handleTokenParameters(mergedOptions)

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request with schema", "error", err)
		return nil, fmt.Errorf("failed to marshal request with schema: %w", err)
	}

	p.logger.Debug("Full request to OpenAI", "request", string(reqJSON))
	return reqJSON, nil
}

// cleanSchemaForOpenAI removes validation rules that OpenAI doesn't support
func cleanSchemaForOpenAI(schema any) any {
	schemaMap, ok := schema.(map[string]any)
	if !ok {
		return schema
	}

	result := cleanSchemaMap(schemaMap)

	// Add additionalProperties: false at each object level
	if schemaMap["type"] == "object" {
		result["additionalProperties"] = false
	}

	return result
}

// cleanSchemaMap processes a schema map and cleans its properties
func cleanSchemaMap(schemaMap map[string]any) map[string]any {
	result := make(map[string]any)

	for k, v := range schemaMap {
		if !isAllowedSchemaKey(k) {
			continue
		}

		result[k] = processSchemaValue(k, v)
	}

	return result
}

// isAllowedSchemaKey checks if a schema key is allowed for OpenAI
func isAllowedSchemaKey(key string) bool {
	switch key {
	case "type", "properties", "required", "items":
		return true
	default:
		return false
	}
}

// processSchemaValue processes a schema value based on its key
func processSchemaValue(key string, value any) any {
	switch key {
	case "properties":
		return cleanProperties(value)
	case "items":
		return cleanSchemaForOpenAI(value)
	default:
		return value
	}
}

// cleanProperties cleans a properties map
func cleanProperties(value any) map[string]any {
	props := make(map[string]any)
	propsMap, ok := value.(map[string]any)
	if !ok {
		return props
	}

	for propName, propSchema := range propsMap {
		props[propName] = cleanSchemaForOpenAI(propSchema)
	}

	return props
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
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
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
	data, err := json.Marshal(functionCalls)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal function calls: %w", err)
	}
	return data, nil
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
		if _, hasMaxTokens := mergedOptions[openAIKeyMaxTokens]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions[openAIKeyMaxTokens]
			delete(mergedOptions, openAIKeyMaxTokens)
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions[openAIKeyMaxTokens] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		requestBody[k] = v
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
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

// addSystemPromptToRequest adds the system prompt to the request if present
func (p *OpenAIProvider) addSystemPromptToRequest(request map[string]any, options map[string]any) {
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		if messagesArray, ok := request["messages"].([]map[string]any); ok {
			request["messages"] = append(messagesArray, map[string]any{
				"role":    "system",
				"content": systemPrompt,
			})
		}
	}
}

// addMemoryMessagesToRequest converts MemoryMessage objects to OpenAI format and adds them to request
func (p *OpenAIProvider) addMemoryMessagesToRequest(request map[string]any, messages []types.MemoryMessage) {
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

		if messagesArray, ok := request["messages"].([]map[string]any); ok {
			request["messages"] = append(messagesArray, message)
		}
	}
}

// addToolsToRequest handles tool configuration for the request
func (p *OpenAIProvider) addToolsToRequest(request map[string]any, options map[string]any) {
	// Handle tool_choice
	if toolChoice, ok := options[openAIKeyToolChoice].(string); ok {
		request[openAIKeyToolChoice] = toolChoice
	}

	// Handle tools
	if tools, ok := options[KeyTools].([]types.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]any, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true,
			}
		}
		request[KeyTools] = openAITools
	}
}

// handleTokenParameters handles max_tokens/max_completion_tokens conflict
func (p *OpenAIProvider) handleTokenParameters(mergedOptions map[string]any) {
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions[openAIKeyMaxTokens]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions[openAIKeyMaxTokens]
			delete(mergedOptions, openAIKeyMaxTokens)
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions[openAIKeyMaxTokens] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}
}

func (p *OpenAIProvider) PrepareRequestWithMessages(
	messages []types.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	request := map[string]any{
		"model":    p.model,
		"messages": []map[string]any{},
	}

	// Build messages array
	p.addSystemPromptToRequest(request, options)
	p.addMemoryMessagesToRequest(request, messages)

	// Handle tools and tool choice
	p.addToolsToRequest(request, options)

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]any)

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != KeyTools && k != openAIKeyToolChoice && k != KeySystemPrompt && k != KeyStructuredMessages {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != KeyTools && k != openAIKeyToolChoice && k != KeySystemPrompt && k != KeyStructuredMessages {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions[openAIKeyMaxTokens]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions[openAIKeyMaxTokens]
			delete(mergedOptions, openAIKeyMaxTokens)
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions[openAIKeyMaxTokens] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	data, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	return data, nil
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
