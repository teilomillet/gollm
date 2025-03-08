// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// OpenRouterProvider implements the Provider interface for OpenRouter API.
// It provides access to multiple LLMs through a single API, with features like
// model routing, fallbacks, prompt caching.
type OpenRouterProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewOpenRouterProvider creates a new OpenRouter provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: OpenRouter API key for authentication
//   - model: The model to use (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured OpenRouter Provider instance
func NewOpenRouterProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenRouterProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenRouter provider.
func (p *OpenRouterProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// Name returns the identifier for this provider ("openrouter").
func (p *OpenRouterProvider) Name() string {
	return "openrouter"
}

// Endpoint returns the OpenRouter API endpoint URL for chat completions.
func (p *OpenRouterProvider) Endpoint() string {
	return "https://openrouter.ai/api/v1/chat/completions"
}

// CompletionsEndpoint returns the OpenRouter API endpoint URL for text completions.
// This is used for the legacy completions interface.
func (p *OpenRouterProvider) CompletionsEndpoint() string {
	return "https://openrouter.ai/api/v1/completions"
}

// GenerationEndpoint returns the OpenRouter API endpoint for retrieving generation details.
// This can be used to query stats like cost and token usage after a request.
func (p *OpenRouterProvider) GenerationEndpoint(generationID string) string {
	return fmt.Sprintf("https://openrouter.ai/api/v1/generation?id=%s", generationID)
}

// SetOption sets a model-specific option for the OpenRouter provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - route: Routing strategy (e.g., "fallback", "lowest-latency")
//   - transforms: Array of transformations to apply to the response
//   - provider: Provider preferences for routing
func (p *OpenRouterProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *OpenRouterProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}

	// OpenRouter-specific defaults
	// Reasoning transforms are enabled via options rather than config
	if _, ok := p.options["enable_reasoning"]; ok {
		p.SetOption("transforms", []string{"reasoning"})
	}
}

// SupportsJSONSchema indicates whether this provider supports JSON schema validation.
// OpenRouter supports JSON schema validation when using supported models.
func (p *OpenRouterProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the HTTP headers required for OpenRouter API requests.
func (p *OpenRouterProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
		"HTTP-Referer":  "https://github.com/teilomillet/gollm", // Identify the app to OpenRouter
	}

	// Add OpenRouter specific headers
	headers["X-Title"] = "GoLLM Integration"

	// Add any extra headers
	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

// PrepareRequest creates a chat completion request for the OpenRouter API.
func (p *OpenRouterProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Start with the passed options
	req := map[string]interface{}{}
	for k, v := range options {
		req[k] = v
	}

	// Add model
	req["model"] = p.model

	// Add options from the provider
	for k, v := range p.options {
		if _, exists := req[k]; !exists {
			req[k] = v
		}
	}

	// Handle fallback models if specified
	if fallbackModels, ok := req["fallback_models"].([]string); ok {
		req["models"] = append([]string{p.model}, fallbackModels...)
		delete(req, "fallback_models")
	} else if autoRoute, ok := req["auto_route"].(bool); ok && autoRoute {
		// Use OpenRouter's auto-routing capability
		req["model"] = "openrouter/auto"
		delete(req, "auto_route")
	}

	// Handle provider routing preferences if provided
	if providerPrefs, ok := req["provider_preferences"].(map[string]interface{}); ok {
		req["provider"] = providerPrefs
		delete(req, "provider_preferences")
	}

	// Create messages array with system and user messages
	messages := []map[string]interface{}{}

	// If there's a system message in the options, use it
	if sysMsg, ok := req["system_message"].(string); ok {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": sysMsg,
		})
		delete(req, "system_message")
	}

	// Add the user prompt
	messages = append(messages, map[string]interface{}{
		"role":    "user",
		"content": prompt,
	})

	req["messages"] = messages

	// Handle tools/function calling if provided
	if tools, ok := req["tools"].([]interface{}); ok && len(tools) > 0 {
		req["tools"] = tools
	}

	if toolChoice, ok := req["tool_choice"]; ok {
		req["tool_choice"] = toolChoice
	}

	// Add streaming if requested
	if stream, ok := req["stream"].(bool); ok && stream {
		req["stream"] = true
	}

	// Handle prompt caching for supported models
	if caching, ok := req["enable_prompt_caching"].(bool); ok && caching {
		// OpenRouter handles caching automatically for supported providers
		delete(req, "enable_prompt_caching")
	}

	return json.Marshal(req)
}

// PrepareCompletionRequest creates a text completion request for the OpenRouter API.
// This uses the legacy completions endpoint rather than chat completions.
func (p *OpenRouterProvider) PrepareCompletionRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Start with the passed options
	req := map[string]interface{}{}
	for k, v := range options {
		req[k] = v
	}

	// Add model
	req["model"] = p.model

	// Add options from the provider
	for k, v := range p.options {
		if _, exists := req[k]; !exists {
			req[k] = v
		}
	}

	// Handle fallback models if specified
	if fallbackModels, ok := req["fallback_models"].([]string); ok {
		req["models"] = append([]string{p.model}, fallbackModels...)
		delete(req, "fallback_models")
	} else if autoRoute, ok := req["auto_route"].(bool); ok && autoRoute {
		// Use OpenRouter's auto-routing capability
		req["model"] = "openrouter/auto"
		delete(req, "auto_route")
	}

	// Add the prompt
	req["prompt"] = prompt

	// Handle provider routing preferences if provided
	if providerPrefs, ok := req["provider_preferences"].(map[string]interface{}); ok {
		req["provider"] = providerPrefs
		delete(req, "provider_preferences")
	}

	// Add streaming if requested
	if stream, ok := req["stream"].(bool); ok && stream {
		req["stream"] = true
	}

	return json.Marshal(req)
}

// PrepareRequestWithSchema creates a request with JSON schema validation.
func (p *OpenRouterProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	// Start with standard request preparation
	req := map[string]interface{}{}
	for k, v := range options {
		req[k] = v
	}

	// Add model
	req["model"] = p.model

	// Add default options from the provider
	for k, v := range p.options {
		if _, exists := req[k]; !exists {
			req[k] = v
		}
	}

	// Handle fallback models if specified
	if fallbackModels, ok := req["fallback_models"].([]string); ok {
		req["models"] = append([]string{p.model}, fallbackModels...)
		delete(req, "fallback_models")
	} else if autoRoute, ok := req["auto_route"].(bool); ok && autoRoute {
		// Use OpenRouter's auto-routing capability
		req["model"] = "openrouter/auto"
		delete(req, "auto_route")
	}

	// Handle provider routing preferences if provided
	if providerPrefs, ok := req["provider_preferences"].(map[string]interface{}); ok {
		req["provider"] = providerPrefs
		delete(req, "provider_preferences")
	}

	// Create messages array with system and user messages
	messages := []map[string]interface{}{}

	// If there's a system message in the options, use it
	if sysMsg, ok := req["system_message"].(string); ok {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": sysMsg,
		})
		delete(req, "system_message")
	}

	// Add the user prompt
	messages = append(messages, map[string]interface{}{
		"role":    "user",
		"content": prompt,
	})

	req["messages"] = messages

	// Add JSON schema to the response format
	req["response_format"] = map[string]interface{}{
		"type":   "json_object",
		"schema": schema,
	}

	// Handle tools/function calling if provided
	if tools, ok := req["tools"].([]interface{}); ok && len(tools) > 0 {
		req["tools"] = tools
	}

	if toolChoice, ok := req["tool_choice"]; ok {
		req["tool_choice"] = toolChoice
	}

	// Handle prompt caching for supported models
	if caching, ok := req["enable_prompt_caching"].(bool); ok && caching {
		// OpenRouter handles caching automatically for supported providers
		delete(req, "enable_prompt_caching")
	}

	return json.Marshal(req)
}

// ParseResponse extracts the completion text from the OpenRouter API response.
func (p *OpenRouterProvider) ParseResponse(body []byte) (string, error) {
	// First try to parse as chat completion to see if it's a chat/completions response
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason       string `json:"finish_reason"`
			NativeFinishReason string `json:"native_finish_reason"`
		} `json:"choices"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		ID    string `json:"id"`
		Model string `json:"model"`
	}

	// Try to parse as a chat completion
	chatErr := json.Unmarshal(body, &chatResp)

	// Check if we have valid chat completion choices
	if chatErr == nil && len(chatResp.Choices) > 0 && chatResp.Choices[0].Message.Content != "" {
		// This is a chat completion response

		// Check for errors
		if chatResp.Error.Message != "" {
			return "", fmt.Errorf("OpenRouter API error: %s", chatResp.Error.Message)
		}

		// Store the generation ID and used model in the logger for potential later use
		if chatResp.ID != "" {
			p.logger.Debug("Generation ID", "id", chatResp.ID)
		}
		if chatResp.Model != "" && chatResp.Model != p.model {
			p.logger.Info("Model used", "requested", p.model, "actual", chatResp.Model)
		}

		return chatResp.Choices[0].Message.Content, nil
	}

	// If it wasn't a valid chat completion, try parsing as a text completion
	var textResp struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		ID    string `json:"id"`
		Model string `json:"model"`
	}

	if err := json.Unmarshal(body, &textResp); err != nil {
		// If we can't parse as text completion either, return the original chat parsing error
		return "", fmt.Errorf("error parsing OpenRouter response: %w", chatErr)
	}

	// Check for errors
	if textResp.Error.Message != "" {
		return "", fmt.Errorf("OpenRouter API error: %s", textResp.Error.Message)
	}

	// Check if we have at least one choice
	if len(textResp.Choices) == 0 {
		return "", fmt.Errorf("no completion choices in OpenRouter response")
	}

	// Store the generation ID and used model in the logger for potential later use
	if textResp.ID != "" {
		p.logger.Debug("Generation ID (text completion)", "id", textResp.ID)
	}
	if textResp.Model != "" && textResp.Model != p.model {
		p.logger.Info("Model used (text completion)", "requested", p.model, "actual", textResp.Model)
	}

	p.logger.Debug("Parsed text completion", "text", textResp.Choices[0].Text)

	// Return text completion
	return textResp.Choices[0].Text, nil
}

// HandleFunctionCalls processes function/tool calling in OpenRouter responses.
func (p *OpenRouterProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	// OpenRouter supports function calling for compatible models
	var resp struct {
		Choices []struct {
			Message struct {
				Content      string           `json:"content"`
				FunctionCall *json.RawMessage `json:"function_call"`
				ToolCalls    []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		ID    string `json:"id"`
		Model string `json:"model"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("error parsing OpenRouter function call response: %w", err)
	}

	// Check if we have a function call or tool calls
	if len(resp.Choices) > 0 {
		message := &resp.Choices[0].Message
		if message.FunctionCall != nil || len(message.ToolCalls) > 0 {
			// Store the generation ID and used model in the logger for potential later use
			if resp.ID != "" {
				p.logger.Debug("Generation ID with tool calls", "id", resp.ID)
			}
			if resp.Model != "" && resp.Model != p.model {
				p.logger.Info("Model used for tool calls", "requested", p.model, "actual", resp.Model)
			}

			// Return the original body since it already contains the function call data
			return body, nil
		}
	}

	return nil, nil
}

// SetExtraHeaders configures additional HTTP headers for OpenRouter API requests.
func (p *OpenRouterProvider) SetExtraHeaders(extraHeaders map[string]string) {
	if extraHeaders == nil {
		p.extraHeaders = make(map[string]string)
		return
	}
	p.extraHeaders = extraHeaders
}

// SupportsStreaming indicates whether this provider supports streaming responses.
func (p *OpenRouterProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a streaming request for the OpenRouter API.
func (p *OpenRouterProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	streamOptions := make(map[string]interface{})
	for k, v := range options {
		streamOptions[k] = v
	}
	streamOptions["stream"] = true
	return p.PrepareRequest(prompt, streamOptions)
}

// ParseStreamResponse processes a chunk from a streaming OpenRouter response.
func (p *OpenRouterProvider) ParseStreamResponse(chunk []byte) (string, error) {
	// Skip empty chunks and "[DONE]" markers
	if len(chunk) == 0 || string(chunk) == "[DONE]" {
		return "", nil
	}

	// Parse the chunk
	var resp struct {
		Choices []struct {
			Delta struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		ID    string `json:"id"`
		Model string `json:"model"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(chunk, &resp); err != nil {
		return "", fmt.Errorf("error parsing OpenRouter stream chunk: %w", err)
	}

	// Check for errors
	if resp.Error.Message != "" {
		return "", fmt.Errorf("OpenRouter API streaming error: %s", resp.Error.Message)
	}

	// Store the generation ID and used model in the logger for potential later use
	if resp.ID != "" {
		p.logger.Debug("Streaming generation ID", "id", resp.ID)
	}
	if resp.Model != "" && resp.Model != p.model {
		p.logger.Info("Model used for streaming", "requested", p.model, "actual", resp.Model)
	}

	// If we have usage information, log it
	if resp.Usage != nil {
		p.logger.Debug("Token usage",
			"prompt_tokens", resp.Usage.PromptTokens,
			"completion_tokens", resp.Usage.CompletionTokens,
			"total_tokens", resp.Usage.TotalTokens)
	}

	// Check if we have at least one choice with content
	if len(resp.Choices) == 0 {
		return "", nil
	}

	// Handle tool calls in streaming mode
	if len(resp.Choices[0].Delta.ToolCalls) > 0 {
		toolCallData, err := json.Marshal(resp.Choices[0].Delta.ToolCalls)
		if err == nil {
			p.logger.Debug("Tool call in streaming mode", "data", string(toolCallData))
		}
	}

	return resp.Choices[0].Delta.Content, nil
}

// PrepareRequestWithMessages creates a request with structured message objects.
func (p *OpenRouterProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	// Start with the passed options
	req := map[string]interface{}{}
	for k, v := range options {
		req[k] = v
	}

	// Add model
	req["model"] = p.model

	// Add options from the provider
	for k, v := range p.options {
		if _, exists := req[k]; !exists {
			req[k] = v
		}
	}

	// Handle fallback models if specified
	if fallbackModels, ok := req["fallback_models"].([]string); ok {
		req["models"] = append([]string{p.model}, fallbackModels...)
		delete(req, "fallback_models")
	} else if autoRoute, ok := req["auto_route"].(bool); ok && autoRoute {
		// Use OpenRouter's auto-routing capability
		req["model"] = "openrouter/auto"
		delete(req, "auto_route")
	}

	// Handle provider routing preferences if provided
	if providerPrefs, ok := req["provider_preferences"].(map[string]interface{}); ok {
		req["provider"] = providerPrefs
		delete(req, "provider_preferences")
	}

	// Convert memory messages to OpenRouter format
	formattedMessages := make([]map[string]interface{}, 0, len(messages))
	for _, msg := range messages {
		formattedMsg := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}

		// Handle Anthropic-style prompt caching if enabled
		if caching, ok := req["enable_prompt_caching"].(bool); ok && caching && msg.Role == "user" {
			// Check if the message is large enough to benefit from caching
			if len(msg.Content) > 1000 {
				// For Anthropic models, we need to use multipart messages with cache_control
				if strings.HasPrefix(p.model, "anthropic/") {
					formattedMsg["content"] = []map[string]interface{}{
						{
							"type": "text",
							"text": msg.Content,
							"cache_control": map[string]string{
								"type": "ephemeral",
							},
						},
					}
				}
			}
		}

		formattedMessages = append(formattedMessages, formattedMsg)
	}

	req["messages"] = formattedMessages

	// Handle tools/function calling if provided
	if tools, ok := req["tools"].([]interface{}); ok && len(tools) > 0 {
		req["tools"] = tools
	}

	if toolChoice, ok := req["tool_choice"]; ok {
		req["tool_choice"] = toolChoice
	}

	// Remove prompt caching flag as it's been handled
	delete(req, "enable_prompt_caching")

	// Add streaming if requested
	if stream, ok := req["stream"].(bool); ok && stream {
		req["stream"] = true
	}

	return json.Marshal(req)
}

func init() {
	// Register the OpenRouter provider
	registry := GetDefaultRegistry()
	registry.Register("openrouter", NewOpenRouterProvider)
}
