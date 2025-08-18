// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

const (
	openRouterDefaultModel = "openrouter/auto"

	// Option keys
	optionTemperature         = "temperature"
	optionMaxTokens           = "max_tokens"
	optionTopP                = "top_p"
	optionRoute               = "route"
	optionTransforms          = "transforms"
	optionProvider            = "provider"
	optionSeed                = "seed"
	optionEnableReasoning     = "enable_reasoning"
	optionFallbackModels      = "fallback_models"
	optionAutoRoute           = "auto_route"
	optionProviderPreferences = "provider_preferences"
	optionSystemMessage       = "system_message"
	optionTools               = "tools"
	optionToolChoice          = "tool_choice"
	optionStream              = "stream"
	optionEnablePromptCaching = "enable_prompt_caching"
)

// OpenRouterProvider implements the Provider interface for OpenRouter API.
// It provides access to multiple LLMs through a single API, with features like
// model routing, fallbacks, prompt caching.
type OpenRouterProvider struct {
	apiKey       string            // API key for authentication
	model        string            // Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
	extraHeaders map[string]string // Additional HTTP headers
	options      map[string]any    // Model-specific options
	logger       logging.Logger    // Logger instance
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
func NewOpenRouterProvider(apiKey, model string, extraHeaders map[string]string) *OpenRouterProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenRouterProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenRouter provider.
func (p *OpenRouterProvider) SetLogger(logger logging.Logger) {
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
	return "https://openrouter.ai/api/v1/generation?id=" + generationID
}

// SetOption sets a model-specific option for the OpenRouter provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - route: Routing strategy (e.g., "fallback", "lowest-latency")
//   - transforms: Array of transformations to apply to the response
//   - provider: Provider preferences for routing
func (p *OpenRouterProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *OpenRouterProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}

	// OpenRouter-specific defaults
	// Reasoning transforms are enabled via options rather than config
	if _, ok := p.options["enable_reasoning"]; ok {
		p.SetOption("transforms", []string{"reasoning"})
	}
}

// SupportsStreaming indicates whether this provider supports streaming responses.
// All OpenRouter models support streaming.
func (p *OpenRouterProvider) SupportsStreaming() bool {
	return true
}

// SupportsStructuredResponse indicates whether this provider supports structured response (JSON schema).
// OpenRouter supports structured responses for specific models.
//
// registry.
//
//nolint:funlen // This function is long due to the extensive list of models, it will be fixed with the new capability
func (p *OpenRouterProvider) SupportsStructuredResponse() bool {
	models := []string{
		"mistralai/mistral-medium-3.1",
		"openai/gpt-5-chat",
		"openai/gpt-5",
		"openai/gpt-5-mini",
		"openai/gpt-5-nano",
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b:free",
		"openai/gpt-oss-20b",
		"mistralai/codestral-2508",
		"z-ai/glm-4.5-air",
		"qwen/qwen3-coder",
		"google/gemini-2.5-flash-lite",
		"qwen/qwen3-235b-a22b-2507",
		"moonshotai/kimi-k2",
		"mistralai/devstral-medium",
		"mistralai/devstral-small",
		"cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
		"x-ai/grok-4",
		"thedrummer/anubis-70b-v1.1",
		"inception/mercury",
		"mistralai/mistral-small-3.2-24b-instruct:free",
		"mistralai/mistral-small-3.2-24b-instruct",
		"minimax/minimax-m1",
		"google/gemini-2.5-flash-lite-preview-06-17",
		"google/gemini-2.5-flash",
		"google/gemini-2.5-pro",
		"openai/o3-pro",
		"x-ai/grok-3-mini",
		"x-ai/grok-3",
		"mistralai/magistral-small-2506",
		"mistralai/magistral-medium-2506",
		"mistralai/magistral-medium-2506:thinking",
		"google/gemini-2.5-pro-preview",
		"deepseek/deepseek-r1-0528",
		"mistralai/devstral-small-2505",
		"openai/codex-mini",
		"mistralai/mistral-medium-3",
		"google/gemini-2.5-pro-preview-05-06",
		"inception/mercury-coder",
		"qwen/qwen3-4b:free",
		"qwen/qwen3-30b-a3b",
		"qwen/qwen3-14b",
		"qwen/qwen3-32b",
		"qwen/qwen3-235b-a22b:free",
		"qwen/qwen3-235b-a22b",
		"openai/o4-mini-high",
		"openai/o3",
		"openai/o4-mini",
		"openai/gpt-4.1",
		"openai/gpt-4.1-mini",
		"openai/gpt-4.1-nano",
		"meta-llama/llama-4-maverick",
		"meta-llama/llama-4-scout",
		"google/gemini-2.5-pro-exp-03-25",
		"qwen/qwen2.5-vl-32b-instruct",
		"deepseek/deepseek-chat-v3-0324",
		"openai/o1-pro",
		"mistralai/mistral-small-3.1-24b-instruct:free",
		"mistralai/mistral-small-3.1-24b-instruct",
		"google/gemma-3-4b-it:free",
		"google/gemma-3-12b-it:free",
		"cohere/command-a",
		"openai/gpt-4o-mini-search-preview",
		"openai/gpt-4o-search-preview",
		"google/gemma-3-27b-it:free",
		"thedrummer/skyfall-36b-v2",
		"qwen/qwq-32b:free",
		"qwen/qwq-32b",
		"google/gemini-2.0-flash-lite-001",
		"mistralai/mistral-saba",
		"openai/o3-mini-high",
		"google/gemini-2.0-flash-001",
		"qwen/qwen2.5-vl-72b-instruct:free",
		"openai/o3-mini",
		"mistralai/mistral-small-24b-instruct-2501",
		"deepseek/deepseek-r1-distill-llama-70b",
		"deepseek/deepseek-r1",
		"mistralai/codestral-2501",
		"microsoft/phi-4",
		"deepseek/deepseek-chat",
		"sao10k/l3.3-euryale-70b",
		"openai/o1",
		"cohere/command-r7b-12-2024",
		"meta-llama/llama-3.3-70b-instruct",
		"openai/gpt-4o-2024-11-20",
		"mistralai/mistral-large-2411",
		"mistralai/mistral-large-2407",
		"mistralai/pixtral-large-2411",
		"thedrummer/unslopnemo-12b",
		"mistralai/ministral-8b",
		"mistralai/ministral-3b",
		"qwen/qwen-2.5-7b-instruct",
		"google/gemini-flash-1.5-8b",
		"thedrummer/rocinante-12b",
		"meta-llama/llama-3.2-3b-instruct",
		"meta-llama/llama-3.2-11b-vision-instruct",
		"meta-llama/llama-3.2-1b-instruct",
		"neversleep/llama-3.1-lumimaid-8b",
		"mistralai/pixtral-12b",
		"cohere/command-r-plus-08-2024",
		"cohere/command-r-08-2024",
		"qwen/qwen-2.5-vl-7b-instruct",
		"sao10k/l3.1-euryale-70b",
		"nousresearch/hermes-3-llama-3.1-70b",
		"openai/chatgpt-4o-latest",
		"openai/gpt-4o-2024-08-06",
		"meta-llama/llama-3.1-405b-instruct:free",
		"meta-llama/llama-3.1-405b-instruct",
		"meta-llama/llama-3.1-70b-instruct",
		"meta-llama/llama-3.1-8b-instruct",
		"mistralai/mistral-nemo",
		"openai/gpt-4o-mini-2024-07-18",
		"openai/gpt-4o-mini",
		"google/gemma-2-27b-it",
		"nousresearch/hermes-2-pro-llama-3-8b",
		"google/gemini-flash-1.5",
		"openai/gpt-4o",
		"openai/gpt-4o:extended",
		"openai/gpt-4o-2024-05-13",
		"mistralai/mixtral-8x22b-instruct",
		"openai/gpt-4-turbo",
		"google/gemini-pro-1.5",
		"cohere/command-r-plus",
		"cohere/command-r-plus-04-2024",
		"cohere/command",
		"cohere/command-r",
		"cohere/command-r-03-2024",
		"mistralai/mistral-large",
		"openai/gpt-3.5-turbo-0613",
		"openai/gpt-4-turbo-preview",
		"mistralai/mistral-small",
		"mistralai/mistral-tiny",
		"neversleep/noromaid-20b",
		"alpindale/goliath-120b",
		"openai/gpt-4-1106-preview",
		"openai/gpt-3.5-turbo-instruct",
		"openai/gpt-3.5-turbo-16k",
		"undi95/remm-slerp-l2-13b",
		"gryphe/mythomax-l2-13b",
		"openai/gpt-4",
		"openai/gpt-3.5-turbo",
		"openai/gpt-4-0314",
	}

	return slices.Contains(models, p.model)
}

// SupportsFunctionCalling indicates whether this provider supports function calling.
// OpenRouter supports function calling for specific models.
//
// registry.
//
//nolint:funlen // This function is long due to the extensive list of models, it will be fixed with the new capability
func (p *OpenRouterProvider) SupportsFunctionCalling() bool {
	models := []string{
		"mistralai/mistral-medium-3.1",
		"z-ai/glm-4.5v",
		"ai21/jamba-mini-1.7",
		"ai21/jamba-large-1.7",
		"openai/gpt-5",
		"openai/gpt-5-mini",
		"openai/gpt-5-nano",
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b",
		"anthropic/claude-opus-4.1",
		"mistralai/codestral-2508",
		"z-ai/glm-4.5",
		"z-ai/glm-4.5-air:free",
		"z-ai/glm-4.5-air",
		"qwen/qwen3-235b-a22b-thinking-2507",
		"z-ai/glm-4-32b",
		"qwen/qwen3-coder:free",
		"qwen/qwen3-coder",
		"google/gemini-2.5-flash-lite",
		"qwen/qwen3-235b-a22b-2507",
		"moonshotai/kimi-k2:free",
		"moonshotai/kimi-k2",
		"mistralai/devstral-medium",
		"mistralai/devstral-small",
		"x-ai/grok-4",
		"inception/mercury",
		"mistralai/mistral-small-3.2-24b-instruct:free",
		"mistralai/mistral-small-3.2-24b-instruct",
		"minimax/minimax-m1",
		"google/gemini-2.5-flash-lite-preview-06-17",
		"google/gemini-2.5-flash",
		"google/gemini-2.5-pro",
		"openai/o3-pro",
		"x-ai/grok-3-mini",
		"x-ai/grok-3",
		"mistralai/magistral-small-2506",
		"mistralai/magistral-medium-2506",
		"mistralai/magistral-medium-2506:thinking",
		"google/gemini-2.5-pro-preview",
		"deepseek/deepseek-r1-0528",
		"anthropic/claude-opus-4",
		"anthropic/claude-sonnet-4",
		"mistralai/devstral-small-2505:free",
		"mistralai/devstral-small-2505",
		"openai/codex-mini",
		"mistralai/mistral-medium-3",
		"google/gemini-2.5-pro-preview-05-06",
		"arcee-ai/virtuoso-large",
		"inception/mercury-coder",
		"qwen/qwen3-4b:free",
		"qwen/qwen3-30b-a3b",
		"qwen/qwen3-14b",
		"qwen/qwen3-32b",
		"qwen/qwen3-235b-a22b:free",
		"qwen/qwen3-235b-a22b",
		"openai/o4-mini-high",
		"openai/o3",
		"openai/o4-mini",
		"openai/gpt-4.1",
		"openai/gpt-4.1-mini",
		"openai/gpt-4.1-nano",
		"x-ai/grok-3-mini-beta",
		"x-ai/grok-3-beta",
		"meta-llama/llama-4-maverick",
		"meta-llama/llama-4-scout",
		"google/gemini-2.5-pro-exp-03-25",
		"deepseek/deepseek-chat-v3-0324:free",
		"deepseek/deepseek-chat-v3-0324",
		"mistralai/mistral-small-3.1-24b-instruct:free",
		"mistralai/mistral-small-3.1-24b-instruct",
		"google/gemini-2.0-flash-lite-001",
		"anthropic/claude-3.7-sonnet",
		"anthropic/claude-3.7-sonnet:thinking",
		"anthropic/claude-3.7-sonnet:beta",
		"mistralai/mistral-saba",
		"openai/o3-mini-high",
		"google/gemini-2.0-flash-001",
		"qwen/qwen-turbo",
		"qwen/qwen-plus",
		"qwen/qwen-max",
		"openai/o3-mini",
		"mistralai/mistral-small-24b-instruct-2501",
		"deepseek/deepseek-r1-distill-llama-70b",
		"deepseek/deepseek-r1",
		"mistralai/codestral-2501",
		"deepseek/deepseek-chat",
		"openai/o1",
		"x-ai/grok-2-1212",
		"google/gemini-2.0-flash-exp:free",
		"meta-llama/llama-3.3-70b-instruct:free",
		"meta-llama/llama-3.3-70b-instruct",
		"amazon/nova-lite-v1",
		"amazon/nova-micro-v1",
		"amazon/nova-pro-v1",
		"openai/gpt-4o-2024-11-20",
		"mistralai/mistral-large-2411",
		"mistralai/mistral-large-2407",
		"mistralai/pixtral-large-2411",
		"thedrummer/unslopnemo-12b",
		"anthropic/claude-3.5-haiku-20241022",
		"anthropic/claude-3.5-haiku",
		"anthropic/claude-3.5-sonnet",
		"mistralai/ministral-8b",
		"nvidia/llama-3.1-nemotron-70b-instruct",
		"google/gemini-flash-1.5-8b",
		"thedrummer/rocinante-12b",
		"meta-llama/llama-3.2-3b-instruct",
		"qwen/qwen-2.5-72b-instruct",
		"mistralai/pixtral-12b",
		"cohere/command-r-plus-08-2024",
		"cohere/command-r-08-2024",
		"microsoft/phi-3.5-mini-128k-instruct",
		"nousresearch/hermes-3-llama-3.1-70b",
		"openai/gpt-4o-2024-08-06",
		"meta-llama/llama-3.1-405b-instruct",
		"meta-llama/llama-3.1-70b-instruct",
		"meta-llama/llama-3.1-8b-instruct",
		"mistralai/mistral-nemo",
		"openai/gpt-4o-mini-2024-07-18",
		"openai/gpt-4o-mini",
		"anthropic/claude-3.5-sonnet-20240620",
		"mistralai/mistral-7b-instruct-v0.3",
		"mistralai/mistral-7b-instruct:free",
		"mistralai/mistral-7b-instruct",
		"microsoft/phi-3-mini-128k-instruct",
		"microsoft/phi-3-medium-128k-instruct",
		"google/gemini-flash-1.5",
		"openai/gpt-4o",
		"openai/gpt-4o:extended",
		"openai/gpt-4o-2024-05-13",
		"meta-llama/llama-3-70b-instruct",
		"meta-llama/llama-3-8b-instruct",
		"mistralai/mixtral-8x22b-instruct",
		"openai/gpt-4-turbo",
		"google/gemini-pro-1.5",
		"cohere/command-r-plus",
		"cohere/command-r-plus-04-2024",
		"cohere/command-r",
		"anthropic/claude-3-haiku",
		"anthropic/claude-3-opus",
		"cohere/command-r-03-2024",
		"mistralai/mistral-large",
		"openai/gpt-3.5-turbo-0613",
		"openai/gpt-4-turbo-preview",
		"mistralai/mistral-small",
		"mistralai/mistral-tiny",
		"mistralai/mixtral-8x7b-instruct",
		"openai/gpt-4-1106-preview",
		"mistralai/mistral-7b-instruct-v0.1",
		"openai/gpt-3.5-turbo-16k",
		"openai/gpt-4",
		"openai/gpt-3.5-turbo",
		"openai/gpt-4-0314",
	}

	return slices.Contains(models, p.model)
}

// Headers return the HTTP headers required for OpenRouter API requests.
func (p *OpenRouterProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
		"HTTP-Referer":  "https://github.com/weave-labs/gollm", // Identify the app to OpenRouter
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
func (p *OpenRouterProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody(options)
	p.handleModelRouting(requestBody)
	p.handleProviderPreferences(requestBody)
	p.addMessages(requestBody, req)
	p.handleStructuredResponse(requestBody, req)
	p.handleToolsAndOptions(requestBody)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	return data, nil
}

// initializeRequestBody creates the base request body with options and model
func (p *OpenRouterProvider) initializeRequestBody(options map[string]any) map[string]any {
	requestBody := map[string]any{}
	for k, v := range options {
		requestBody[k] = v
	}

	requestBody["model"] = p.model

	for k, v := range p.options {
		if _, exists := requestBody[k]; !exists {
			requestBody[k] = v
		}
	}

	return requestBody
}

// handleModelRouting processes fallback models and auto-routing options
func (p *OpenRouterProvider) handleModelRouting(requestBody map[string]any) {
	if fallbackModels, ok := requestBody[optionFallbackModels].([]string); ok {
		requestBody["models"] = append([]string{p.model}, fallbackModels...)
		delete(requestBody, optionFallbackModels)
	} else if autoRoute, ok := requestBody[optionAutoRoute].(bool); ok && autoRoute {
		requestBody["model"] = openRouterDefaultModel
		delete(requestBody, optionAutoRoute)
	}
}

// handleProviderPreferences processes provider preference options
func (p *OpenRouterProvider) handleProviderPreferences(requestBody map[string]any) {
	if providerPrefs, ok := requestBody[optionProviderPreferences].(map[string]any); ok {
		requestBody[optionProvider] = providerPrefs
		delete(requestBody, optionProviderPreferences)
	}
}

// addMessages adds system prompt and messages to the request body
func (p *OpenRouterProvider) addMessages(requestBody map[string]any, req *Request) {
	messages := make([]map[string]any, 0, len(req.Messages)+1)

	// Add system prompt if provided
	if req.SystemPrompt != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": req.SystemPrompt,
		})
	}

	// Add messages from the Request
	for _, msg := range req.Messages {
		openRouterMsg := p.convertMessage(&msg)
		messages = append(messages, openRouterMsg)
	}

	requestBody["messages"] = messages
}

// convertMessage converts a Message to OpenRouter format
func (p *OpenRouterProvider) convertMessage(msg *Message) map[string]any {
	openRouterMsg := map[string]any{
		"role":    msg.Role,
		"content": msg.Content,
	}
	if msg.Name != "" {
		openRouterMsg["name"] = msg.Name
	}
	if msg.ToolCallID != "" {
		openRouterMsg["tool_call_id"] = msg.ToolCallID
	}
	if len(msg.ToolCalls) > 0 {
		openRouterMsg["tool_calls"] = msg.ToolCalls
	}
	return openRouterMsg
}

// handleStructuredResponse adds structured response schema if supported
func (p *OpenRouterProvider) handleStructuredResponse(requestBody map[string]any, req *Request) {
	if req.ResponseSchema != nil && p.SupportsStructuredResponse() {
		requestBody["response_format"] = map[string]any{
			"type":   "json_object",
			"schema": req.ResponseSchema,
		}
	}
}

// handleToolsAndOptions processes tools, streaming, and caching options
func (p *OpenRouterProvider) handleToolsAndOptions(requestBody map[string]any) {
	// Handle tools/function calling if provided
	if tools, ok := requestBody[optionTools].([]any); ok && len(tools) > 0 {
		requestBody[optionTools] = tools
	}

	if toolChoice, ok := requestBody[optionToolChoice]; ok {
		requestBody[optionToolChoice] = toolChoice
	}

	// Add streaming if requested
	if stream, ok := requestBody[optionStream].(bool); ok && stream {
		requestBody[optionStream] = true
	}

	// Handle prompt caching for supported models
	if caching, ok := requestBody[optionEnablePromptCaching].(bool); ok && caching {
		// OpenRouter handles caching automatically for supported providers
		delete(requestBody, optionEnablePromptCaching)
	}
}

// PrepareCompletionRequest creates a text completion request for the OpenRouter API.
// This uses the legacy completions endpoint rather than chat completions.
func (p *OpenRouterProvider) PrepareCompletionRequest(prompt string, options map[string]any) ([]byte, error) {
	// Start with the passed options
	req := map[string]any{}
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
		req["model"] = openRouterDefaultModel
		delete(req, "auto_route")
	}

	// Add the prompt
	req["prompt"] = prompt

	// Handle provider routing preferences if provided
	if providerPrefs, ok := req["provider_preferences"].(map[string]any); ok {
		req["provider"] = providerPrefs
		delete(req, "provider_preferences")
	}

	// Add streaming if requested
	if stream, ok := req["stream"].(bool); ok && stream {
		req["stream"] = true
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	return data, nil
}

// ParseResponse extracts the completion text from the OpenRouter API response.
func (p *OpenRouterProvider) ParseResponse(body []byte) (*Response, error) {
	// Try to parse as chat completion first
	resp, err := p.parseChatCompletion(body)
	if err == nil {
		return resp, nil
	}

	// If chat completion failed, try text completion
	return p.parseTextCompletion(body, err)
}

// parseChatCompletion attempts to parse the response as a chat completion
func (p *OpenRouterProvider) parseChatCompletion(body []byte) (*Response, error) {
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
		Usage *struct {
			PromptTokens             int64 `json:"prompt_tokens"`
			CompletionTokens         int64 `json:"completion_tokens"`
			TotalTokens              int64 `json:"total_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal chat response: %w", err)
	}

	// Check if we have valid chat completion choices
	if len(chatResp.Choices) == 0 || chatResp.Choices[0].Message.Content == "" {
		return nil, errors.New("invalid chat completion response")
	}

	// Check for errors
	if chatResp.Error.Message != "" {
		return nil, fmt.Errorf("OpenRouter API error: %s", chatResp.Error.Message)
	}

	// Log generation info
	if chatResp.ID != "" {
		p.logger.Debug("Generation ID", "id", chatResp.ID)
	}
	if chatResp.Model != "" && chatResp.Model != p.model {
		p.logger.Info("Model used", "requested", p.model, "actual", chatResp.Model)
	}

	resp := &Response{Content: Text{Value: chatResp.Choices[0].Message.Content}}
	if chatResp.Usage != nil {
		resp.Usage = NewUsage(
			chatResp.Usage.PromptTokens,
			chatResp.Usage.CacheCreationInputTokens+chatResp.Usage.CacheReadInputTokens,
			chatResp.Usage.CompletionTokens,
			0,
			0,
		)
	}
	return resp, nil
}

// parseTextCompletion attempts to parse the response as a text completion
func (p *OpenRouterProvider) parseTextCompletion(body []byte, chatErr error) (*Response, error) {
	var textResp struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		ID    string `json:"id"`
		Model string `json:"model"`
		Usage *struct {
			PromptTokens             int64 `json:"prompt_tokens"`
			CompletionTokens         int64 `json:"completion_tokens"`
			TotalTokens              int64 `json:"total_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &textResp); err != nil {
		// Return the original chat parsing error if text parsing also fails
		return nil, fmt.Errorf("error parsing OpenRouter response: %w", chatErr)
	}

	// Check for errors
	if textResp.Error.Message != "" {
		return nil, fmt.Errorf("OpenRouter API error: %s", textResp.Error.Message)
	}

	// Check if we have at least one choice
	if len(textResp.Choices) == 0 {
		return nil, errors.New("no completion choices in OpenRouter response")
	}

	// Log generation info
	if textResp.ID != "" {
		p.logger.Debug("Generation ID (text completion)", "id", textResp.ID)
	}
	if textResp.Model != "" && textResp.Model != p.model {
		p.logger.Info("Model used (text completion)", "requested", p.model, "actual", textResp.Model)
	}

	p.logger.Debug("Parsed text completion", "text", textResp.Choices[0].Text)

	resp := &Response{Content: Text{Value: textResp.Choices[0].Text}}
	if textResp.Usage != nil {
		resp.Usage = NewUsage(
			textResp.Usage.PromptTokens,
			textResp.Usage.CacheCreationInputTokens+textResp.Usage.CacheReadInputTokens,
			textResp.Usage.CompletionTokens,
			0,
			0,
		)
	}
	return resp, nil
}

// SetExtraHeaders configures additional HTTP headers for OpenRouter API requests.
func (p *OpenRouterProvider) SetExtraHeaders(extraHeaders map[string]string) {
	if extraHeaders == nil {
		p.extraHeaders = make(map[string]string)
		return
	}
	p.extraHeaders = extraHeaders
}

// PrepareStreamRequest creates a streaming request for the OpenRouter API.
func (p *OpenRouterProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	if !p.SupportsStreaming() {
		return nil, errors.New("streaming is not supported by this provider")
	}

	streamOptions := make(map[string]any)
	for k, v := range options {
		streamOptions[k] = v
	}
	streamOptions[optionStream] = true
	return p.PrepareRequest(req, streamOptions)
}

// ParseStreamResponse processes a chunk from a streaming OpenRouter response.
func (p *OpenRouterProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Skip empty chunks
	if len(bytes.TrimSpace(chunk)) == 0 {
		return nil, errors.New("empty chunk")
	}
	// Handle "[DONE]" marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return nil, io.EOF
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
		return nil, fmt.Errorf("malformed response: %w", err)
	}

	// Check for errors
	if resp.Error.Message != "" {
		return nil, fmt.Errorf("OpenRouter API streaming error: %s", resp.Error.Message)
	}

	// Store the generation ID and used model in the logger for potential later use
	if resp.ID != "" {
		p.logger.Debug("Streaming generation ID", "id", resp.ID)
	}
	if resp.Model != "" && resp.Model != p.model {
		p.logger.Info("Model used for streaming", "requested", p.model, "actual", resp.Model)
	}

	// Prepare usage if present
	var usage *Usage
	if resp.Usage != nil {
		p.logger.Debug("Token usage",
			"prompt_tokens", resp.Usage.PromptTokens,
			"completion_tokens", resp.Usage.CompletionTokens,
			"total_tokens", resp.Usage.TotalTokens)
		usage = NewUsage(int64(resp.Usage.PromptTokens), 0, int64(resp.Usage.CompletionTokens), 0, 0)
	}

	// Check if we have at least one choice with content
	if len(resp.Choices) == 0 || resp.Choices[0].Delta.Content == "" {
		return nil, errors.New("skip token")
	}

	// Handle tool calls in streaming mode (log only)
	if len(resp.Choices[0].Delta.ToolCalls) > 0 {
		if toolCallData, err := json.Marshal(resp.Choices[0].Delta.ToolCalls); err == nil {
			p.logger.Debug("Tool call in streaming mode", "data", string(toolCallData))
		}
	}

	return &Response{
		Content: Text{Value: resp.Choices[0].Delta.Content},
		Usage:   usage,
	}, nil
}
