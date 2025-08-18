// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Groq-specific parameter keys
const (
	groqKeyMessages                 = "messages"
	groqKeySystemPrompt             = "system_prompt"
	groqKeyTools                    = "tools"
	groqKeyToolChoice               = "tool_choice"
	groqKeyStructuredMessages       = "structured_messages"
	groqKeyStructuredResponseSchema = "structured_response_schema"
	groqKeyMaxTokens                = "max_tokens"
	groqKeyStream                   = "stream"
	groqKeyModel                    = "model"
)

// GroqProvider implements the Provider interface for Groq's API.
// It supports Groq's optimized language models and provides access to their
// high-performance inference capabilities.
type GroqProvider struct {
	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewGroqProvider creates a new Groq provider instance.
// It initializes the provider with the given API key, model, and optional headers.
func NewGroqProvider(apiKey, model string, extraHeaders map[string]string) *GroqProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &GroqProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Groq provider.
// This is used for debugging and monitoring API interactions.
func (p *GroqProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// Name returns the identifier for this provider ("groq").
func (p *GroqProvider) Name() string {
	return "groq"
}

// Endpoint returns the Groq API endpoint URL.
// This is "https://api.groq.com/openai/v1/chat/completions".
func (p *GroqProvider) Endpoint() string {
	return "https://api.groq.com/openai/v1/chat/completions"
}

// Headers returns the HTTP headers required for Groq API requests.
// This includes the authorization token and content type headers.
func (p *GroqProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *GroqProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *GroqProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption(groqKeyMaxTokens, cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
}

// SetOption sets a model-specific option for the Groq provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
func (p *GroqProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// PrepareRequest creates the request body for a Groq API call using the new Request structure.
// It formats the messages and options according to Groq's API requirements.
func (p *GroqProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	p.addMessagesToRequestBody(requestBody, req.Messages, options)

	if req.SystemPrompt != "" {
		p.addSystemPromptToRequestBody(requestBody, req.SystemPrompt)
	}

	if req.ResponseSchema != nil && options[groqKeyStream] != true {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// PrepareStreamRequest creates the request body for a streaming Groq API call.
// It uses the same structure as PrepareRequest but adds the stream parameter.
func (p *GroqProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	if !p.SupportsStreaming() {
		return nil, errors.New("streaming is not supported by this provider")
	}

	if options == nil {
		options = make(map[string]any)
	}
	options[groqKeyStream] = true

	return p.PrepareRequest(req, options)
}

// ParseResponse extracts the generated text from the Groq API response.
// It handles Groq's response format and extracts the content.
func (p *GroqProvider) ParseResponse(body []byte) (*Response, error) {
	var response struct {
		Usage *struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
			TotalTokens      int64 `json:"total_tokens"`
		} `json:"usage"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	err := json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return nil, errors.New("empty response from API")
	}

	resp := &Response{Content: Text{Value: response.Choices[0].Message.Content}}
	if response.Usage != nil {
		resp.Usage = NewUsage(response.Usage.PromptTokens, 0, response.Usage.CompletionTokens, 0, 0)
	}
	return resp, nil
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *GroqProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	var response struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(chunk, &response); err != nil {
		return nil, fmt.Errorf("malformed response: %w", err)
	}
	if len(response.Choices) == 0 || response.Choices[0].Delta.Content == "" {
		return nil, errors.New("skip resp")
	}
	return &Response{Content: Text{Value: response.Choices[0].Delta.Content}}, nil
}

// SupportsStreaming indicates that Groq supports streaming responses
func (p *GroqProvider) SupportsStreaming() bool {
	supportedModels := []string{
		"openai/gpt-oss-20b",
		"openai/gpt-oss-120b",
		"moonshotai/kimi-k2-instruct",
		"meta-llama/llama-4-maverick-17b-128e-instruct",
		"meta-llama/llama-4-scout-17b-16e-instruct",
	}

	return slices.Contains(supportedModels, p.model)
}

// SupportsStructuredResponse indicates that Groq supports structured output
// through JSON schema validation
func (p *GroqProvider) SupportsStructuredResponse() bool {
	return true
}

// SupportsFunctionCalling indicates that Groq supports function calling
func (p *GroqProvider) SupportsFunctionCalling() bool {
	return true
}

// Private helper methods

// initializeRequestBody creates the base request body with model information
func (p *GroqProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		groqKeyModel: p.model,
	}
}

// addMessagesToRequestBody adds messages to the request body in Groq format
func (p *GroqProvider) addMessagesToRequestBody(
	requestBody map[string]any,
	messages []Message,
	_ map[string]any,
) {
	groqMessages := make([]map[string]any, 0, len(messages))

	for _, msg := range messages {
		groqMessage := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		// Add optional fields if present
		if msg.Name != "" {
			groqMessage["name"] = msg.Name
		}
		if msg.ToolCallID != "" {
			groqMessage["tool_call_id"] = msg.ToolCallID
		}
		if len(msg.ToolCalls) > 0 {
			groqMessage["tool_calls"] = msg.ToolCalls
		}

		groqMessages = append(groqMessages, groqMessage)
	}

	requestBody[groqKeyMessages] = groqMessages
}

// addSystemPromptToRequestBody adds system prompt as a system message
func (p *GroqProvider) addSystemPromptToRequestBody(requestBody map[string]any, systemPrompt string) {
	if messages, ok := requestBody[groqKeyMessages].([]map[string]any); ok {
		systemMessage := map[string]any{
			"role":    "system",
			"content": systemPrompt,
		}
		// Prepend system message
		requestBody[groqKeyMessages] = append([]map[string]any{systemMessage}, messages...)
	}
}

// addStructuredResponseToRequest adds structured response schema to the request
func (p *GroqProvider) addStructuredResponseToRequest(requestBody map[string]any, schema any) {
	requestBody["response_format"] = map[string]any{
		"type":   "json_schema",
		"schema": schema,
	}
}

// addRemainingOptions adds provider options and request options to the request body
func (p *GroqProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	// Add provider options first
	for k, v := range p.options {
		if !p.isGlobalOption(k) {
			requestBody[k] = v
		}
	}

	// Add request options (may override provider options)
	for k, v := range options {
		if !p.isGlobalOption(k) {
			requestBody[k] = v
		}
	}
}

// isGlobalOption checks if a key is a global option that should not be added to request body
func (p *GroqProvider) isGlobalOption(key string) bool {
	switch key {
	case groqKeyMessages, groqKeySystemPrompt, groqKeyStructuredMessages, groqKeyStructuredResponseSchema:
		return true
	default:
		return false
	}
}
