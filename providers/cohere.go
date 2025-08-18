package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

const (
	cohereKeyText           = "text"
	cohereKeySystemPrompt   = "system_prompt"
	cohereKeyPreamble       = "preamble"
	cohereKeyMessages       = "messages"
	cohereKeyResponseFormat = "response_format"
	cohereKeyStream         = "stream"
)

// CohereProvider implements the Provider interface for Cohere's API.
// It supports Cohere's language models and provides access to their capabilities,
// including chat completion and structured output
type CohereProvider struct {
	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewCohereProvider creates a new Cohere provider instance.
// It initializes the provider with the given API key, model, and optional headers.
func NewCohereProvider(apiKey, model string, extraHeaders map[string]string) *CohereProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	return &CohereProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
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

// Headers return the required HTTP headers for Cohere API requests.
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

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *CohereProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *CohereProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	p.SetOption(cohereKeyStream, false)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
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

// SetLogger configures the logger for the Cohere provider.
// This is used for debugging and monitoring API interactions.
func (p *CohereProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// SupportsStructuredResponse indicates that Cohere supports structured output
// through its system prompts and response formatting capabilities.
// Only specific models support structured output.
func (p *CohereProvider) SupportsStructuredResponse() bool {
	// Models that support structured output according to Cohere documentation
	supportedModels := []string{
		"command-a-03-2025",
		"command-r-plus-08-2024",
		"command-r-plus",
		"command-r-08-2024",
		"command-r",
	}

	return slices.Contains(supportedModels, p.model)
}

// SupportsStreaming returns whether the provider supports streaming responses
func (p *CohereProvider) SupportsStreaming() bool {
	return true
}

// SupportsFunctionCalling indicates if the provider supports function calling
func (p *CohereProvider) SupportsFunctionCalling() bool {
	return true
}

// PrepareRequest creates the request body for a Cohere API call
func (p *CohereProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	p.addMessagesToRequestBody(requestBody, req.Messages)

	if req.SystemPrompt != "" {
		requestBody[cohereKeyPreamble] = req.SystemPrompt
	} else if systemPrompt, ok := options[cohereKeySystemPrompt].(string); ok && systemPrompt != "" {
		requestBody[cohereKeyPreamble] = systemPrompt
	}

	if req.ResponseSchema != nil && p.SupportsStructuredResponse() {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseResponse extracts the generated text from the Cohere API response.
// It handles various response formats and error cases
func (p *CohereProvider) ParseResponse(body []byte) (*Response, error) {
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

	if len(response.Message.Content) == 0 {
		return nil, errors.New("empty response from API")
	}

	var finalResponse strings.Builder

	for _, content := range response.Message.Content {
		if content.Type == cohereKeyText {
			finalResponse.WriteString(content.Text)
			p.logger.Debug("Text content: %s", content.Text)
		}
	}

	for _, toolCall := range response.Message.ToolCalls {
		var args any
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("error parsing function arguments: %w", err)
		}

		functionCall, err := FormatFunctionCall(toolCall.Function.Name, args)
		if err != nil {
			return nil, fmt.Errorf("error formatting function call: %w", err)
		}
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(functionCall)
	}

	p.logger.Debug("Final response: %s", finalResponse.String())
	return &Response{Content: Text{Value: finalResponse.String()}}, nil
}

// PrepareStreamRequest prepares a request body for streaming
func (p *CohereProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()
	requestBody[cohereKeyStream] = true

	p.addMessagesToRequestBody(requestBody, req.Messages)

	if req.SystemPrompt != "" {
		requestBody[cohereKeyPreamble] = req.SystemPrompt
	} else if systemPrompt, ok := options[cohereKeySystemPrompt].(string); ok && systemPrompt != "" {
		requestBody[cohereKeyPreamble] = systemPrompt
	}

	if req.ResponseSchema != nil && p.SupportsStructuredResponse() {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *CohereProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	var response struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(chunk, &response); err != nil {
		return nil, fmt.Errorf("malformed response: %w", err)
	}
	if response.Text == "" {
		return nil, errors.New("skip resp")
	}
	return &Response{Content: Text{Value: response.Text}}, nil
}

// initializeRequestBody creates the base request structure
func (p *CohereProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		"model":           p.model,
		cohereKeyMessages: []map[string]any{},
	}
}

// addMessagesToRequestBody converts and adds messages to the request
func (p *CohereProvider) addMessagesToRequestBody(
	requestBody map[string]any,
	messages []Message,
) {
	cohereMessages := make([]map[string]any, 0, len(messages))

	for i := range messages {
		cohereMsg := p.convertMessageToCohereFormat(&messages[i])
		cohereMessages = append(cohereMessages, cohereMsg)
	}

	requestBody[cohereKeyMessages] = cohereMessages
}

// convertMessageToCohereFormat converts a Message to Cohere's format
func (p *CohereProvider) convertMessageToCohereFormat(msg *Message) map[string]any {
	cohereMsg := map[string]any{
		"role":    msg.Role,
		"content": msg.Content,
	}

	if msg.Name != "" {
		cohereMsg["name"] = msg.Name
	}

	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]map[string]any, len(msg.ToolCalls))
		for i, toolCall := range msg.ToolCalls {
			toolCalls[i] = map[string]any{
				"id":   toolCall.ID,
				"type": toolCall.Type,
				"function": map[string]any{
					"name":      toolCall.Function.Name,
					"arguments": string(toolCall.Function.Arguments),
				},
			}
		}
		cohereMsg["tool_calls"] = toolCalls
	}

	return cohereMsg
}

// addStructuredResponseToRequest adds structured response schema to the request
func (p *CohereProvider) addStructuredResponseToRequest(requestBody map[string]any, schema any) {
	requestBody[cohereKeyResponseFormat] = map[string]any{
		"type":        "json_object",
		"json_schema": schema,
	}
}

// addRemainingOptions adds non-handled options to the request
func (p *CohereProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	// First, add default options
	for k, v := range p.options {
		if !p.isGlobalOption(k) {
			requestBody[k] = v
		}
	}

	// Then, add any additional options (which may override defaults)
	for k, v := range options {
		if !p.isGlobalOption(k) {
			requestBody[k] = v
		}
	}
}

// isGlobalOption checks if an option is already handled
func (p *CohereProvider) isGlobalOption(key string) bool {
	return key == cohereKeySystemPrompt ||
		key == cohereKeyPreamble ||
		key == cohereKeyMessages ||
		key == cohereKeyResponseFormat ||
		key == cohereKeyStream
}
