// Package providers implement LLM provider interfaces and their implementations.
package providers

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"strings"

	"github.com/weave-labs/gollm/internal/models"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Gemini-specific parameter keys
const (
	geminiKeySystemPrompt       = "system_prompt"
	geminiKeyTools              = "tools"
	geminiKeyToolChoice         = "tool_choice"
	geminiKeyStructuredMessages = "structured_messages"
	geminiKeyMaxOutputTokens    = "maxOutputTokens"
	geminiKeyTemperature        = "temperature"
	geminiKeyTopP               = "topP"
	geminiKeyTopK               = "topK"
	geminiKeyStopSequences      = "stopSequences"
)

// GeminiProvider implements the Provider interface for Google's Gemini API (Generative Language API).
// It supports chat completions with system instructions, function/tool calling with JSON schemas,
// streaming via Server-Sent Events (SSE), and usage reporting.
type GeminiProvider struct {
	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
	streamMode   bool
}

// NewGeminiProvider creates a new Google Gemini API provider instance.
func NewGeminiProvider(apiKey, model string, extraHeaders map[string]string) *GeminiProvider {
	provider := &GeminiProvider{
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

// SetLogger configures the logger for the Gemini provider.
// This is used for debugging and monitoring API interactions.
func (p *GeminiProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Gemini provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
//   - stop_sequences: Custom stop sequences
func (p *GeminiProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *GeminiProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption(geminiKeyTemperature, cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
}

// Name returns "google" as the provider identifier.
func (p *GeminiProvider) Name() string {
	return "google"
}

// Endpoint returns the Google Gemini API endpoint URL.
func (p *GeminiProvider) Endpoint() string {
	modelName := p.model
	if !strings.HasPrefix(modelName, "models/") {
		modelName = "models/" + modelName
	}
	if p.streamMode {
		// Streaming endpoint with SSE
		return fmt.Sprintf(
			"https://generativelanguage.googleapis.com/v1beta/%s:streamGenerateContent?alt=sse",
			modelName,
		)
	}
	return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s:generateContent", modelName)
}

// Headers return the HTTP headers required for Google AI requests.
func (p *GeminiProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type": "application/json",
	}
	if p.streamMode {
		headers["x-goog-api-key"] = p.apiKey
		headers["Accept"] = "text/event-stream"
	} else {
		headers["Authorization"] = "Bearer " + p.apiKey
	}

	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

// SetExtraHeaders configures additional HTTP headers for API requests.
func (p *GeminiProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// PrepareRequest prepares a request using the new unified Request structure.
func (p *GeminiProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	systemPrompt := p.extractSystemPromptFromRequest(req, options)

	p.addSystemPromptToRequestBody(requestBody, systemPrompt)

	p.handleToolsForRequest(requestBody, options)

	if req.ResponseSchema != nil {
		if err := p.addStructuredResponseToRequest(requestBody, req.ResponseSchema); err != nil {
			return nil, fmt.Errorf("failed to add structured response: %w", err)
		}
	}

	p.addMessagesToRequestBody(requestBody, req.Messages)

	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// PrepareStreamRequest prepares a streaming request using the new unified Request structure.
func (p *GeminiProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	if !p.SupportsStreaming() {
		return nil, errors.New("streaming is not supported by this provider")
	}

	p.streamMode = true
	return p.PrepareRequest(req, options)
}

// ParseResponse parses the response from the Gemini API.
func (p *GeminiProvider) ParseResponse(body []byte) (*Response, error) {
	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.New("no candidates in response")
	}

	candidate := geminiResp.Candidates[0]
	if len(candidate.Content.Parts) == 0 {
		return nil, errors.New("no content parts in response")
	}

	// Build response content
	var finalText strings.Builder
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			finalText.WriteString(part.Text)
		}
		if part.FunctionCall != nil {
			functionCallText := p.formatFunctionCall(part.FunctionCall)
			finalText.WriteString(functionCallText)
		}
	}

	response := &Response{
		Role:    "assistant",
		Content: Text{Value: finalText.String()},
	}

	// Add usage information if available
	if geminiResp.UsageMetadata != nil {
		um := geminiResp.UsageMetadata
		inputTokens := um.PromptTokenCount
		cachedInput := um.CachedContentTokenCount
		outputTokens := um.CandidatesTokenCount
		cachedOutput := int64(0)
		response.Usage = NewUsage(inputTokens, cachedInput, outputTokens, cachedOutput, 0)
	}

	return response, nil
}

// ParseStreamResponse parses streaming response chunks from the Gemini API.
func (p *GeminiProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Handle SSE format - remove "data: " prefix if present
	dataStr := strings.TrimPrefix(string(chunk), "data: ")

	// Skip empty chunks or [DONE] markers
	if strings.TrimSpace(dataStr) == "" || strings.TrimSpace(dataStr) == "[DONE]" {
		return nil, errors.New("skip chunk")
	}

	var resp geminiResponse
	if err := json.Unmarshal([]byte(dataStr), &resp); err != nil {
		// Skip malformed chunks
		return nil, errors.New("skip chunk")
	}

	// If usage metadata is present (possibly in the final chunk), return a Response with usage info
	if resp.UsageMetadata != nil {
		um := resp.UsageMetadata
		usageResp := &Response{
			Usage: NewUsage(um.PromptTokenCount, um.CachedContentTokenCount, um.CandidatesTokenCount, 0, 0),
		}
		return usageResp, nil
	}

	// If no candidates or parts, skip this chunk
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("skip chunk")
	}

	candidate := resp.Candidates[0]
	var finalText strings.Builder

	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			finalText.WriteString(part.Text)
		}
		if part.FunctionCall != nil {
			p.processFunctionCall(part.FunctionCall, &finalText)
		}
	}

	if finalText.Len() == 0 {
		return nil, errors.New("skip chunk")
	}

	return &Response{
		Role:    "assistant",
		Content: Text{Value: finalText.String()},
	}, nil
}

// Private helper methods

func (p *GeminiProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		"contents": []map[string]any{},
	}
}

func (p *GeminiProvider) extractSystemPromptFromRequest(req *Request, options map[string]any) string {
	if req.SystemPrompt != "" {
		return req.SystemPrompt
	}
	if systemPrompt, ok := options[geminiKeySystemPrompt].(string); ok {
		return systemPrompt
	}
	return ""
}

func (p *GeminiProvider) addSystemPromptToRequestBody(requestBody map[string]any, systemPrompt string) {
	if systemPrompt == "" {
		return
	}

	requestBody["systemInstruction"] = map[string]any{
		"parts": []map[string]any{
			{"text": systemPrompt},
		},
	}
}

func (p *GeminiProvider) handleToolsForRequest(requestBody map[string]any, options map[string]any) {
	tools, ok := options[geminiKeyTools].([]models.Tool)
	if !ok || len(tools) == 0 {
		return
	}

	funcDecls := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		funcDecl := map[string]any{
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
		}
		funcDecls = append(funcDecls, funcDecl)
	}

	requestBody[geminiKeyTools] = []map[string]any{
		{"functionDeclarations": funcDecls},
	}

	// Add function calling mode if specified
	if mode, ok := options["function_call_mode"].(string); ok && mode != "" {
		requestBody["toolConfig"] = map[string]any{
			"functionCallingConfig": map[string]any{
				"mode": mode,
			},
		}
	}
}

func (p *GeminiProvider) addStructuredResponseToRequest(requestBody map[string]any, schema *jsonschema.Schema) error {
	if schema == nil {
		return nil
	}

	//schemaJSON, err := schema.MarshalJSON()
	//if err != nil {
	//	return fmt.Errorf("failed to marshal response schema: %w", err)
	//}

	if genConfig, ok := requestBody["generationConfig"].(map[string]any); ok {
		genConfig["responseMimeType"] = "application/json"
		genConfig["responseSchema"] = schema
	} else {
		requestBody["generationConfig"] = map[string]any{
			"responseMimeType": "application/json",
			"responseSchema":   schema,
		}
	}

	return nil
}

func (p *GeminiProvider) addMessagesToRequestBody(requestBody map[string]any, messages []Message) {
	contents := make([]map[string]any, 0, len(messages))

	for i := range messages {
		content := p.convertMessageToGeminiFormat(&messages[i])
		if content != nil {
			contents = append(contents, content)
		}
	}

	requestBody["contents"] = contents
}

func (p *GeminiProvider) convertMessageToGeminiFormat(msg *Message) map[string]any {
	role := p.mapRoleToGemini(msg.Role)
	if role == "" {
		return nil // Skip unknown roles
	}

	parts := make([]map[string]any, 0, len(msg.ToolCalls)+1)

	if msg.Content != "" {
		parts = append(parts, map[string]any{"text": msg.Content})
	}

	for _, toolCall := range msg.ToolCalls {
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{
				"name": toolCall.Function.Name,
				"args": toolCall.Function.Arguments,
			},
		})
	}

	return map[string]any{
		"role":  role,
		"parts": parts,
	}
}

func (p *GeminiProvider) mapRoleToGemini(role string) string {
	switch role {
	case "user":
		return "user"
	case "assistant":
		return "model"
	case "tool":
		return "function"
	default:
		return ""
	}
}

func (p *GeminiProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	// Build generation config
	genConfig := p.buildGenerationConfig()
	if len(genConfig) > 0 {
		if existing, ok := requestBody["generationConfig"].(map[string]any); ok {
			// Merge with existing generation config
			for k, v := range genConfig {
				existing[k] = v
			}
		} else {
			requestBody["generationConfig"] = genConfig
		}
	}

	// Add any remaining unhandled options
	for key, value := range options {
		if !p.isGlobalOption(key) {
			requestBody[key] = value
		}
	}
}

func (p *GeminiProvider) buildGenerationConfig() map[string]any {
	genConfig := make(map[string]any)

	if temp, ok := p.options[geminiKeyTemperature]; ok {
		genConfig["temperature"] = temp
	}
	if maxTokens, ok := p.options["max_tokens"]; ok {
		genConfig[geminiKeyMaxOutputTokens] = maxTokens
	}
	if topP, ok := p.options["top_p"]; ok {
		genConfig[geminiKeyTopP] = topP
	}
	if topK, ok := p.options["top_k"]; ok {
		genConfig[geminiKeyTopK] = topK
	}
	if stopSeq, ok := p.options["stop_sequences"]; ok {
		genConfig[geminiKeyStopSequences] = stopSeq
	}

	return genConfig
}

func (p *GeminiProvider) isGlobalOption(key string) bool {
	switch key {
	case geminiKeySystemPrompt, geminiKeyTools, geminiKeyToolChoice, geminiKeyStructuredMessages, "stream",
		"function_call_mode", geminiKeyTemperature, "top_p", "top_k", "stop_sequences", "max_tokens":
		return true
	default:
		return false
	}
}

func (p *GeminiProvider) processFunctionCall(functionCall map[string]any, finalText *strings.Builder) {
	functionCallText := p.formatFunctionCall(functionCall)
	finalText.WriteString(functionCallText)
}

func (p *GeminiProvider) formatFunctionCall(functionCall map[string]any) string {
	name, ok := functionCall["name"].(string)
	if !ok {
		name = ""
	}
	args := functionCall["args"]

	// Convert args to JSON string if it's not already
	var argsJSON string
	if argsStr, ok := args.(string); ok {
		argsJSON = argsStr
	} else {
		argsBytes, err := json.Marshal(args)
		if err != nil {
			argsJSON = "{}"
		} else {
			argsJSON = string(argsBytes)
		}
	}

	return fmt.Sprintf(`{"function_call": {"name": %q, "arguments": %s}}`, name, argsJSON)
}

// SupportsStreaming returns true as all Google Gemini models support streaming.
func (p *GeminiProvider) SupportsStreaming() bool {
	return true
}

// SupportsStructuredResponse returns true for Google Gemini models that support structured output.
// Supported models: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite,
// gemini-2.0-pro, gemini-2.0-flash-lite, gemini-2.0-flash
func (p *GeminiProvider) SupportsStructuredResponse() bool {
	switch p.model {
	case "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
		"gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.0-flash":
		return true
	default:
		return false
	}
}

// SupportsFunctionCalling returns true for Google Gemini models that support function calling.
// Supported models: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite,
// gemini-2.0-pro, gemini-2.0-flash-lite, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
func (p *GeminiProvider) SupportsFunctionCalling() bool {
	switch p.model {
	case "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
		"gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.0-flash",
		"gemini-1.5-pro", "gemini-1.5-flash":
		return true
	default:
		return false
	}
}

// Response types for Gemini API
//
//nolint:tagliatelle // These types are specific to the Gemini API response structure
type geminiResponse struct {
	UsageMetadata *geminiUsage      `json:"usageMetadata"`
	Candidates    []geminiCandidate `json:"candidates"`
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

//nolint:tagliatelle // These types are specific to the Gemini API response structure
type geminiPart struct {
	FunctionCall map[string]any `json:"functionCall,omitempty"`
	Text         string         `json:"text,omitempty"`
}

//nolint:tagliatelle // These types are specific to the Gemini API response structure
type geminiUsage struct {
	PromptTokenCount        int64 `json:"promptTokenCount"`
	CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
	TotalTokenCount         int64 `json:"totalTokenCount"`
	CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
}
