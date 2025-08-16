// Package providers implements LLM provider interfaces and their implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/weave-labs/gollm/internal/models"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

const (
	geminiKeyTools        = "tools"
	geminiKeyMaxTokens    = "max_tokens"
	geminiKeySystemPrompt = "system_prompt"
)

// GeminiProvider implements the Provider interface for Google's Gemini API (Generative Language API).
// It supports chat completions with system instructions, function/tool calling with JSON schemas,
// streaming via Server-Sent Events (SSE), and usage reporting.
type GeminiProvider struct {
	apiKey       string            // API key or Bearer token for authentication
	model        string            // Model name (e.g., "gemini-2.0-pro", "gemini-1.5-pro")
	extraHeaders map[string]string // Additional HTTP headers
	options      map[string]any    // Provider-specific options (temperature, max_tokens, etc.)
	logger       logging.Logger    // Logger instance for debugging
}

// NewGeminiProvider creates a new Google Gemini API provider instance.
//
// Parameters:
//   - apiKey: API key or OAuth Bearer token for Google Generative Language API
//   - model: The model to use (e.g., "gemini-2.0-pro").
//     This can be just the model ID; the provider will format the full resource name.
//   - extraHeaders: Additional HTTP headers to include in requests (can be nil)
//
// Returns:
//   - A configured GeminiProvider instance
func NewGeminiProvider(apiKey, model string, extraHeaders map[string]string) *GeminiProvider {
	provider := &GeminiProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo), // default logger
	}
	// Copy any provided extra headers
	for k, v := range extraHeaders {
		provider.extraHeaders[k] = v
	}

	return provider
}

// Name returns the provider's identifier string.
func (p *GeminiProvider) Name() string {
	return "google"
}

// Endpoint returns the full API endpoint URL for Google Gemini generateContent calls.
// It inserts the model name into the endpoint path. For example:
// "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent".
func (p *GeminiProvider) Endpoint() string {
	// Ensure model path is properly formatted
	modelName := p.model
	if !strings.HasPrefix(modelName, "models/") {
		modelName = "models/" + modelName
	}
	return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s:generateContent", modelName)
}

// Headers returns the HTTP headers required for Google API requests, including authorization.
func (p *GeminiProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}
	// Include any additional headers that were set
	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *GeminiProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SetLogger allows configuring a custom logger for the Google provider.
func (p *GeminiProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Google provider (e.g., "temperature", "max_tokens").
// Supported options include temperature, max_tokens, top_p, top_k, stop_sequences, etc.
func (p *GeminiProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions applies global config defaults (temperature, max tokens, etc.) to this provider.
func (p *GeminiProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
}

// SupportsJSONSchema indicates that the Google provider supports structured output control via JSON schema.
func (p *GeminiProvider) SupportsStructuredResponse() bool {
	return true
}

// SupportsStreaming indicates that streaming responses (SSE) are supported by the Google provider.
func (p *GeminiProvider) SupportsStreaming() bool {
	return true
}

// addSystemInstruction adds system instruction to the request if provided
func (p *GeminiProvider) addSystemInstruction(requestBody map[string]any, options map[string]any) {
	if sys, ok := options[geminiKeySystemPrompt].(string); ok && sys != "" {
		requestBody["systemInstruction"] = map[string]any{
			"parts": []map[string]string{
				{"text": sys},
			},
		}
	}
}

// addUserContent adds user message content to the request
func (p *GeminiProvider) addUserContent(requestBody map[string]any, prompt string) {
	userContent := map[string]any{
		"role": "user",
		"parts": []map[string]string{
			{"text": prompt},
		},
	}
	if contents, ok := requestBody["contents"].([]map[string]any); ok {
		requestBody["contents"] = append(contents, userContent)
	}
}

// addToolsToGeminiRequest adds tool definitions to the request
func (p *GeminiProvider) addToolsToGeminiRequest(requestBody map[string]any, options map[string]any) {
	tools, ok := options[geminiKeyTools].([]models.Tool)
	if !ok || len(tools) == 0 {
		return
	}

	// Build functionDeclarations for each provided tool (function)
	funcDecls := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		decl := map[string]any{
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
		}
		funcDecls = append(funcDecls, decl)
	}
	requestBody[geminiKeyTools] = []map[string]any{
		{"functionDeclarations": funcDecls},
	}

	// Optionally, set function calling mode if specified (e.g., "NONE", "AUTO", "ANY")
	if mode, ok := options["function_call_mode"].(string); ok && mode != "" {
		requestBody["toolConfig"] = map[string]any{
			"functionCallingConfig": map[string]any{
				"mode": mode,
			},
		}
	}
}

// buildGenerationConfig builds the generation configuration from provider options
func (p *GeminiProvider) buildGenerationConfig() map[string]any {
	genConfig := make(map[string]any)

	// If the max_tokens option is set, map it to maxOutputTokens
	if maxTokens, ok := p.options["max_tokens"].(int); ok && maxTokens > 0 {
		genConfig["maxOutputTokens"] = maxTokens
	}
	// Temperature
	if temp, ok := p.options["temperature"].(float64); ok {
		genConfig["temperature"] = temp
	}
	// Top-p
	if topP, ok := p.options["top_p"].(float64); ok {
		genConfig["topP"] = topP
	}
	// Top-k
	if topK, ok := p.options["top_k"].(int); ok {
		genConfig["topK"] = topK
	}
	// Stop sequences
	if stops, ok := p.options["stop_sequences"].([]string); ok && len(stops) > 0 {
		genConfig["stopSequences"] = stops
	}

	return genConfig
}

// mergeRemainingOptions merges unhandled options into the request body
func (p *GeminiProvider) mergeRemainingOptions(requestBody map[string]any, options map[string]any) {
	for k, v := range options {
		if p.isHandledOption(k) {
			continue
		}
		requestBody[k] = v
	}
}

// isHandledOption checks if an option has already been handled
func (p *GeminiProvider) isHandledOption(key string) bool {
	switch key {
	case geminiKeySystemPrompt, geminiKeyTools, "function_call_mode",
		geminiKeyMaxTokens, "temperature", "top_p", "top_k", "stop_sequences":
		return true
	default:
		return false
	}
}

// PrepareRequest builds the JSON request body for a single-turn completion or prompt.
// It includes the user prompt (as content with role "user"), optional system instruction, tools, and generation
// parameters.
func (p *GeminiProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
	// Base request structure
	requestBody := map[string]any{
		"model":    p.model,            // model ID (the API also requires it in path, set in Endpoint())
		"contents": []map[string]any{}, // conversation content list
	}

	// Add system instruction and user content
	p.addSystemInstruction(requestBody, options)
	p.addUserContent(requestBody, prompt)

	// Add tools if provided
	p.addToolsToGeminiRequest(requestBody, options)

	// Add generation config
	genConfig := p.buildGenerationConfig()
	if len(genConfig) > 0 {
		requestBody["generationConfig"] = genConfig
	}

	// Merge remaining options
	p.mergeRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// PrepareRequestWithMessages creates a request body from a sequence of structured messages (conversation history).
// It maps each MemoryMessage to a Gemini API Content, preserving roles and content. System prompts and tools are
// handled similarly to PrepareRequest.
func (p *GeminiProvider) PrepareRequestWithMessages(
	messages []models.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	requestBody := p.initializeGeminiRequestBody()

	// Add system instruction
	p.addSystemInstructionForMessages(requestBody, options)

	// Add tools and function calling
	p.addToolsForMessages(requestBody, options)

	// Convert and add messages
	p.addConvertedMessages(requestBody, messages)

	// Add generation config
	p.addGenerationConfigForMessages(requestBody)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// initializeGeminiRequestBody creates the base request structure for Gemini
func (p *GeminiProvider) initializeGeminiRequestBody() map[string]any {
	return map[string]any{
		"model":    p.model,
		"contents": []map[string]any{},
	}
}

// addSystemInstructionForMessages adds system instruction for message requests
func (p *GeminiProvider) addSystemInstructionForMessages(requestBody map[string]any, options map[string]any) {
	sys, ok := options[geminiKeySystemPrompt].(string)
	if !ok || sys == "" {
		return
	}

	requestBody["systemInstruction"] = map[string]any{
		"parts": []map[string]string{
			{"text": sys},
		},
	}
}

// addToolsForMessages adds tools and function calling configuration for messages
func (p *GeminiProvider) addToolsForMessages(requestBody map[string]any, options map[string]any) {
	tools, ok := options[geminiKeyTools].([]models.Tool)
	if !ok || len(tools) == 0 {
		return
	}

	funcDecls := p.buildFunctionDeclarations(tools)
	requestBody[geminiKeyTools] = []map[string]any{
		{"functionDeclarations": funcDecls},
	}

	p.addFunctionCallingMode(requestBody, options)
}

// buildFunctionDeclarations creates function declarations from tools
func (p *GeminiProvider) buildFunctionDeclarations(tools []models.Tool) []map[string]any {
	funcDecls := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		funcDecl := map[string]any{
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
		}
		funcDecls = append(funcDecls, funcDecl)
	}
	return funcDecls
}

// addFunctionCallingMode adds function calling mode configuration
func (p *GeminiProvider) addFunctionCallingMode(requestBody map[string]any, options map[string]any) {
	mode, ok := options["function_call_mode"].(string)
	if !ok || mode == "" {
		return
	}

	requestBody["toolConfig"] = map[string]any{
		"functionCallingConfig": map[string]any{
			"mode": mode,
		},
	}
}

// addConvertedMessages converts and adds MemoryMessage objects to request
func (p *GeminiProvider) addConvertedMessages(requestBody map[string]any, messages []models.MemoryMessage) {
	for _, msg := range messages {
		contentEntry := p.convertMessageToGeminiFormat(msg)
		if contentEntry == nil {
			continue
		}
		if contents, ok := requestBody["contents"].([]map[string]any); ok {
			requestBody["contents"] = append(contents, contentEntry)
		}
	}
}

// convertMessageToGeminiFormat converts a MemoryMessage to Gemini format
func (p *GeminiProvider) convertMessageToGeminiFormat(msg models.MemoryMessage) map[string]any {
	role := p.mapRoleToGemini(msg.Role)
	if role == "" {
		return nil // Skip unknown roles
	}

	contentParts := []map[string]any{
		{"text": msg.Content},
	}

	return map[string]any{
		"role":  role,
		"parts": contentParts,
	}
}

// mapRoleToGemini maps standard roles to Gemini-specific roles
func (p *GeminiProvider) mapRoleToGemini(role string) string {
	if role == "assistant" {
		return "model"
	}
	if role == "user" || role == "model" || role == "function" {
		return role
	}
	return "" // Unknown role
}

// addGenerationConfigForMessages adds generation config parameters for message requests
func (p *GeminiProvider) addGenerationConfigForMessages(requestBody map[string]any) {
	genConfig := p.buildGenerationConfig()
	if len(genConfig) > 0 {
		requestBody["generationConfig"] = genConfig
	}
}

// PrepareRequestWithSchema builds a request similar to PrepareRequest but enforces a JSON output schema.
// It uses the GenerationConfig.responseMimeType and responseSchema fields to instruct the model to return JSON adhering
// to the schema.
func (p *GeminiProvider) PrepareRequestWithSchema(prompt string, _ map[string]any, schema any) ([]byte, error) {
	// Base content with user prompt
	requestBody := map[string]any{
		"model": p.model,
		"contents": []map[string]any{
			{
				"role": "user",
				"parts": []map[string]string{
					{"text": prompt},
				},
			},
		},
	}
	// If a system prompt is provided, it is ignored here to avoid conflicting instructions
	// (We rely on the schema enforcement to structure the output)
	// Merge generation options
	genConfig := map[string]any{
		"responseMimeType": "application/json",
		"responseSchema":   schema, // JSON schema for desired output format
	}
	// Include other options (like max_tokens, temperature, etc.) in generationConfig
	if maxTokens, ok := p.options["max_tokens"].(int); ok && maxTokens > 0 {
		genConfig["maxOutputTokens"] = maxTokens
	}
	if temp, ok := p.options["temperature"].(float64); ok {
		genConfig["temperature"] = temp
	}
	if topP, ok := p.options["top_p"].(float64); ok {
		genConfig["topP"] = topP
	}
	if topK, ok := p.options["top_k"].(int); ok {
		genConfig["topK"] = topK
	}
	if stops, ok := p.options["stop_sequences"].([]string); ok && len(stops) > 0 {
		genConfig["stopSequences"] = stops
	}
	requestBody["generationConfig"] = genConfig

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseResponse parses the JSON response from a non-streaming GenerateContent request.
// It extracts the text content and any function call outputs, and populates the Usage statistics.
func (p *GeminiProvider) ParseResponse(body []byte) (*Response, error) {
	p.logger.Debug("Raw API response: %s", string(body))

	// Define structures to unmarshal relevant parts of the response
	var resp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text             *string         `json:"text,omitempty"`
					FunctionCall     *map[string]any `json:"functionCall,omitempty"`
					FunctionResponse *map[string]any `json:"functionResponse,omitempty"`
				} `json:"parts"`
			} `json:"content"`
			// finishReason and safetyRatings could be present, but not needed for core content parsing
		} `json:"candidates"`
		UsageMetadata *struct {
			PromptTokenCount        int64 `json:"promptTokenCount"`
			CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
			CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
			// totalTokenCount is also available but we can compute total from parts
		} `json:"usageMetadata,omitempty"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %w", err)
	}
	if len(resp.Candidates) == 0 {
		return nil, errors.New("no candidates in response")
	}

	// We consider only the first candidate (primary response)
	candidate := resp.Candidates[0]

	var finalText strings.Builder

	// Iterate through parts of the content
	for _, part := range candidate.Content.Parts {
		// If the part is text, append it to the final text (add space if needed)
		if part.Text != nil {
			text := *part.Text
			if finalText.Len() > 0 && text != "" {
				finalText.WriteString(" ")
			}
			finalText.WriteString(text)
		}
		// If the part is a function call, format it as a string and collect it
		if part.FunctionCall != nil {
			p.processFunctionCall(*part.FunctionCall, &finalText)
		}
		// If the part is a function response (function output fed back to model), we skip it for final content.
		if part.FunctionResponse != nil {
			// (Typically, functionResponse parts would appear only as input to next model turn, not in model's own
			// output.)
			continue
		}
	}

	// Create the Response object
	result := finalText.String()
	p.logger.Debug("Final parsed content: %s", result)
	response := &Response{
		Content: Text{Value: result},
		Usage:   nil,
	}

	// Extract usage stats if available
	if resp.UsageMetadata != nil {
		um := resp.UsageMetadata
		inputTokens := um.PromptTokenCount
		cachedInput := um.CachedContentTokenCount
		outputTokens := um.CandidatesTokenCount
		// Google API does not explicitly break out cached output tokens; assume 0
		cachedOutput := int64(0)
		response.Usage = NewUsage(inputTokens, cachedInput, outputTokens, cachedOutput)
	}
	return response, nil
}

// HandleFunctionCalls inspects the final response content for any function call strings and returns them as JSON.
// This helps isolate tool invocations from the text.
func (p *GeminiProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Extracting function calls from response content")
	content := string(body)
	calls, err := ExtractFunctionCalls(content)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}
	if len(calls) == 0 {
		return nil, nil
	}
	p.logger.Debug("Function calls found: %v", calls)
	data, err := json.Marshal(calls)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal function calls: %w", err)
	}
	return data, nil
}

// PrepareStreamRequest creates the request body for a streaming content request.
// It is similar to PrepareRequest, but (in usage) the endpoint called will be streamGenerateContent with SSE.
func (p *GeminiProvider) PrepareStreamRequest(prompt string, options map[string]any) ([]byte, error) {
	// We can reuse the same payload as PrepareRequest.
	// (The streaming vs non-streaming is determined by using the streaming endpoint and alt=sse.)
	return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse processes a single SSE data chunk from a streaming response.
// It returns a Response containing either a piece of text, a function call, or usage info, or io.EOF when the stream is
// done.
func (p *GeminiProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Trim whitespace
	data := bytes.TrimSpace(chunk)
	if len(data) == 0 {
		return nil, errors.New("empty chunk")
	}
	// OpenAI-style "[DONE]" check (not typically used by Google, but included for completeness)
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil, io.EOF
	}

	// Parse JSON chunk
	var resp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text             *string         `json:"text,omitempty"`
					FunctionCall     *map[string]any `json:"functionCall,omitempty"`
					FunctionResponse *map[string]any `json:"functionResponse,omitempty"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason,omitempty"`
		} `json:"candidates"`
		UsageMetadata *struct {
			PromptTokenCount        int64 `json:"promptTokenCount"`
			CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
			CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
		} `json:"usageMetadata,omitempty"`
	}

	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("malformed JSON chunk: %w", err)
	}

	// If usage metadata is present (possibly in final chunk), return a Response with usage info
	if resp.UsageMetadata != nil {
		um := resp.UsageMetadata
		usageResp := &Response{
			Usage: NewUsage(um.PromptTokenCount, um.CachedContentTokenCount, um.CandidatesTokenCount, 0),
		}
		return usageResp, nil
	}

	// If no candidates or parts, skip this chunk
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("skip chunk")
	}

	// We handle only the first candidate and first part for streaming incremental response
	part := resp.Candidates[0].Content.Parts[0]
	if part.Text != nil && *part.Text != "" {
		// Return the text token
		return &Response{Content: Text{Value: *part.Text}}, nil
	}
	if part.FunctionCall != nil {
		// Format function call as in ParseResponse
		formattedCall := p.formatFunctionCall(*part.FunctionCall)
		if formattedCall != "" {
			return &Response{Content: Text{Value: formattedCall}}, nil
		}
	}
	// Ignore functionResponse parts in stream
	if part.FunctionResponse != nil {
		return nil, errors.New("skip chunk")
	}

	return nil, errors.New("skip chunk")
}

// processFunctionCall formats a function call and adds it to the final text
func (p *GeminiProvider) processFunctionCall(functionCall map[string]any, finalText *strings.Builder) {
	nameVal, nameOK := functionCall["name"].(string)
	if !nameOK {
		return
	}

	argsVal, argsOK := functionCall["args"]
	var formattedCall string
	if argsOK {
		// Marshal the args object to JSON string
		argsJSON, err := json.Marshal(argsVal)
		if err != nil {
			formattedCall = nameVal + "()"
		} else {
			formattedCall = fmt.Sprintf("%s(%s)", nameVal, argsJSON)
		}
	} else {
		formattedCall = nameVal + "()"
	}

	// If there is pending text not yet added to output, add a newline before function call
	if finalText.Len() > 0 {
		finalText.WriteString("\n")
	}
	finalText.WriteString(formattedCall)
}

// formatFunctionCall formats a function call and returns the formatted string
func (p *GeminiProvider) formatFunctionCall(functionCall map[string]any) string {
	nameVal, nameOK := functionCall["name"].(string)
	if !nameOK {
		return ""
	}

	argsVal, argsOK := functionCall["args"]
	if argsOK {
		argsJSON, err := json.Marshal(argsVal)
		if err != nil {
			return nameVal + "()"
		}
		return fmt.Sprintf("%s(%s)", nameVal, argsJSON)
	}
	return nameVal + "()"
}
