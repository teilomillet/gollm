package providers

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// AnthropicProvider implements the Provider interface for Anthropic
type AnthropicProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
	options      map[string]interface{}
	logger       utils.Logger
}

func (p *AnthropicProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]interface{}),
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

func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

func (p *AnthropicProvider) Endpoint() string {
	return "https://api.anthropic.com/v1/messages"
}

func (p *AnthropicProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

func (p *AnthropicProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

func (p *AnthropicProvider) SupportsJSONSchema() bool {
	return false
}

func (p *AnthropicProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "prompt-caching-2024-07-31",
	}
	return headers
}

func (p *AnthropicProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": p.options["max_tokens"],
		"system":     []map[string]interface{}{},
		"messages":   []map[string]interface{}{},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"]; ok {
		if sp, ok := systemPrompt.(string); ok && sp != "" {
			parts := splitSystemPrompt(sp, 3)
			for i, part := range parts {
				systemMessage := map[string]interface{}{
					"type": "text",
					"text": part,
				}
				if i > 0 {
					systemMessage["cache_control"] = map[string]string{"type": "ephemeral"}
				}
				requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
			}
		}
	}

	// Handle user message with potential caching
	userMessage := map[string]interface{}{
		"role": "user",
		"content": []map[string]interface{}{
			{
				"type": "text",
				"text": prompt,
			},
		},
	}

	// Add cache_control only if caching is enabled
	if caching, ok := options["enable_caching"].(bool); ok && caching {
		userMessage["content"].([]map[string]interface{})[0]["cache_control"] = map[string]string{"type": "ephemeral"}
	}

	requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), userMessage)

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		anthropicTools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			anthropicTools[i] = map[string]interface{}{
				"name":         tool.Function.Name,
				"description":  tool.Function.Description,
				"input_schema": tool.Function.Parameters,
			}
		}
		requestBody["tools"] = anthropicTools
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		requestBody["tool_choice"] = map[string]interface{}{
			"type": toolChoice,
		}
	}

	// Add other options
	for k, v := range options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" && k != "enable_caching" {
			requestBody[k] = v
		}
	}

	return json.Marshal(requestBody)
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
	for i := 0; i < n; i++ {
		end := currentIndex + paragraphsPerPart
		if i < extraParagraphs {
			end++
		}
		result[i] = strings.Join(paragraphs[currentIndex:end], "\n\n")
		currentIndex = end
	}

	return result
}

func (p *AnthropicProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type":   "json_schema",
			"schema": schema,
		},
	}

	// Add any additional options
	for k, v := range options {
		requestBody[k] = v
	}

	// Add strict option if provided
	if strict, ok := options["strict"].(bool); ok && strict {
		requestBody["response_format"].(map[string]interface{})["strict"] = true
	}

	return json.Marshal(requestBody)
}

func (p *AnthropicProvider) ParseResponse(body []byte) (string, error) {
	p.logger.Debug("Raw API response: %s", string(body))

	var response struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type  string          `json:"type"`
			Text  string          `json:"text,omitempty"`
			ID    string          `json:"id,omitempty"`
			Name  string          `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		StopReason string  `json:"stop_reason"`
		StopSeq    *string `json:"stop_sequence"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}
	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	var finalResponse strings.Builder
	var functionCalls []string

	for _, content := range response.Content {
		switch content.Type {
		case "text":
			finalResponse.WriteString(content.Text)
			p.logger.Debug("Text content: %s", content.Text)
		case "tool_use":
			functionCall, err := json.Marshal(map[string]interface{}{
				"name":      content.Name,
				"arguments": content.Input,
			})
			if err != nil {
				return "", fmt.Errorf("error marshaling function call: %w", err)
			}
			functionCalls = append(functionCalls, string(functionCall))
			p.logger.Debug("Function call detected: %s", string(functionCall))
		}
	}

	// If there are function calls, append them to the response
	if len(functionCalls) > 0 {
		for _, call := range functionCalls {
			finalResponse.WriteString("\n<function_call>")
			finalResponse.WriteString(call)
			finalResponse.WriteString("</function_call>")
			p.logger.Debug("Appending function call to response")
		}
	}

	p.logger.Debug("Final response: %s", finalResponse.String())
	return finalResponse.String(), nil
}

func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls from response")
	var response string
	err := json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %w", err)
	}

	p.logger.Debug("Response body: %s", response)

	// Extract function calls from the response
	var functionCalls []map[string]interface{}
	functionCallRegex := regexp.MustCompile(`<function_call>(.*?)</function_call>`)
	matches := functionCallRegex.FindAllStringSubmatch(response, -1)

	for _, match := range matches {
		if len(match) > 1 {
			var functionCall map[string]interface{}
			err := json.Unmarshal([]byte(match[1]), &functionCall)
			if err != nil {
				return nil, fmt.Errorf("error unmarshaling function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Extracted function call: %v", functionCall)
		}
	}

	if len(functionCalls) == 0 {
		p.logger.Debug("No function calls found in the response")
		return nil, nil // No function calls
	}

	p.logger.Debug("Function calls to handle: %v", functionCalls)
	return json.Marshal(functionCalls)
}

func (p *AnthropicProvider) SetExtraHeaders(headers map[string]string) {
	p.extraHeaders = headers
}
