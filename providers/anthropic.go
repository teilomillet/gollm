package providers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/teilomillet/gollm/config"
)

// AnthropicProvider implements the Provider interface for Anthropic
type AnthropicProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
	options      map[string]interface{}
}

func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string), // Initialize the map
		options:      make(map[string]interface{}),
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

	// Handle user message
	userMessage := map[string]interface{}{
		"role": "user",
		"content": []map[string]interface{}{
			{
				"type":          "text",
				"text":          prompt,
				"cache_control": map[string]string{"type": "ephemeral"},
			},
		},
	}
	requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), userMessage)

	// Add tools if provided
	if tools, ok := options["tools"].([]interface{}); ok && len(tools) > 0 {
		requestBody["tools"] = tools
	}

	// Add tool_choice if provided
	if toolChoice, ok := options["tool_choice"].(string); ok {
		requestBody["tool_choice"] = toolChoice
	}

	// Add other options
	for k, v := range p.options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" {
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
	var response struct {
		Content []struct {
			Type    string `json:"type"`
			Text    string `json:"text"`
			ToolUse *struct {
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"tool_use,omitempty"`
		} `json:"content"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	var finalResponse strings.Builder

	for _, content := range response.Content {
		switch content.Type {
		case "text":
			finalResponse.WriteString(content.Text)
		case "tool_use":
			if content.ToolUse != nil {
				functionCall, _ := json.Marshal(map[string]interface{}{
					"name":      content.ToolUse.Name,
					"arguments": string(content.ToolUse.Input),
				})
				finalResponse.WriteString(fmt.Sprintf("<function_call>%s</function_call>", functionCall))
			}
		}
	}

	return finalResponse.String(), nil
}

func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	var response struct {
		Content []struct {
			Type    string `json:"type"`
			ToolUse *struct {
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"tool_use,omitempty"`
		} `json:"content"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	for _, content := range response.Content {
		if content.Type == "tool_use" && content.ToolUse != nil {
			return json.Marshal(map[string]interface{}{
				"name":      content.ToolUse.Name,
				"arguments": json.RawMessage(content.ToolUse.Input),
			})
		}
	}

	return nil, nil // No function call
}

func (p *AnthropicProvider) SetExtraHeaders(headers map[string]string) {
	p.extraHeaders = headers
}
