package providers

import (
	"encoding/json"
	"fmt"
	"strings"
)

// AnthropicProvider implements the Provider interface for Anthropic
type AnthropicProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
}

func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
	}
}

func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

func (p *AnthropicProvider) Endpoint() string {
	return "https://api.anthropic.com/v1/messages"
}

func (p *AnthropicProvider) SupportsJSONSchema() bool {
	return false
}

func (p *AnthropicProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
	}
	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

func (p *AnthropicProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
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
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	// Initialize the final response
	var finalResponse string

	for _, content := range response.Content {
		if content.Type == "text" {
			finalResponse += content.Text
		} else if content.Type == "tool_call" {
			// Extract the function call from the text
			start := strings.Index(content.Text, "<function_call>")
			end := strings.Index(content.Text, "</function_call>")
			if start != -1 && end != -1 {
				functionCall := content.Text[start+len("<function_call>") : end]
				var function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}
				if err := json.Unmarshal([]byte(functionCall), &function); err != nil {
					return "", fmt.Errorf("failed to parse function call: %w", err)
				}
				// Append the function call to the final response
				finalResponse += fmt.Sprintf("<function_call>%s</function_call>", functionCall)
			}
		}
	}

	return finalResponse, nil
}

func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	var response struct {
		Content []struct {
			Text      string `json:"text"`
			ToolCalls []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"content"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Content) == 0 || len(response.Content[0].ToolCalls) == 0 {
		return nil, nil // No function call
	}

	// Extract the first function call
	firstFunctionCall := response.Content[0].ToolCalls[0].Function

	// Parse the arguments
	var argsMap map[string]interface{}
	if err := json.Unmarshal(firstFunctionCall.Arguments, &argsMap); err != nil {
		return nil, fmt.Errorf("failed to parse arguments: %w", err)
	}

	// Return the parsed function call
	return json.Marshal(map[string]interface{}{
		"name":      firstFunctionCall.Name,
		"arguments": argsMap,
	})
}

func (p *AnthropicProvider) SetExtraHeaders(headers map[string]string) {
	p.extraHeaders = headers
}
