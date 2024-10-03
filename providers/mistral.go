package providers

import (
	"encoding/json"
	"fmt"
)

// RegisterMistralProvider registers the Mistral provider with the provider registry.
func RegisterMistralProvider(registry *ProviderRegistry) {
	registry.Register("mistral", NewMistralProvider)
}

// MistralProvider implements the Provider interface for Mistral-AI
type MistralProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
}

// NewMistralProvider creates a new Mistral provider.
func NewMistralProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &MistralProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
	}
}

// Name returns the name of the provider.
func (p *MistralProvider) Name() string {
	return "mistral"
}

// Endpoint returns the API endpoint for the provider.
func (p *MistralProvider) Endpoint() string {
	return "https://api.mistral.ai/v1/chat/completions"
}

// Headers returns the headers for the API request.
func (p *MistralProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

// PrepareRequest prepares the request body for the API call.
func (p *MistralProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
	}

	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// ParseResponse parses the API response and returns the content.
func (p *MistralProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty response from API")
	}

	// Combine content and tool calls
	var finalResponse string
	finalResponse += response.Choices[0].Message.Content

	for _, toolCall := range response.Choices[0].Message.ToolCalls {
		finalResponse += fmt.Sprintf("<function_call>{\"name\": \"%s\", \"arguments\": %s}</function_call>", toolCall.Function.Name, toolCall.Function.Arguments)
	}

	return finalResponse, nil
}

// HandleFunctionCalls extracts and returns the function call details.
func (p *MistralProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	var response struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || len(response.Choices[0].Message.ToolCalls) == 0 {
		return nil, nil // No function call
	}

	return json.Marshal(response.Choices[0].Message.ToolCalls[0].Function)
}

// SetExtraHeaders sets additional headers for the API request.
func (p *MistralProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}
