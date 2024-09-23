package llm

import (
	"encoding/json"
	"fmt"
)

// GroqProvider implements the Provider interface for Groq
type GroqProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
}

func NewGroqProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &GroqProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
	}
}

func (p *GroqProvider) Name() string {
	return "groq"
}

func (p *GroqProvider) Endpoint() string {
	return "https://api.groq.com/openai/v1/chat/completions"
}

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

func (p *GroqProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
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

func (p *GroqProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	err := json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Message.Content, nil
}

func (p *GroqProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	var response struct {
		Choices []struct {
			Message struct {
				FunctionCall *struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function_call"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.FunctionCall == nil {
		return nil, nil // No function call
	}

	return json.Marshal(response.Choices[0].Message.FunctionCall)
}

func (p *GroqProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}
