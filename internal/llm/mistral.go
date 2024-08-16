package llm

import (
	"encoding/json"
	"fmt"
)

func RegisterMistralProvider(registry *ProviderRegistry) {
	registry.Register("mistral", NewMistralProvider)
}

// MistralProvider implements the Provider interface for Mistral-AI
type MistralProvider struct {
	apiKey string
	model  string
}

func NewMistralProvider(apiKey, model string) Provider {
	return &MistralProvider{
		apiKey: apiKey,
		model:  model,
	}
}

func (p *MistralProvider) Name() string {
	return "mistral"
}

func (p *MistralProvider) Endpoint() string {
	return "https://api.mistral.ai/v1/chat/completions"
}

func (p *MistralProvider) Headers() map[string]string {
	return map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}
}

func (p *MistralProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
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

func (p *MistralProvider) ParseResponse(body []byte) (string, error) {
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
