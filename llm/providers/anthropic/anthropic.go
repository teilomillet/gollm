package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/teilomillet/goal/llm"
	"io"
	"strings"
)

func init() {
	llm.RegisterProvider("anthropic", NewAnthropicProvider)
}

// AnthropicProvider implements the llm.Provider interface for Anthropic
type AnthropicProvider struct {
	apiKey string
	model  string
}

func NewAnthropicProvider(apiKey, model string) llm.Provider {
	return &AnthropicProvider{
		apiKey: apiKey,
		model:  model,
	}
}

func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

func (p *AnthropicProvider) Endpoint() string {
	return "https://api.anthropic.com/v1/messages"
}

func (p *AnthropicProvider) Headers() map[string]string {
	return map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
	}
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

func (p *AnthropicProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}

	err := json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Content) == 0 || response.Content[0].Text == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Content[0].Text, nil
}

func (p *AnthropicProvider) ParseStreamResponse(body io.Reader) (<-chan string, <-chan error) {
	textChan := make(chan string)
	errChan := make(chan error, 1)

	go func() {
		defer close(textChan)
		defer close(errChan)

		scanner := bufio.NewScanner(body)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					return
				}

				var streamResponse struct {
					Type  string `json:"type"`
					Delta struct {
						Text string `json:"text"`
					} `json:"delta"`
				}

				err := json.Unmarshal([]byte(data), &streamResponse)
				if err != nil {
					errChan <- fmt.Errorf("error unmarshaling stream data: %w", err)
					return
				}

				if streamResponse.Type == "content_block_delta" {
					textChan <- streamResponse.Delta.Text
				}
			}
		}

		if err := scanner.Err(); err != nil {
			errChan <- fmt.Errorf("error reading stream: %w", err)
		}
	}()

	return textChan, errChan
}

func init() {
	llm.RegisterProvider("anthropic", NewAnthropicProvider)
}
