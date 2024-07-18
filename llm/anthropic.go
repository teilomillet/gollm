package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// AnthropicLLM is a concrete type that implements the LLM interface for Anthropic
type AnthropicLLM struct {
	BaseLLM
	apiKey string
	model  string
	// Anthropic-specific options
	System        *string
	Metadata      map[string]interface{}
	StopSequences []string
	TopK          *int
	TopP          *float64
	Tools         []AnthropicTool
	ToolChoice    *AnthropicToolChoice
}

type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type AnthropicToolChoice struct {
	Type string `json:"type"`
	Tool string `json:"tool,omitempty"`
}

// NewAnthropicLLM creates a new instance of AnthropicLLM
func NewAnthropicLLM(apiKey, model string) *AnthropicLLM {
	return &AnthropicLLM{
		BaseLLM: BaseLLM{
			Options: BaseOptions{
				Temperature: new(float64),
				MaxTokens:   new(int),
				TopP:        new(float64),
				Stream:      new(bool),
			},
		},
		apiKey: apiKey,
		model:  model,
	}
}

func (a *AnthropicLLM) SetProviderOption(opt ProviderOption) error {
	return opt.apply(a)
}

func (a *AnthropicLLM) Generate(ctx context.Context, prompt string) (string, error) {
	url := "https://api.anthropic.com/v1/messages"

	messages := []map[string]string{
		{"role": "user", "content": prompt},
	}

	requestBody := map[string]interface{}{
		"model":    a.model,
		"messages": messages,
	}

	// Add common options
	if a.Options.MaxTokens != nil {
		requestBody["max_tokens"] = *a.Options.MaxTokens
	}
	if a.Options.Temperature != nil {
		requestBody["temperature"] = *a.Options.Temperature
	}
	if a.Options.TopP != nil {
		requestBody["top_p"] = *a.Options.TopP
	}
	if a.Options.Stream != nil {
		requestBody["stream"] = *a.Options.Stream
	}

	// Add Anthropic-specific options
	if a.System != nil {
		requestBody["system"] = *a.System
	}
	if a.Metadata != nil {
		requestBody["metadata"] = a.Metadata
	}
	if a.StopSequences != nil {
		requestBody["stop_sequences"] = a.StopSequences
	}
	if a.TopK != nil {
		requestBody["top_k"] = *a.TopK
	}
	if a.Tools != nil {
		requestBody["tools"] = a.Tools
	}
	if a.ToolChoice != nil {
		requestBody["tool_choice"] = a.ToolChoice
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error: %s", body)
	}

	var response struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	err = json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Content) == 0 || response.Content[0].Text == "" {
		return "", fmt.Errorf("API returned empty response")
	}

	return response.Content[0].Text, nil
}

// Anthropic-specific option setters
type anthropicOption struct {
	f func(*AnthropicLLM) error
}

func (o anthropicOption) apply(llm LLM) error {
	anthropic, ok := llm.(*AnthropicLLM)
	if !ok {
		return fmt.Errorf("option only applicable to AnthropicLLM")
	}
	return o.f(anthropic)
}

func WithAnthropicSystem(system string) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.System = &system
		return nil
	}}
}

func WithAnthropicMetadata(metadata map[string]interface{}) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.Metadata = metadata
		return nil
	}}
}

func WithAnthropicStopSequences(stopSequences []string) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.StopSequences = stopSequences
		return nil
	}}
}

func WithAnthropicTopK(topK int) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.TopK = &topK
		return nil
	}}
}

func WithAnthropicTools(tools []AnthropicTool) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.Tools = tools
		return nil
	}}
}

func WithAnthropicToolChoice(toolChoice AnthropicToolChoice) ProviderOption {
	return anthropicOption{func(a *AnthropicLLM) error {
		a.ToolChoice = &toolChoice
		return nil
	}}
}
