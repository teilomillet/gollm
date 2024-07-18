// llm/groq.go
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type GroqLLM struct {
	BaseLLM
	apiKey string
	model  string
	// Groq-specific options
	FunctionCall      interface{}
	Functions         interface{}
	LogitBias         map[string]int
	Logprobs          *bool
	ParallelToolCalls *bool
	ResponseFormat    interface{}
	Seed              *int
	StreamOptions     interface{}
	ToolChoice        interface{}
	Tools             interface{}
	TopLogprobs       *int
}

func NewGroqLLM(apiKey, model string) *GroqLLM {
	return &GroqLLM{
		BaseLLM: BaseLLM{
			Options: BaseOptions{
				Temperature: new(float64),
				MaxTokens:   new(int),
				// Initialize other fields...
			},
		},
		apiKey: apiKey,
		model:  model,
	}
}

func (g *GroqLLM) SetProviderOption(opt ProviderOption) error {
	return opt.apply(g)
}

func (g *GroqLLM) Generate(ctx context.Context, prompt string) (string, error) {
	url := "https://api.groq.com/openai/v1/chat/completions"

	requestBody := map[string]interface{}{
		"messages": []LLMMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		"model": g.model,
	}

	// Add common options
	if g.Options.Temperature != nil {
		requestBody["temperature"] = *g.Options.Temperature
	}
	if g.Options.MaxTokens != nil {
		requestBody["max_tokens"] = *g.Options.MaxTokens
	}
	if g.Options.FrequencyPenalty != nil {
		requestBody["frequency_penalty"] = *g.Options.FrequencyPenalty
	}
	if g.Options.PresencePenalty != nil {
		requestBody["presence_penalty"] = *g.Options.PresencePenalty
	}
	if g.Options.TopP != nil {
		requestBody["top_p"] = *g.Options.TopP
	}
	if g.Options.N != nil {
		requestBody["n"] = *g.Options.N
	}
	if g.Options.Stream != nil {
		requestBody["stream"] = *g.Options.Stream
	}
	if g.Options.Stop != nil {
		requestBody["stop"] = g.Options.Stop
	}
	if g.Options.User != nil {
		requestBody["user"] = *g.Options.User
	}

	// Add Groq-specific options
	if g.FunctionCall != nil {
		requestBody["function_call"] = g.FunctionCall
	}
	if g.Functions != nil {
		requestBody["functions"] = g.Functions
	}
	if g.LogitBias != nil {
		requestBody["logit_bias"] = g.LogitBias
	}
	if g.Logprobs != nil {
		requestBody["logprobs"] = *g.Logprobs
	}
	if g.ParallelToolCalls != nil {
		requestBody["parallel_tool_calls"] = *g.ParallelToolCalls
	}
	if g.ResponseFormat != nil {
		requestBody["response_format"] = g.ResponseFormat
	}
	if g.Seed != nil {
		requestBody["seed"] = *g.Seed
	}
	if g.StreamOptions != nil {
		requestBody["stream_options"] = g.StreamOptions
	}
	if g.ToolChoice != nil {
		requestBody["tool_choice"] = g.ToolChoice
	}
	if g.Tools != nil {
		requestBody["tools"] = g.Tools
	}
	if g.TopLogprobs != nil {
		requestBody["top_logprobs"] = *g.TopLogprobs
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", g.apiKey))
	req.Header.Set("Content-Type", "application/json")

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
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	err = json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("API returned empty response")
	}

	return response.Choices[0].Message.Content, nil
}

// Groq-specific option setters
type groqOption struct {
	f func(*GroqLLM) error
}

func (o groqOption) apply(llm LLM) error {
	g, ok := llm.(*GroqLLM)
	if !ok {
		return fmt.Errorf("option only applicable to GroqLLM")
	}
	return o.f(g)
}

func WithGroqFunctionCall(functionCall interface{}) ProviderOption {
	return groqOption{func(g *GroqLLM) error {
		g.FunctionCall = functionCall
		return nil
	}}
}

func WithGroqFunctions(functions interface{}) ProviderOption {
	return groqOption{func(g *GroqLLM) error {
		g.Functions = functions
		return nil
	}}
}

// Add more Groq-specific option setters here...
