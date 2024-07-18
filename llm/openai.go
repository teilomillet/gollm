package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type OpenAILLM struct {
	BaseLLM
	apiKey string
	model  string
	// OpenAI-specific options
	FrequencyPenalty *float64
	LogitBias        map[string]int
	Logprobs         *bool
	TopLogprobs      *int
	PresencePenalty  *float64
	ResponseFormat   *ResponseFormat
	Seed             *int
	Stop             []string
	Tools            []Tool
	ToolChoice       interface{}
	User             *string
}

type ResponseFormat struct {
	Type string `json:"type"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// NewOpenAILLM creates a new instance of OpenAILLM
func NewOpenAILLM(apiKey, model string) *OpenAILLM {
	return &OpenAILLM{
		BaseLLM: BaseLLM{
			Options: BaseOptions{},
		},
		apiKey: apiKey,
		model:  model,
	}
}

func (o *OpenAILLM) SetProviderOption(opt ProviderOption) error {
	return opt.apply(o)
}

func (o *OpenAILLM) Generate(ctx context.Context, prompt string) (string, error) {
	url := "https://api.openai.com/v1/chat/completions"

	messages := []map[string]string{
		{"role": "user", "content": prompt},
	}

	requestBody := map[string]interface{}{
		"model":    o.model,
		"messages": messages,
	}

	// Only add options that have been explicitly set
	if o.Options.Temperature != nil {
		requestBody["temperature"] = *o.Options.Temperature
	}
	if o.Options.MaxTokens != nil {
		requestBody["max_tokens"] = *o.Options.MaxTokens
	}
	if o.Options.TopP != nil {
		requestBody["top_p"] = *o.Options.TopP
	}
	if o.Options.N != nil {
		requestBody["n"] = *o.Options.N
	}
	if o.Options.Stream != nil {
		requestBody["stream"] = *o.Options.Stream
	}

	// Add OpenAI-specific options only if they're set
	if o.FrequencyPenalty != nil {
		requestBody["frequency_penalty"] = *o.FrequencyPenalty
	}
	if len(o.LogitBias) > 0 {
		requestBody["logit_bias"] = o.LogitBias
	}
	if o.Logprobs != nil {
		requestBody["logprobs"] = *o.Logprobs
	}
	if o.TopLogprobs != nil {
		requestBody["top_logprobs"] = *o.TopLogprobs
	}
	if o.PresencePenalty != nil {
		requestBody["presence_penalty"] = *o.PresencePenalty
	}
	if o.ResponseFormat != nil {
		requestBody["response_format"] = o.ResponseFormat
	}
	if o.Seed != nil {
		requestBody["seed"] = *o.Seed
	}
	if len(o.Stop) > 0 {
		requestBody["stop"] = o.Stop
	}
	if len(o.Tools) > 0 {
		requestBody["tools"] = o.Tools
	}
	if o.ToolChoice != nil {
		requestBody["tool_choice"] = o.ToolChoice
	}
	if o.User != nil {
		requestBody["user"] = *o.User
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
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", o.apiKey))

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

// OpenAI-specific option setters
type openAIOption struct {
	f func(*OpenAILLM) error
}

func (o openAIOption) apply(llm LLM) error {
	openai, ok := llm.(*OpenAILLM)
	if !ok {
		return fmt.Errorf("option only applicable to OpenAILLM")
	}
	return o.f(openai)
}

func WithOpenAIFrequencyPenalty(penalty float64) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.FrequencyPenalty = &penalty
		return nil
	}}
}

func WithOpenAILogitBias(logitBias map[string]int) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.LogitBias = logitBias
		return nil
	}}
}

func WithOpenAILogprobs(logprobs bool) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.Logprobs = &logprobs
		return nil
	}}
}

func WithOpenAITopLogprobs(topLogprobs int) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.TopLogprobs = &topLogprobs
		return nil
	}}
}

func WithOpenAIPresencePenalty(penalty float64) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.PresencePenalty = &penalty
		return nil
	}}
}

func WithOpenAIResponseFormat(format ResponseFormat) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.ResponseFormat = &format
		return nil
	}}
}

func WithOpenAISeed(seed int) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.Seed = &seed
		return nil
	}}
}

func WithOpenAIStop(stop []string) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.Stop = stop
		return nil
	}}
}

func WithOpenAITools(tools []Tool) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.Tools = tools
		return nil
	}}
}

func WithOpenAIToolChoice(toolChoice interface{}) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.ToolChoice = toolChoice
		return nil
	}}
}

func WithOpenAIUser(user string) ProviderOption {
	return openAIOption{func(o *OpenAILLM) error {
		o.User = &user
		return nil
	}}
}
