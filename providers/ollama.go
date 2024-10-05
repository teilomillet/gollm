package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
	"io"
	"net/http"
	"strings"
)

type OllamaProvider struct {
	model        string
	endpoint     string
	logger       utils.Logger
	extraHeaders map[string]string
	options      map[string]interface{}
}

func NewOllamaProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OllamaProvider{
		model:        model,
		endpoint:     "http://localhost:11434", // Default endpoint
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

func (p *OllamaProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

func (p *OllamaProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	if config.OllamaEndpoint != "" {
		p.SetEndpoint(config.OllamaEndpoint)
	}
	p.SetOption("top_p", config.TopP)
	p.SetOption("min_p", config.MinP)
	p.SetOption("repeat_penalty", config.RepeatPenalty)
	p.SetOption("repeat_last_n", config.RepeatLastN)
	p.SetOption("mirostat", config.Mirostat)
	p.SetOption("mirostat_eta", config.MirostatEta)
	p.SetOption("mirostat_tau", config.MirostatTau)
	p.SetOption("tfs_z", config.TfsZ)
}

func (p *OllamaProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

func (p *OllamaProvider) SupportsJSONSchema() bool {
	return false
}

// Implement LLM interface methods

func (p *OllamaProvider) Generate(ctx context.Context, prompt string) (string, string, error) {
	reqBody, err := p.PrepareRequest(prompt, p.options)
	if err != nil {
		return "", "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", "", err
	}

	for k, v := range p.Headers() {
		req.Header.Set(k, v)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}

	result, err := p.ParseResponse(body)
	if err != nil {
		return "", "", err
	}

	return result, prompt, nil
}

func (p *OllamaProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	if p.logger != nil {
		p.logger.Debug("Setting option for Ollama", "key", key, "value", value)
	}
}

// Add the missing SetDebugLevel method
func (p *OllamaProvider) SetDebugLevel(level utils.LogLevel) {
	if p.logger != nil {
		p.logger.SetLevel(level)
	}
}

func (p *OllamaProvider) SetEndpoint(endpoint string) {
	p.endpoint = endpoint
}

// Existing Provider interface methods

func (p *OllamaProvider) Endpoint() string {
	return p.endpoint + "/api/generate"
}

func (p *OllamaProvider) Name() string {
	return "ollama"
}

func (p *OllamaProvider) Headers() map[string]string {
	return map[string]string{
		"Content-Type": "application/json",
	}
}

func (p *OllamaProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  p.model,
		"prompt": prompt,
	}

	for k, v := range options {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

func (p *OllamaProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	// Ollama doesn't support JSON schema validation natively
	// We'll just use the regular PrepareRequest method
	return p.PrepareRequest(prompt, options)
}

func (p *OllamaProvider) ParseResponse(body []byte) (string, error) {
	var fullResponse strings.Builder
	decoder := json.NewDecoder(bytes.NewReader(body))

	for decoder.More() {
		var response struct {
			Model    string `json:"model"`
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := decoder.Decode(&response); err != nil {
			return "", fmt.Errorf("error parsing Ollama response: %w", err)
		}
		fullResponse.WriteString(response.Response)
		if response.Done {
			break
		}
	}

	return fullResponse.String(), nil
}

func (p *OllamaProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	// Ollama doesn't support function calling natively, so we return nil
	return nil, nil
}
