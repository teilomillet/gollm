package providers

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/teilomillet/gollm/utils"
)

type AzureOpenAIProvider struct {
	Provider     // Embed the OpenAIProvider
	extraHeaders map[string]string
	logger       utils.Logger
	endpoint     string
	model        string
	apiKey       string
	apiVersion   string
}

func NewAzureOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	openAI := NewOpenAIProvider(apiKey, model, extraHeaders)
	return &AzureOpenAIProvider{
		Provider:     openAI,
		apiKey:       apiKey,
		extraHeaders: extraHeaders,
		model:        model,
	}

}
func (p *AzureOpenAIProvider) Endpoint() string {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	if endpoint == "" {
		panic("AZURE_OPENAI_ENDPOINT is not set")
	}
	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		panic("AZURE_OPENAI_API_VERSION is not set")
	}
	url := "%s/openai/deployments/%s/chat/completions?api-version=%s"
	return fmt.Sprintf(url, endpoint, p.model, apiVersion)
}

func (p *AzureOpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type": "application/json",
	}
	if p.apiKey != "" {
		headers["api-key"] = p.apiKey
	}
	slog.Info("Headers prepared", "headers", headers)

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	//p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}
