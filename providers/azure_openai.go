// Package providers implements LLM provider interfaces and implementations.
package providers

// NewAzureOpenAIProvider creates a new Azure OpenAI provider instance.
// Azure OpenAI requires the endpoint to be configured via ExtraHeaders with
// the "azure_endpoint" key, since endpoints vary per deployment.
//
// Parameters:
//   - apiKey: Azure OpenAI API key for authentication
//   - model: The deployment name to use
//   - extraHeaders: Must include "azure_endpoint" with the full endpoint URL
//
// Example endpoint format:
//
//	https://{resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2023-05-15
//
// Returns:
//   - A configured Azure OpenAI Provider instance
func NewAzureOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "azure-openai", extraHeaders)
}
