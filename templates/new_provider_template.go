package llm

import (
	"encoding/json"
	"fmt"
	"io"
)

// Provider interface definition
// This should typically be in a separate file (e.g., provider.go) and imported
type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
	ParseStreamResponse(body io.Reader) (<-chan string, <-chan error)
}

// RegisterProvider function
// This should typically be in a separate file (e.g., provider_registry.go) and imported
var providerRegistry = make(map[string]func(apiKey, model string) Provider)

func RegisterProvider(name string, constructor func(apiKey, model string) Provider) {
	providerRegistry[name] = constructor
}

// NewProviderName represents the provider for NewProvider's language models.
// Replace "NewProvider" with the actual name of the provider you're implementing.
type NewProviderName struct {
	apiKey string
	model  string
}

// NewNewProviderName creates a new instance of NewProviderName.
// This function will be called by the factory when creating a new provider.
func NewNewProviderName(apiKey, model string) Provider {
	return &NewProviderName{
		apiKey: apiKey,
		model:  model,
	}
}

// Name returns the name of the provider.
func (p *NewProviderName) Name() string {
	return "newprovider" // Replace with the lowercase name of your provider
}

// Endpoint returns the API endpoint for the provider.
func (p *NewProviderName) Endpoint() string {
	return "https://api.newprovider.com/v1/completions" // Replace with the actual API endpoint
}

// Headers returns the necessary headers for API requests.
func (p *NewProviderName) Headers() map[string]string {
	return map[string]string{
		"Authorization": "Bearer " + p.apiKey,
		"Content-Type":  "application/json",
		// Add any other headers required by the provider
	}
}

// PrepareRequest creates the request body for the API call.
func (p *NewProviderName) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Create a map with the necessary parameters for the API request
	requestBody := map[string]interface{}{
		"model":      p.model,
		"prompt":     prompt,
		"max_tokens": options["max_tokens"],
		// Add any other parameters required by the provider
	}

	// Add any additional options passed to the function
	for k, v := range options {
		requestBody[k] = v
	}

	// Convert the map to JSON
	return json.Marshal(requestBody)
}

// ParseResponse extracts the generated text from the API response.
func (p *NewProviderName) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}

	err := json.Unmarshal(body, &response)
	if err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Text == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Text, nil
}

// ParseStreamResponse handles streaming responses from the API.
func (p *NewProviderName) ParseStreamResponse(body io.Reader) (<-chan string, <-chan error) {
	textChan := make(chan string)
	errChan := make(chan error, 1)

	go func() {
		defer close(textChan)
		defer close(errChan)

		decoder := json.NewDecoder(body)
		for decoder.More() {
			var streamResponse struct {
				Choices []struct {
					Text string `json:"text"`
				} `json:"choices"`
			}

			if err := decoder.Decode(&streamResponse); err != nil {
				errChan <- fmt.Errorf("error decoding stream response: %w", err)
				return
			}

			if len(streamResponse.Choices) > 0 {
				textChan <- streamResponse.Choices[0].Text
			}
		}
	}()

	return textChan, errChan
}

func init() {
	// Register the new provider in the provider registry
	RegisterProvider("newprovider", NewNewProviderName)
}

