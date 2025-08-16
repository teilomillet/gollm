# Building Custom Providers

While the Generic Provider system handles most LLM APIs, you may occasionally need to create a custom provider. This guide explains when and how to create custom providers for Gollm.

## When to Create a Custom Provider

Create a custom provider when:

1. An LLM API has a **significantly different** request/response format than existing types
2. You need **special handling** for requests or responses that can't be achieved with configuration
3. You want to integrate with a **local LLM** or custom implementation

## Options for Creating Providers

Gollm offers a spectrum of approaches for provider creation, from simplest to most complex:

1. **Use a built-in provider** - For standard providers like OpenAI, Anthropic, etc.
2. **Register a GenericProvider configuration** - For OpenAI/Anthropic-compatible APIs
3. **Extend an existing provider** - For providers that are similar to existing ones
4. **Implement a custom provider** - For completely unique APIs

### Approach 1: Register a GenericProvider (Recommended)

For most API-compatible providers, this is all you need:

```go
import "github.com/weave-labs/gollm/providers"

// Define provider configuration
config := providers.ProviderConfig{
    Name:       "my-provider",
    Type:       providers.TypeOpenAI,
    Endpoint:   "https://api.myprovider.com/v1/completions",
    AuthHeader: "Authorization",
    AuthPrefix: "Bearer ",
    RequiredHeaders: map[string]string{
        "Content-Type": "application/json",
    },
    SupportsSchema:    true,
    SupportsStreaming: true,
}

// Register it
providers.RegisterGenericProvider("my-provider", config)
```

### Approach 2: Extend an Existing Provider

If a provider is similar to an existing one but needs minor customization:

```go
package providers

// MyProvider extends the OpenAI provider
type MyProvider struct {
    OpenAIProvider
}

// Create a new instance
func NewMyProvider(apiKey, model string, extraHeaders map[string]string) Provider {
    provider := &MyProvider{
        OpenAIProvider: *NewOpenAIProvider(apiKey, model, extraHeaders).(*OpenAIProvider),
    }
    return provider
}

// Override any methods that need customization
func (p *MyProvider) Name() string {
    return "my-provider"
}

func (p *MyProvider) Endpoint() string {
    return "https://custom-endpoint.com/api"
}

// Register the provider
func init() {
    GetDefaultRegistry().Register("my-provider", NewMyProvider)
}
```

### Approach 3: Implement a Custom Provider

For completely unique APIs:

```go
package providers

import (
    "encoding/json"
    "fmt"
    "github.com/weave-labs/gollm/config"
    "github.com/weave-labs/gollm/utils"
)

// Define your provider struct
type CustomProvider struct {
    apiKey       string
    model        string
    extraHeaders map[string]string
    options      map[string]any
    logger       utils.Logger
}

// Implement the constructor
func NewCustomProvider(apiKey, model string, extraHeaders map[string]string) Provider {
    if extraHeaders == nil {
        extraHeaders = make(map[string]string)
    }
    
    return &CustomProvider{
        apiKey:       apiKey,
        model:        model,
        extraHeaders: extraHeaders,
        options:      make(map[string]any),
        logger:       utils.NewLogger(utils.LogLevelInfo),
    }
}

// Implement all required interface methods
func (p *CustomProvider) Name() string {
    return "custom"
}

func (p *CustomProvider) Endpoint() string {
    return "https://api.custom-provider.com/generate"
}

func (p *CustomProvider) Headers() map[string]string {
    headers := map[string]string{
        "Content-Type": "application/json",
    }
    
    // Add auth header if needed
    if p.apiKey != "" {
        headers["X-API-Key"] = p.apiKey
    }
    
    // Add extra headers
    for k, v := range p.extraHeaders {
        headers[k] = v
    }
    
    return headers
}

func (p *CustomProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
    // Create the custom request format
    request := map[string]any{
        "text":       prompt,
        "model_name": p.model,
    }
    
    // Add options from the provider
    for k, v := range p.options {
        request[k] = v
    }
    
    // Override with passed options
    for k, v := range options {
        request[k] = v
    }
    
    return json.Marshal(request)
}

// Implement the remaining methods...
// ParseResponse, PrepareRequestWithSchema, SetExtraHeaders, etc.

// Register the provider
func init() {
    GetDefaultRegistry().Register("custom", NewCustomProvider)
}
```

## Tips for Custom Provider Implementation

1. **Start by extending**: When possible, extend an existing provider rather than starting from scratch
2. **Proper error handling**: Ensure all API errors are properly captured and returned
3. **Comprehensive logging**: Add detailed logging to help with debugging
4. **Test thoroughly**: Create unit tests for your provider implementation
5. **Consistent behavior**: Ensure your provider behaves consistently with other providers

## Complete Example: Custom Local Provider

Here's a more complete example of a custom provider for a local LLM server:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    
    "github.com/weave-labs/gollm"
    "github.com/weave-labs/gollm/config"
    "github.com/weave-labs/gollm/providers"
    "github.com/weave-labs/gollm/utils"
)

// LocalProvider is a custom provider for a local LLM server
type LocalProvider struct {
    serverURL    string
    extraHeaders map[string]string
    options      map[string]any
    logger       utils.Logger
}

// NewLocalProvider creates a new provider for a local LLM
func NewLocalProvider(apiKey, model string, extraHeaders map[string]string) providers.Provider {
    // In this case, we use apiKey parameter as the server URL
    serverURL := apiKey
    if serverURL == "" {
        serverURL = "http://localhost:8080/generate" // Default
    }
    
    return &LocalProvider{
        serverURL:    serverURL,
        extraHeaders: extraHeaders,
        options:      map[string]any{"model": model},
        logger:       utils.NewLogger(utils.LogLevelInfo),
    }
}

// Name returns the provider identifier
func (p *LocalProvider) Name() string {
    return "local"
}

// Endpoint returns the API endpoint
func (p *LocalProvider) Endpoint() string {
    return p.serverURL
}

// Headers returns HTTP headers for requests
func (p *LocalProvider) Headers() map[string]string {
    headers := map[string]string{
        "Content-Type": "application/json",
    }
    
    for k, v := range p.extraHeaders {
        headers[k] = v
    }
    
    return headers
}

// PrepareRequest creates the request body
func (p *LocalProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
    request := map[string]any{
        "prompt": prompt,
    }
    
    // Add all options
    for k, v := range p.options {
        request[k] = v
    }
    
    for k, v := range options {
        request[k] = v
    }
    
    return json.Marshal(request)
}

// ParseResponse extracts text from response
func (p *LocalProvider) ParseResponse(body []byte) (string, error) {
    var response struct {
        Text  string `json:"text"`
        Error string `json:"error"`
    }
    
    if err := json.Unmarshal(body, &response); err != nil {
        return "", fmt.Errorf("failed to parse response: %v", err)
    }
    
    if response.Error != "" {
        return "", fmt.Errorf("API error: %s", response.Error)
    }
    
    return response.Text, nil
}

// Implement other required methods...

// Register the provider
func init() {
    providers.GetDefaultRegistry().Register("local", NewLocalProvider)
}

func main() {
    // Now use your custom provider
    llm, err := gollm.NewLLM(
        config.SetProvider("local"),
        config.SetAPIKey("http://localhost:8080/generate"), // URL as the API key
        config.SetModel("llama2"),
    )
    
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    ctx := context.Background()
    prompt := gollm.NewPrompt("Explain quantum computing")
    
    response, err := llm.Generate(ctx, prompt)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Println(response)
}
```

## Testing Custom Providers

It's important to test your custom providers:

```go
package providers_test

import (
    "testing"
    
    "github.com/weave-labs/gollm/providers"
)

func TestCustomProvider(t *testing.T) {
    // Get provider from registry
    provider, err := providers.GetDefaultRegistry().Get("custom", "fake-key", "model", nil)
    if err != nil {
        t.Fatalf("Failed to get provider: %v", err)
    }
    
    // Test basic properties
    if provider.Name() != "custom" {
        t.Errorf("Expected name 'custom', got '%s'", provider.Name())
    }
    
    // Test request preparation
    body, err := provider.PrepareRequest("Test prompt", nil)
    if err != nil {
        t.Fatalf("Failed to prepare request: %v", err)
    }
    
    // Verify request format
    // ... additional tests
}
```

For more details on the Provider interface, refer to the [Provider System Documentation](provider_system.md). 