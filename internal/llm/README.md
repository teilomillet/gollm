# internal/llm Package

This package provides the core functionality for the `goal` Language Model (LLM) interface. It's designed to be flexible, extensible, and provider-agnostic.

## Package Structure

The package consists of several key files:

- `config.go`: Handles configuration loading and management
- `llm.go`: Defines the main LLM interface and implementation
- `logging.go`: Sets up logging for the package
- `prompt.go`: Implements the Prompt structure for building prompts
- `provider_registry.go`: Manages the registry of LLM providers
- `anthropic.go`: Implements the Anthropic provider
- `groq.go`: Implements the Groq provider
- `compare.go`: Provides functionality to compare responses from different providers
- `errors.go`: Defines custom error types and error handling functions
- `factory.go`: Contains factory functions for creating provider instances

## Key Components

### Config

The `Config` struct in `config.go` defines the configuration for an LLM instance. It includes fields for:

- Provider
- Model
- Temperature
- MaxTokens
- LogLevel

The package provides functions to load configurations from YAML files and manage default configurations.

### LLM Interface

The `LLM` interface in `llm.go` defines the core functionality:

```go
type LLM interface {
    Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
    SetOption(key string, value interface{})
}
```

The `LLMImpl` struct provides a generic implementation of this interface.

### Provider Interface

The `Provider` interface defines the contract for implementing new LLM providers:

```go
type Provider interface {
    Name() string
    Endpoint() string
    Headers() map[string]string
    PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
    ParseResponse(body []byte) (string, error)
}
```

## Comparison Functionality

The `compare.go` file provides functionality to compare responses from different providers:

- `CompareProviders` function to generate responses from multiple providers concurrently
- `PrintComparisonResults` function to display the comparison results

## Error Handling

The `errors.go` file defines custom error types and error handling functions:

- `LLMError` struct for package-specific errors
- `ErrorType` enum for categorizing errors
- `HandleError` function for consistent error logging and handling

## Factory Functions

The `factory.go` file contains the `getProvider` function, which creates provider instances based on the provider name and model. It also handles API key retrieval from environment variables.

## Configuration

Configurations are loaded from YAML files. The package supports loading a single configuration file or multiple files from a directory. If no configuration is provided, it falls back to a default configuration.

## Logging

The package uses `uber-go/zap` for logging. The log level can be configured in the configuration file or set programmatically using `SetLogLevel()`.

## Provider System

The package uses a provider registry system to manage different LLM providers. New providers can be registered using the `RegisterProvider()` function:

```go
RegisterProvider("provider_name", ProviderConstructorFunction)
```

Providers are instantiated using the `GetProvider()` function.

## Extending the Package

To add a new provider:

1. Create a new file (e.g., `newprovider.go`) implementing the `Provider` interface.
2. Register the provider in an `init()` function:

```go
func init() {
    RegisterProvider("newprovider", NewNewProvider)
}

func NewNewProvider(apiKey, model string) Provider {
    return &NewProviderImpl{
        APIKey: apiKey,
        Model:  model,
    }
}
```

3. Implement the required methods of the `Provider` interface.

When adding new features or modifying existing ones, ensure to:

- Update configurations if necessary
- Add appropriate logging
- Use the existing error types or create new ones if needed
- Update tests to cover new functionality

Remember to maintain backwards compatibility when making changes to existing interfaces or structures.

## Comparison and Benchmarking

The `CompareProviders` function in `compare.go` can be used to benchmark different providers or models. This is useful for evaluating the performance and quality of responses across different LLMs.

## Error Handling Best Practices

When working with the package, use the custom `LLMError` type for error creation and the `HandleError` function for error handling. This ensures consistent error management across the package.
