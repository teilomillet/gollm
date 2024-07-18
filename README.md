# goal - Go Abstract Language Model Interface

`goal` is a Go package that provides a simple, unified interface for interacting with various Language Model (LLM) providers. It abstracts away the differences between different LLM APIs, allowing you to easily switch between providers or use multiple providers in your application.

## Installation

To install the `goal` package, use `go get`:

```
go get github.com/teilomillet/goal
```

## Usage

Here's a quick example of how to use `goal`:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/goal/llm"
	"go.uber.org/zap"
)

func main() {
	// Set log level
	llm.SetLogLevel(zap.InfoLevel)

	// Ensure the appropriate API key is set in the environment
	provider := "anthropic"
	apiKeyEnv := fmt.Sprintf("%s_API_KEY", provider)
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		llm.Logger.Fatal("API key not set", zap.String("env_var", apiKeyEnv))
	}

	// Create LLM provider
	model := "claude-3-opus-20240229"
	llmProvider, err := llm.GetProvider(provider, apiKey, model)
	if err != nil {
		llm.Logger.Fatal("Error creating LLM provider", zap.Error(err))
	}

	// Create LLM client
	llmClient := llm.NewLLM(llmProvider)

	// Set options
	llmClient.SetOption("temperature", 0.7)
	llmClient.SetOption("max_tokens", 100)

	// Generate text
	ctx := context.Background()
	prompt := "Explain the concept of recursion in programming."
	response, err := llmClient.Generate(ctx, prompt)
	if err != nil {
		llm.Logger.Fatal("Error generating text", zap.Error(err))
	}

	fmt.Println("Response:", response)
}
```

# goal - Go Abstract Language Model Interface

...

## Package Structure

The `goal` package is structured as follows:

- `llm/`: This directory contains the core package files.
  - `provider.go`: Defines the `Provider` interface.
  - `provider_registry.go`: Contains the provider registry and `RegisterProvider` function.
  - `llm.go`: Implements the main LLM client.
  - Individual provider implementations (e.g., `anthropic.go`, `openai.go`).
- `templates/`: Contains templates for adding new providers.
- `examples/`: Contains example usage of the package.

## Adding New Providers

To add support for a new LLM provider, you can use the template provided in `templates/new_provider_template.go`. Here's how to do it:

1. Copy the `new_provider_template.go` file and rename it to match your new provider (e.g., `myprovider.go`).
2. Place the new file in the `llm/` directory.
3. Replace all occurrences of `NewProviderName` with your provider's name (e.g., `MyProvider`).
4. Update the `Name()` method to return the lowercase name of your provider.
5. Modify the `Endpoint()` method to return the correct API endpoint for your provider.
6. Update the `Headers()` method to include any necessary headers for API requests.
7. Adjust the `PrepareRequest()` method to create the correct request body for your provider's API.
8. Modify the `ParseResponse()` method to correctly extract the generated text from your provider's API response.
9. Update the `ParseStreamResponse()` method if your provider supports streaming responses.
10. In the `init()` function, register your new provider with the correct name.

Note: The `Provider` interface and `RegisterProvider` function are already defined in the package. You don't need to include their definitions in your new provider file.

For a detailed example of how to implement a provider, look at the existing provider implementations in the `llm` directory.

...

## Command-line Interface

The `goal` package also includes a command-line interface for quick interactions with LLMs. You can find it in `cmd/goal/main.go`. To use it, build the binary and run:

```
goal [flags] <provider> <model> <prompt>
```

or

```
goal [flags] <model> <prompt>
```

For more details, run `goal -help`.

## Contributing

We welcome contributions to the `goal` package! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
