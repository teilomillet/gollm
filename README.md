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
	"github.com/teilomillet/goal/llm"
)

func main() {
	// Create a new LLM instance
	model, err := llm.NewLLM("anthropic", "claude-3-opus-20240229")
	if err != nil {
		panic(err)
	}

	// Generate text
	response, err := model.Generate(context.Background(), "Tell me a joke about programming")
	if err != nil {
		panic(err)
	}

	fmt.Println(response)
}
```

## Package Structure

### llm.go
This file defines the core interfaces and structures for the LLM abstraction:
- `LLM`: The main interface for interacting with language models.
- `Provider`: Interface for different LLM providers.
- `LLMImpl`: A generic implementation of the `LLM` interface.

### provider_registry.go
Manages the registration and retrieval of LLM providers:
- `RegisterProvider`: Adds a new provider to the registry.
- `GetProvider`: Returns a provider instance based on the name.

### factory.go
Contains factory functions for creating LLM instances:
- `getProvider`: Returns the appropriate `Provider` based on the provider name and model.

### anthropic.go
Implements the `Provider` interface for Anthropic's Claude models:
- `AnthropicProvider`: Struct implementing the `Provider` interface.
- `NewAnthropicProvider`: Constructor for creating a new Anthropic provider.

### logging.go
Sets up logging for the package using `zap`:
- `Logger`: Global logger for the LLM package.
- `SetLogLevel`: Function to set the log level.

## Contributing

We welcome contributions to the `goal` package! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
