# goal - Go Abstract Language Model Interface

`goal` is a Go package that provides a simple, unified interface for interacting with various Language Model (LLM) providers. It abstracts away the differences between different LLM APIs, allowing you to easily switch between providers or use multiple providers in your application.

## Features

- Unified interface for multiple LLM providers (Anthropic, OpenAI, etc.)
- Flexible and powerful prompt creation and management
- High-level functions for common tasks (QuestionAnswer, Summarize, etc.)
- Easy configuration management
- Support for advanced prompt engineering techniques

## Installation

To install the `goal` package, use `go get`:

```bash
go get github.com/teilomillet/goal
```

## Quick Start

Here's a simple example to get you started:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	// Create a new LLM client
	llm, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	// Create a prompt
	prompt := goal.NewPrompt("Explain the concept of recursion").
		Directive("Use a simple example").
		Output("Explanation:")

	// Generate a response
	ctx := context.Background()
	response, _, err := llm.Generate(ctx, prompt.String())
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}
```

## Advanced Prompt Usage

The `Prompt` struct in `goal` provides a powerful way to create sophisticated prompts. Here are some of the key features:

### Adding Directives

Directives guide the LLM on how to approach the task:

```go
prompt := goal.NewPrompt("Analyze the impact of AI on healthcare").
	Directive("Consider both positive and negative impacts").
	Directive("Provide specific examples where possible")
```

### Specifying Output Format

You can specify the desired output format:

```go
prompt := goal.NewPrompt("List the top 5 programming languages of 2023").
	Output("Numbered list:")
```

### Adding Context

Provide additional context to inform the LLM's response:

```go
prompt := goal.NewPrompt("Summarize the main points").
	Context("The following text is from a research paper on climate change:").
	Input("...") // Your input text here
```

### Setting Maximum Length

Limit the length of the LLM's response:

```go
prompt := goal.NewPrompt("Describe the process of photosynthesis").
	MaxLength(100) // Limit to approximately 100 words
```

### Including Examples

You can include examples to guide the LLM's output format or style:

```go
prompt := goal.NewPrompt("Generate a creative name for a tech startup").
	Examples("path/to/examples.txt", 3, "random")
```

Note: The `Examples` method reads examples from a file. Make sure to prepare this file beforehand.

### Combining Features

You can combine these features to create highly specific prompts:

```go
prompt := goal.NewPrompt("Analyze the future of electric vehicles").
	Directive("Consider technological, economic, and environmental factors").
	Directive("Provide both optimistic and pessimistic scenarios").
	Context("Recent advancements in battery technology have led to increased range and decreased charging times for electric vehicles.").
	Output("Analysis:").
	MaxLength(200)
```

## Using Prompts with LLM

Once you've created a prompt, you can use it with the LLM client:

```go
response, _, err := llm.Generate(ctx, prompt.String())
if err != nil {
	log.Fatalf("Failed to generate response: %v", err)
}
fmt.Printf("Response: %s\n", response)
```

## Configuration

Create a YAML configuration file in `~/.goal/configs/`:

```yaml
provider: anthropic
model: claude-3-opus-20240229
temperature: 0.7
max_tokens: 100
log_level: info
```

Then, use it when creating an LLM client:

```go
llm, err := goal.NewLLM("path/to/config.yaml")
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
