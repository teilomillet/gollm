# goal - Go Abstract Language Model Interface

`goal` is a Go package designed to simplify and streamline interactions with various Language Model (LLM) providers. It's built for AI engineers and developers who want a unified, flexible, and powerful interface to work with multiple LLM APIs.

## Key Features

- **Unified API**: Work with multiple LLM providers (OpenAI, Anthropic, Groq) through a single, consistent interface.
- **Easy Provider Switching**: Seamlessly switch between different LLM providers or models with minimal code changes.
- **Flexible Configuration**: Configure your LLM interactions via environment variables, code, or command-line flags.
- **High-Level AI Functions**: Utilize pre-built functions for common AI tasks like question-answering, summarization, and chain-of-thought reasoning.
- **Advanced Prompt Engineering**: Create sophisticated prompts with directives, context, and examples.
- **Provider Comparison**: Easily compare responses from multiple LLM providers for the same prompt.
- **Extensible Architecture**: Add new LLM providers with minimal effort.
- **CLI Tool**: Use `goal` directly from the command line for quick experiments and workflows.
- **Robust Error Handling**: Detailed error types for better error management and debugging.

## Installation

```bash
go get github.com/teilomillet/goal
```

## Quick Start

### Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	cfg := goal.NewConfigBuilder().
		SetProvider("openai").
		SetModel("gpt-3.5-turbo").
		SetMaxTokens(100).
		SetAPIKey("your-api-key-here").
		Build()

	llm, err := goal.NewLLM(cfg)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	prompt := goal.NewPrompt("Tell me a short joke about programming.",
		goal.WithMaxLength(50),
	)
	response, _, err := llm.Generate(ctx, prompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Response: %s\n", response)
}
```

## Advanced Usage

### Prompt Types

`goal` supports various prompt types to cater to different use cases:

1. **Basic Prompt**: Simple text input.
2. **Prompt with Directives**: Guide the LLM's response.
3. **Prompt with Context**: Provide background information.
4. **Prompt with Max Length**: Limit response length.
5. **Prompt with Examples**: Provide example inputs/outputs.

Example:

```go
directivePrompt := goal.NewPrompt("Explain the concept of recursion",
	goal.WithDirectives("Use a simple example to illustrate", "Keep it concise"),
	goal.WithOutput("Explanation of recursion:"),
)
```

### Comparing Providers

Easily compare responses from different LLM providers:

```go
providers := []string{"openai", "anthropic"}
llms := make(map[string]goal.LLM)

for _, provider := range providers {
	cfg := goal.NewConfigBuilder().
		SetProvider(provider).
		SetMaxTokens(100).
		SetAPIKey("your-api-key-here").
		Build()

	llm, _ := goal.NewLLM(cfg)
	llms[provider] = llm
}

question := "What is the capital of France?"
for provider, llm := range llms {
	answer, _ := goal.QuestionAnswer(ctx, llm, question)
	fmt.Printf("%s answer: %s\n", provider, answer)
}
```

### Advanced Prompt Templates

Create reusable prompt templates for complex tasks:

```go
advancedPromptTemplate := goal.NewPromptTemplate(
	"AdvancedAnalysis",
	"Analyze a topic from multiple perspectives",
	"Analyze the following topic from multiple perspectives: {{.Topic}}",
	goal.WithPromptOptions(
		goal.WithDirectives(
			"Consider technological, economic, social, and ethical implications",
			"Provide at least one potential positive and one potential negative outcome for each perspective",
			"Conclude with a balanced summary of no more than 3 sentences",
		),
		goal.WithOutput("Multi-perspective Analysis:"),
		goal.WithMaxLength(300),
	),
)

prompt, _ := advancedPromptTemplate.Execute(map[string]interface{}{
	"Topic": "The widespread adoption of artificial intelligence in healthcare",
})

analysis, _, _ := llmClient.Generate(ctx, prompt.String())
fmt.Printf("Analysis:\n%s\n", analysis)
```

## Error Handling

`goal` provides detailed error types to help you handle different failure scenarios:

- `ErrorTypeProvider`: Issues with the LLM provider
- `ErrorTypeRequest`: Problems preparing or sending the request
- `ErrorTypeResponse`: Issues parsing or processing the API response
- `ErrorTypeAPI`: Errors returned by the LLM provider's API
- `ErrorTypeRateLimit`: Rate limiting errors
- `ErrorTypeAuthentication`: Authentication-related errors
- `ErrorTypeInvalidInput`: Invalid input errors

Example error handling:

```go
response, _, err := llm.Generate(ctx, prompt.String())
if err != nil {
	if llmErr, ok := err.(*goal.LLMError); ok {
		switch llmErr.Type {
		case goal.ErrorTypeRateLimit:
			// Handle rate limiting
		case goal.ErrorTypeAuthentication:
			// Handle authentication issues
		default:
			log.Printf("An error occurred: %v", llmErr)
		}
	} else {
		log.Printf("An unexpected error occurred: %v", err)
	}
	return
}
```

## Performance Considerations

While `goal` adds a thin abstraction layer, its impact on performance is minimal. The main performance factors will be the responsiveness of the chosen LLM provider and the complexity of your prompts.

## Streaming Support

Currently, `goal` does not support streaming responses. This feature is on our roadmap for future development.

## Project Status

`goal` is actively maintained and under continuous development. We welcome contributions and feedback from the community.

## Examples and Tutorials

Check out our [examples directory](https://github.com/teilomillet/goal/tree/main/examples) for more usage examples, including:

- Basic usage
- Different prompt types
- Comparing providers
- Advanced prompt templates
- Combining multiple `goal` features


## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more information on how to get started.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
