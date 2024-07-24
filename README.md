# goal - Go Abstract Language Model Interface

`goal` is a Go package designed to simplify and streamline interactions with various Language Model (LLM) providers. It's built for AI engineers and developers who want a unified, flexible, and powerful interface to work with multiple LLM APIs.

## Key Features

- **Unified API**: Work with multiple LLM providers (OpenAI, Anthropic, Groq) through a single, consistent interface.
- **Easy Provider Switching**: Seamlessly switch between different LLM providers or models with minimal code changes.
- **Flexible Configuration**: Configure your LLM interactions via environment variables, code, or command-line flags.
- **High-Level AI Functions**: Utilize pre-built functions for common AI tasks like question-answering, summarization, and chain-of-thought reasoning.
- **Advanced Prompt Engineering**: Create sophisticated prompts with directives, context, and examples.
- **Provider Comparison**: Easily compare responses from multiple LLM providers for the same prompt.
- **JSON Output Validation**: Validate and ensure the structure of JSON outputs from LLMs.
- **Extensible Architecture**: Add new LLM providers with minimal effort.
- **CLI Tool**: Use `goal` directly from the command line for quick experiments and workflows.

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
	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-4o-mini"),
		goal.SetMaxTokens(100),
		goal.SetAPIKey("your-api-key-here"),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	prompt := goal.NewPrompt("Tell me a short joke about programming.")
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Response: %s\n", response)
}
```

For more advanced usage, including research and content refinement, check out the examples directory.

## Advanced Usage

### Comparing Models

The `CompareModels` function allows you to easily compare responses from different LLM providers or models:

```go
type JokeResponse struct {
	Setup    string `json:"setup"`
	Punchline string `json:"punchline"`
}

func validateJoke(joke JokeResponse) error {
	if joke.Setup == "" || joke.Punchline == "" {
		return fmt.Errorf("invalid joke structure")
	}
	return nil
}

configs := []*goal.Config{
	{Provider: "openai", Model: "gpt-4o-mini", APIKey: "your-openai-api-key"},
	{Provider: "anthropic", Model: "claude-3-5-sonnet-20240620	", APIKey: "your-anthropic-api-key"},
}

prompt := "Tell me a joke about programming. Respond in JSON format with 'setup' and 'punchline' fields."

results, err := goal.CompareModels(context.Background(), prompt, validateJoke, configs...)
if err != nil {
	log.Fatalf("Error comparing models: %v", err)
}

fmt.Println(goal.AnalyzeComparisonResults(results))
```

This example compares responses from OpenAI and Anthropic models, ensuring that each response is a valid joke with a setup and punchline.

### JSON Output Validation

`goal` now supports automatic validation of JSON outputs from LLMs. This is particularly useful when you expect a specific structure in the LLM's response:

```go
type AnalysisResult struct {
	Topic       string   `json:"topic"`
	Pros        []string `json:"pros"`
	Cons        []string `json:"cons"`
	Conclusion  string   `json:"conclusion"`
}

func validateAnalysis(analysis AnalysisResult) error {
	if analysis.Topic == "" || len(analysis.Pros) == 0 || len(analysis.Cons) == 0 || analysis.Conclusion == "" {
		return fmt.Errorf("invalid analysis structure")
	}
	return nil
}

prompt := goal.NewPrompt("Analyze the pros and cons of remote work.",
	goal.WithOutput("Respond in JSON format with 'topic', 'pros', 'cons', and 'conclusion' fields."),
)

response, err := llm.Generate(ctx, prompt, goal.WithJSONSchemaValidation())
if err != nil {
	log.Fatalf("Failed to generate valid analysis: %v", err)
}

var result AnalysisResult
if err := json.Unmarshal([]byte(response), &result); err != nil {
	log.Fatalf("Failed to parse response: %v", err)
}

if err := validateAnalysis(result); err != nil {
	log.Fatalf("Invalid analysis: %v", err)
}

fmt.Printf("Analysis: %+v\n", result)
```

This example demonstrates how to use JSON schema validation to ensure that the LLM's response matches the expected structure.

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
- JSON output validation


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
