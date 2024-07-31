# gollm - Go Abstract Language Model Interface

`gollm` is a Go package designed to simplify and streamline interactions with various Language Model (LLM) providers. It's built for AI engineers and developers who want a unified, flexible, and powerful interface to work with multiple LLM APIs.

## Key Features

- **Unified API for Multiple LLM Providers:** gollm supports various providers, including OpenAI, Anthropic and Groq, allowing you to switch between models like GPT-4, GPT-4o-mini, and Claude and llama-3.1 seamlessly.
- **Easy Provider and Model Switching:** Configure your preferred provider and model with simple function calls, making it effortless to experiment with different LLMs.
- **Flexible Configuration Options:** Set up your LLM interactions using environment variables, code-based configuration, or configuration files to suit your project's needs.
- **Advanced Prompt Engineering:** Create sophisticated prompts with context, directives, output specifications, and examples to guide the LLM's responses effectively.
- **Structured Output and Validation:** Generate and validate JSON schemas for structured outputs, ensuring consistency and reliability in LLM responses.
- **Provider Comparison Tools:** Easily compare responses from different LLM providers and models for the same prompt, helping you choose the best option for your use case.
- **High-Level AI Functions:** Utilize pre-built functions like ChainOfThought for complex reasoning tasks.
- **Robust Error Handling and Retries:** Built-in retry mechanisms with customizable delays to handle API rate limits and transient errors gracefully.
- **Extensible Architecture:** Designed to be easily extended to support new LLM providers and features.
Debug Logging: Configurable debug levels to help you troubleshoot and optimize your LLM interactions.


## Real-World Applications

gollm is versatile enough to handle a wide range of AI-powered tasks, including:

- **Content Creation Workflows:** Generate research summaries, article ideas, and refined paragraphs for writing projects.
- **Complex Reasoning Tasks:** Use the ChainOfThought function to break down and analyze complex problems step-by-step.
- **Structured Data Generation:** Create and validate complex data structures with customizable JSON schemas.
- **Model Performance Analysis:** Compare different models' performance for specific tasks to optimize your AI pipeline.

## Installation

```bash
go get github.com/teilomillet/gollm
```

## Quick Start

### Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetMaxTokens(100),
		gollm.SetAPIKey("your-api-key-here"),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	prompt := gollm.NewPrompt("Tell me a short joke about programming.")
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

configs := []*gollm.Config{
	{Provider: "openai", Model: "gpt-4o-mini", APIKey: "your-openai-api-key"},
	{Provider: "anthropic", Model: "claude-3-5-sonnet-20240620	", APIKey: "your-anthropic-api-key"},
	{Provider: "groq", Model: "llama-3.1-70b-versatile", APIKey: "your-anthropic-api-key"},

}

prompt := "Tell me a joke about programming. Respond in JSON format with 'setup' and 'punchline' fields."

results, err := gollm.CompareModels(context.Background(), prompt, validateJoke, configs...)
if err != nil {
	log.Fatalf("Error comparing models: %v", err)
}

fmt.Println(gollm.AnalyzeComparisonResults(results))
```

This example compares responses from OpenAI and Anthropic models, ensuring that each response is a valid joke with a setup and punchline.

### JSON Output Validation

`gollm` now supports automatic validation of JSON outputs from LLMs. This is particularly useful when you expect a specific structure in the LLM's response:

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

prompt := gollm.NewPrompt("Analyze the pros and cons of remote work.",
	gollm.WithOutput("Respond in JSON format with 'topic', 'pros', 'cons', and 'conclusion' fields."),
)

response, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
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

Find other examples that demonstrates how to use JSON schema validation to ensure that the LLM's response matches the expected structure in the examples/.

## Streaming Support

Currently, `gollm` does not support streaming responses. This feature is on our roadmap for future development.

## Project Status

`gollm` is actively maintained and under continuous development. We welcome contributions and feedback from the community.

## Examples and Tutorials

Check out our [examples directory](https://github.com/teilomillet/gollm/tree/main/examples) for more usage examples, including:

- Basic usage
- Different prompt types
- Comparing providers
- Advanced prompt templates
- Combining multiple `gollm` features
- JSON output validation


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
