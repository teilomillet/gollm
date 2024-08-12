# gollm - Go Large Language Model

![Gophers building a robot by Renee French](img/gopherrobot4s.jpg)

`gollm` is a Go package designed to help you build your own AI golems. Just as the mystical golem of legend was brought to life with sacred words, gollm empowers you to breathe life into your AI creations using the power of Large Language Models (LLMs). This package simplifies and streamlines interactions with various LLM providers, offering a unified, flexible, and powerful interface for AI engineers and developers to craft their own digital servants.

[Documentation](https://docs.gollm.co)

## Key Features

- **Unified API for Multiple LLM Providers:** Shape your golem's mind using various providers, including OpenAI, Anthropic, and Groq. Seamlessly switch between models like GPT-4, GPT-4o-mini, Claude, and llama-3.1.
- **Easy Provider and Model Switching:** Mold your golem's capabilities by configuring preferred providers and models with simple function calls.
- **Flexible Configuration Options:** Customize your golem's essence using environment variables, code-based configuration, or configuration files to suit your project's needs.
- **Advanced Prompt Engineering:** Craft sophisticated instructions to guide your golem's responses effectively.
- **PromptOptimizer:** Automatically refine and improve your prompts for better results, with support for custom metrics and different rating systems.
- **Memory Retention:** Maintain context across multiple interactions for more coherent conversations.
- **Structured Output and Validation:** Ensure your golem's outputs are consistent and reliable with JSON schema generation and validation.
- **Provider Comparison Tools:** Test your golem's performance across different LLM providers and models for the same task.
- **High-Level AI Functions:** Empower your golem with pre-built functions like ChainOfThought for complex reasoning tasks.
- **Robust Error Handling and Retries:** Build resilience into your golem with built-in retry mechanisms to handle API rate limits and transient errors.
- **Extensible Architecture:** Easily expand your golem's capabilities by extending support for new LLM providers and features.

## Real-World Applications

Your gollm-powered golems can handle a wide range of AI-powered tasks, including:

- **Content Creation Workflows:** Generate research summaries, article ideas, and refined paragraphs for writing projects.
- **Complex Reasoning Tasks:** Use the ChainOfThought function to break down and analyze complex problems step-by-step.
- **Structured Data Generation:** Create and validate complex data structures with customizable JSON schemas.
- **Model Performance Analysis:** Compare different models' performance for specific tasks to optimize your AI pipeline.
- **Prompt Optimization:** Automatically improve prompts for various tasks, from creative writing to technical documentation.

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

### Using Memory with LLM

gollm supports adding memory to your LLM instances, allowing them to maintain context across multiple interactions:

```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-3.5-turbo"),
    gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
    gollm.SetMemory(4096), // Enable memory with a 4096 token limit
)
```

### Comparing Models

The `CompareModels` function allows you to easily compare responses from different LLM providers or models:

```go
configs := []*gollm.Config{
	{Provider: "openai", Model: "gpt-4o-mini", APIKey: "your-openai-api-key"},
	{Provider: "anthropic", Model: "claude-3-5-sonnet-20240620", APIKey: "your-anthropic-api-key"},
	{Provider: "groq", Model: "llama-3.1-70b-versatile", APIKey: "your-groq-api-key"},
}

prompt := "Tell me a joke about programming. Respond in JSON format with 'setup' and 'punchline' fields."

results, err := gollm.CompareModels(context.Background(), prompt, validateJoke, configs...)
if err != nil {
	log.Fatalf("Error comparing models: %v", err)
}

fmt.Println(gollm.AnalyzeComparisonResults(results))
```

### Prompt Optimization

gollm now includes a powerful PromptOptimizer that can automatically refine and improve your prompts:

```go
optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDescription,
	gollm.WithCustomMetrics(
		gollm.Metric{Name: "Clarity", Description: "How clear and understandable the prompt is"},
		gollm.Metric{Name: "Relevance", Description: "How relevant the prompt is to the task"},
	),
	gollm.WithRatingSystem("numerical"),
	gollm.WithThreshold(0.8),
	gollm.WithVerbose(),
)

optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
if err != nil {
	log.Fatalf("Optimization failed: %v", err)
}

fmt.Printf("Optimized Prompt: %s\n", optimizedPrompt)
```

### JSON Output Validation

gollm supports automatic validation of JSON outputs from LLMs:

```go
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
```

## Examples and Tutorials

Check out our [examples directory](https://github.com/teilomillet/gollm/tree/main/examples) for more usage examples, including:

- Basic usage
- Different prompt types
- Comparing providers
- Advanced prompt templates
- Prompt optimization
- JSON output validation

## Project Status

`gollm` is actively maintained and under continuous development. We welcome contributions and feedback from the community.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
