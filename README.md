# gollm - Go Large Language Model



![Gophers building a robot by Renee French](img/gopherrobot4s.jpg)

<!-- ![The Gollm Golem](img/robot_golem.jpeg "Build your own AI golem with gollm") -->

`gollm` is a Go package designed to help you build your own AI golems. Just as the mystical golem of legend was brought to life with sacred words, gollm empowers you to breathe life into your AI creations using the power of Large Language Models (LLMs). This package simplifies and streamlines interactions with various LLM providers, offering a unified, flexible, and powerful interface for AI engineers and developers to craft their own digital servants.

[Documentation](https://docs.gollm.co)

## Key Features

- **Unified API for Multiple LLM Providers:** Shape your golem's mind using various providers, including OpenAI, Anthropic, and Groq. Seamlessly switch between models like GPT-4, GPT-4o-mini, Claude, and llama-3.1.
- **Easy Provider and Model Switching:** Mold your golem's capabilities by configuring preferred providers and models with simple function calls.
- **Flexible Configuration Options:** Customize your golem's essence using environment variables, code-based configuration, or configuration files to suit your project's needs.
- **Advanced Prompt Engineering:** Craft sophisticated instructions to guide your golem's responses effectively.
- **Memory Retention:** Maintain context across multiple interactions for more coherent conversations.
- **Structured Output and Validation:** Ensure your golem's outputs are consistent and reliable with JSON schema generation and validation.
- **Provider Comparison Tools:** Test your golem's performance across different LLM providers and models for the same task.
- **High-Level AI Functions:** Empower your golem with pre-built functions like ChainOfThought for complex reasoning tasks.
- **Robust Error Handling and Retries:** Build resilience into your golem with built-in retry mechanisms to handle API rate limits and transient errors.
- **Extensible Architecture:** Easily expand your golem's capabilities by extending support for new LLM providers and features.
- **Debug Logging:** Fine-tune your golem's performance with configurable debug levels.

## Real-World Applications

Your gollm-powered golems can handle a wide range of AI-powered tasks, including:

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

## Using Memory with LLM

gollm now supports adding memory to your LLM instances. This allows the LLM to maintain context across multiple interactions. To use this feature:

1. When creating a new LLM instance, use the `SetMemory` option:

```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-3.5-turbo"),
    gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
    gollm.SetMemory(4096), // Enable memory with a 4096 token limit
)
```

2. Use the LLM as usual. The memory will automatically be maintained:

```go
response, err := llm.Generate(context.Background(), gollm.NewPrompt("Hello, how are you?"))
// The next generation will include the context of the previous interaction
response, err = llm.Generate(context.Background(), gollm.NewPrompt("What did I just ask you?"))
```

3. If needed, you can clear the memory:

```go
if llmWithMemory, ok := llm.(*gollm.LLMWithMemory); ok {
    llmWithMemory.ClearMemory()
}
```

The memory feature uses tiktoken for accurate token counting and automatically manages the conversation history within the specified token limit.

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
