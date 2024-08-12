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

## Quick Reference

Here's a quick reference guide for the most commonly used functions and options in the `gollm` package:

### LLM Creation and Configuration
```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4"),
    gollm.SetAPIKey("your-api-key"),
    gollm.SetMaxTokens(100),
    gollm.SetTemperature(0.7),
    gollm.SetMemory(4096),
)
```

### Prompt Creation
```go
prompt := gollm.NewPrompt("Your prompt text here",
    gollm.WithContext("Additional context"),
    gollm.WithDirectives("Be concise", "Use examples"),
    gollm.WithOutput("Expected output format"),
    gollm.WithMaxLength(300),
)
```

### Generate Response
```go
response, err := llm.Generate(ctx, prompt)
```

### Chain of Thought
```go
response, err := gollm.ChainOfThought(ctx, llm, "Your question here")
```

### Prompt Optimization
```go
optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDescription,
    gollm.WithCustomMetrics(/* custom metrics */),
    gollm.WithRatingSystem("numerical"),
    gollm.WithThreshold(0.8),
)
optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
```

### Model Comparison
```go
results, err := gollm.CompareModels(ctx, prompt, validateFunc, config1, config2, config3)
```


## Advanced Usage

The `gollm` package offers a range of advanced features to enhance your AI applications:

- Prompt Engineering
- Pre-built Functions (e.g., Chain of Thought)
- Working with Examples
- Structured Output (JSON output validation)
- Prompt Optimizer
- Model Comparison
- Memory Retention

Here are examples of how to use these advanced features:

### Prompt Engineering

Create sophisticated prompts with multiple components:

```go
prompt := gollm.NewPrompt("Explain the concept of recursion in programming.",
    gollm.WithContext("The audience is beginner programmers."),
    gollm.WithDirectives(
        "Use simple language and avoid jargon.",
        "Provide a practical example.",
        "Explain potential pitfalls and how to avoid them.",
    ),
    gollm.WithOutput("Structure your response with sections: Definition, Example, Pitfalls, Best Practices."),
    gollm.WithMaxLength(300),
)

response, err := llm.Generate(ctx, prompt)
if err != nil {
    log.Fatalf("Failed to generate explanation: %v", err)
}

fmt.Printf("Explanation of Recursion:\n%s\n", response)
```

### Pre-built Functions (Chain of Thought)

Use the `ChainOfThought` function for step-by-step reasoning:

```go
question := "What is the result of 15 * 7 + 22?"
response, err := gollm.ChainOfThought(ctx, llm, question)
if err != nil {
    log.Fatalf("Failed to perform chain of thought: %v", err)
}
fmt.Printf("Chain of Thought:\n%s\n", response)
```

### Working with Examples

Load examples directly from files:

```go
examples, err := gollm.readExamplesFromFile("examples.txt")
if err != nil {
    log.Fatalf("Failed to read examples: %v", err)
}

prompt := gollm.NewPrompt("Generate a similar example:",
    gollm.WithExamples(examples...),
)

response, err := llm.Generate(ctx, prompt)
if err != nil {
    log.Fatalf("Failed to generate example: %v", err)
}
fmt.Printf("Generated Example:\n%s\n", response)
```


### Prompt Templates

Create reusable prompt templates for consistent prompt generation:

```go
// Create a new prompt template
template := gollm.NewPromptTemplate(
    "AnalysisTemplate",
    "A template for analyzing topics",
    "Provide a comprehensive analysis of {{.Topic}}. Consider the following aspects:\n" +
    "1. Historical context\n" +
    "2. Current relevance\n" +
    "3. Future implications",
    gollm.WithPromptOptions(
        gollm.WithDirectives(
            "Use clear and concise language",
            "Provide specific examples where appropriate",
        ),
        gollm.WithOutput("Structure your analysis with clear headings for each aspect."),
    ),
)

// Use the template to create a prompt
data := map[string]interface{}{
    "Topic": "artificial intelligence in healthcare",
}
prompt, err := template.Execute(data)
if err != nil {
    log.Fatalf("Failed to execute template: %v", err)
}

// Generate a response using the created prompt
response, err := llm.Generate(ctx, prompt)
if err != nil {
    log.Fatalf("Failed to generate response: %v", err)
}

fmt.Printf("Analysis:\n%s\n", response)
```

### Structured Output (JSON Output Validation)

Ensure your LLM outputs are in a valid JSON format:

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

fmt.Printf("Analysis: %+v\n", result)
```

### Prompt Optimizer

Use the PromptOptimizer to automatically refine and improve your prompts:

```go
initialPrompt := "Write a short story about a robot learning to love."
taskDescription := "Generate a compelling short story that explores the theme of artificial intelligence developing emotions."

optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDescription,
    gollm.WithCustomMetrics(
        gollm.Metric{Name: "Creativity", Description: "How original and imaginative the story is"},
        gollm.Metric{Name: "Emotional Impact", Description: "How well the story evokes feelings in the reader"},
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

### Model Comparison

Compare responses from different LLM providers or models:

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

### Memory Retention

Enable memory to maintain context across multiple interactions:

```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-3.5-turbo"),
    gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
    gollm.SetMemory(4096), // Enable memory with a 4096 token limit
)
if err != nil {
    log.Fatalf("Failed to create LLM: %v", err)
}

ctx := context.Background()

// First interaction
prompt1 := gollm.NewPrompt("What's the capital of France?")
response1, err := llm.Generate(ctx, prompt1)
if err != nil {
    log.Fatalf("Failed to generate response: %v", err)
}
fmt.Printf("Response 1: %s\n", response1)

// Second interaction, referencing the first
prompt2 := gollm.NewPrompt("What's the population of that city?")
response2, err := llm.Generate(ctx, prompt2)
if err != nil {
    log.Fatalf("Failed to generate response: %v", err)
}
fmt.Printf("Response 2: %s\n", response2)
```

## Best Practices

1. **Prompt Engineering**: 
   - Use the `NewPrompt()` function with options like `WithContext()`, `WithDirectives()`, and `WithOutput()` to create well-structured prompts.
   - Example:
     ```go
     prompt := gollm.NewPrompt("Your main prompt here",
         gollm.WithContext("Provide relevant context"),
         gollm.WithDirectives("Be concise", "Use examples"),
         gollm.WithOutput("Specify expected output format"),
     )
     ```

2. **Utilize Prompt Templates**:
   - For consistent prompt generation, create and use `PromptTemplate` objects.
   - Example:
     ```go
     template := gollm.NewPromptTemplate(
         "CustomTemplate",
         "A template for custom prompts",
         "Generate a {{.Type}} about {{.Topic}}",
         gollm.WithPromptOptions(
             gollm.WithDirectives("Be creative", "Use vivid language"),
             gollm.WithOutput("Your {{.Type}}:"),
         ),
     )
     ```

3. **Leverage Pre-built Functions**:
   - Use provided functions like `ChainOfThought()` for complex reasoning tasks.
   - Example:
     ```go
     response, err := gollm.ChainOfThought(ctx, llm, "Your complex question here")
     ```

4. **Work with Examples**:
   - Use the `readExamplesFromFile()` function to load examples from files for more consistent and varied outputs.
   - Example:
     ```go
     examples, err := gollm.readExamplesFromFile("examples.txt")
     if err != nil {
         log.Fatalf("Failed to read examples: %v", err)
     }
     ```

5. **Implement Structured Output**:
   - Use the `WithJSONSchemaValidation()` option when generating responses to ensure valid JSON outputs.
   - Example:
     ```go
     response, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
     ```

6. **Optimize Prompts**:
   - Utilize the `PromptOptimizer` to refine and improve your prompts automatically.
   - Example:
     ```go
     optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDescription,
         gollm.WithCustomMetrics(
             gollm.Metric{Name: "Relevance", Description: "How relevant the response is to the task"},
         ),
         gollm.WithRatingSystem("numerical"),
         gollm.WithThreshold(0.8),
     )
     optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
     ```

7. **Compare Model Performances**:
   - Use the `CompareModels()` function to evaluate different models or providers for specific tasks.
   - Example:
     ```go
     results, err := gollm.CompareModels(ctx, prompt, validateFunc, config1, config2, config3)
     ```

8. **Implement Memory for Contextual Interactions**:
   - Enable memory retention for maintaining context across multiple interactions.
   - Example:
     ```go
     llm, err := gollm.NewLLM(
         gollm.SetProvider("openai"),
         gollm.SetModel("gpt-3.5-turbo"),
         gollm.SetMemory(4096), // Enable memory with a 4096 token limit
     )
     ```

9. **Error Handling and Retries**:
   - Always check for errors returned by gollm functions.
   - Configure retry mechanisms to handle transient errors and rate limits.
   - Example:
     ```go
     llm, err := gollm.NewLLM(
         gollm.SetMaxRetries(3),
         gollm.SetRetryDelay(time.Second * 2),
     )
     ```

10. **Secure API Key Handling**:
    - Use environment variables or secure configuration management to handle API keys.
    - Example:
      ```go
      llm, err := gollm.NewLLM(
          gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
      )
      ```

By following these best practices, you can make the most effective use of the gollm package, creating more robust, efficient, and maintainable AI-powered applications.
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
