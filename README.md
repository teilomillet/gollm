# gollm - Go Large Language Model

<div align="center">
  <img src="img/gopherrobot4s.jpg" alt="Gophers building a robot by Renee French">
</div>

`gollm` is a Go package designed to help you build your own AI golems. Just as the mystical golem of legend was brought to life with sacred words, `gollm` empowers you to breathe life into your AI creations using the power of Large Language Models (LLMs). This package simplifies and streamlines interactions with various LLM providers, offering a unified, flexible, and powerful interface for AI engineers and developers to craft their own digital servants.

[![Ed Zynda's Video](https://img.youtube.com/vi/679U9tzXDw4/0.jpg)](https://www.youtube.com/watch?v=679U9tzXDw4)

[Documentation](https://docs.gollm.co)

## Table of Contents

- [Key Features](#key-features)
- [Real-World Applications](#real-world-applications)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
- [Quick Reference](#quick-reference)
  - [LLM Creation and Configuration](#llm-creation-and-configuration)
  - [Prompt Creation](#prompt-creation)
  - [Generate Response](#generate-response)
  - [Chain of Thought](#chain-of-thought)
  - [Prompt Optimization](#prompt-optimization)
  - [Model Comparison](#model-comparison)
- [Advanced Usage](#advanced-usage)
  - [Prompt Engineering](#prompt-engineering)
  - [Pre-built Functions (Chain of Thought)](#pre-built-functions-chain-of-thought)
  - [Working with Examples](#working-with-examples)
  - [Prompt Templates](#prompt-templates)
  - [Structured Output (JSON Output Validation)](#structured-output-json-output-validation)
  - [Prompt Optimizer](#prompt-optimizer)
  - [Model Comparison](#model-comparison-1)
  - [Memory Retention](#memory-retention)
- [Best Practices](#best-practices)
- [Examples and Tutorials](#examples-and-tutorials)
- [Project Status](#project-status)
- [Philosophy](#philosophy)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- **Unified API for Multiple LLM Providers:** Interact seamlessly with various providers, including OpenAI, Anthropic, Groq, and Ollama. Easily switch between models like GPT-4, GPT-4o-mini, Claude, and Llama-3.1.
- **Easy Provider and Model Switching:** Configure preferred providers and models with simple options.
- **Flexible Configuration Options:** Customize using environment variables, code-based configuration, or configuration files.
- **Advanced Prompt Engineering:** Craft sophisticated instructions to guide your AI's responses effectively.
- **Prompt Optimizer:** Automatically refine and improve your prompts for better results, with support for custom metrics and different rating systems.
- **Memory Retention:** Maintain context across multiple interactions for more coherent conversations.
- **Structured Output and Validation:** Ensure outputs are consistent and reliable with JSON schema generation and validation.
- **Provider Comparison Tools:** Test performance across different LLM providers and models for the same task.
- **High-Level AI Functions:** Use pre-built functions like `ChainOfThought` for complex reasoning tasks.
- **Robust Error Handling and Retries:** Built-in retry mechanisms to handle API rate limits and transient errors.
- **Extensible Architecture:** Easily expand support for new LLM providers and features.

## Real-World Applications

`gollm` can handle a wide range of AI-powered tasks, including:

- **Content Creation Workflows:** Generate research summaries, article ideas, and refined paragraphs.
- **Complex Reasoning Tasks:** Use the `ChainOfThought` function to analyze complex problems step-by-step.
- **Structured Data Generation:** Create and validate complex data structures with customizable JSON schemas.
- **Model Performance Analysis:** Compare different models' performance for specific tasks.
- **Prompt Optimization:** Automatically improve prompts for various tasks.
- **Mixture of Agents:** Combine responses from multiple LLM providers.

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
    "os"

    "github.com/teilomillet/gollm"
)

func main() {
    // Load API key from environment variable
    apiKey := os.Getenv("OPENAI_API_KEY")
    if apiKey == "" {
        log.Fatalf("OPENAI_API_KEY environment variable is not set")
    }

    // Create a new LLM instance with custom configuration
    llm, err := gollm.NewLLM(
        gollm.SetProvider("openai"),
        gollm.SetModel("gpt-4o-mini"),
        gollm.SetAPIKey(apiKey),
        gollm.SetMaxTokens(200),
        gollm.SetMaxRetries(3),
        gollm.SetRetryDelay(time.Second*2),
        gollm.SetLogLevel(gollm.LogLevelInfo),
    )
    if err != nil {
        log.Fatalf("Failed to create LLM: %v", err)
    }

    ctx := context.Background()

    // Create a basic prompt
    prompt := gollm.NewPrompt("Explain the concept of 'recursion' in programming.")

    // Generate a response
    response, err := llm.Generate(ctx, prompt)
    if err != nil {
        log.Fatalf("Failed to generate text: %v", err)
    }
    fmt.Printf("Response:\n%s\n", response)
}
```

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
response, err := tools.ChainOfThought(ctx, llm, "Your question here")
```

### Prompt Optimization

```go
optimizer := optimizer.NewPromptOptimizer(llm, initialPrompt, taskDescription,
    optimizer.WithCustomMetrics(/* custom metrics */),
    optimizer.WithRatingSystem("numerical"),
    optimizer.WithThreshold(0.8),
)
optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
```

### Model Comparison

```go
results, err := tools.CompareModels(ctx, promptText, validateFunc, configs...)
```

## Advanced Usage

The `gollm` package offers a range of advanced features to enhance your AI applications:

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
response, err := tools.ChainOfThought(ctx, llm, question)
if err != nil {
    log.Fatalf("Failed to perform chain of thought: %v", err)
}
fmt.Printf("Chain of Thought:\n%s\n", response)
```

### Working with Examples

Load examples directly from files:

```go
examples, err := utils.ReadExamplesFromFile("examples.txt")
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

Use the `PromptOptimizer` to automatically refine and improve your prompts:

```go
initialPrompt := gollm.NewPrompt("Write a short story about a robot learning to love.")
taskDescription := "Generate a compelling short story that explores the theme of artificial intelligence developing emotions."

optimizerInstance := optimizer.NewPromptOptimizer(
    llm,
    initialPrompt,
    taskDescription,
    optimizer.WithCustomMetrics(
        optimizer.Metric{Name: "Creativity", Description: "How original and imaginative the story is"},
        optimizer.Metric{Name: "Emotional Impact", Description: "How well the story evokes feelings in the reader"},
    ),
    optimizer.WithRatingSystem("numerical"),
    optimizer.WithThreshold(0.8),
    optimizer.WithVerbose(),
)

optimizedPrompt, err := optimizerInstance.OptimizePrompt(ctx)
if err != nil {
    log.Fatalf("Optimization failed: %v", err)
}

fmt.Printf("Optimized Prompt: %s\n", optimizedPrompt.Input)
```

### Model Comparison

Compare responses from different LLM providers or models:

```go
configs := []*gollm.Config{
    {
        Provider:  "openai",
        Model:     "gpt-4o-mini",
        APIKey:    os.Getenv("OPENAI_API_KEY"),
        MaxTokens: 500,
    },
    {
        Provider:  "anthropic",
        Model:     "claude-3-5-sonnet-20240620",
        APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
        MaxTokens: 500,
    },
    {
        Provider:  "groq",
        Model:     "llama-3.1-70b-versatile",
        APIKey:    os.Getenv("GROQ_API_KEY"),
        MaxTokens: 500,
    },
}

promptText := "Tell me a joke about programming. Respond in JSON format with 'setup' and 'punchline' fields."

validateJoke := func(joke map[string]interface{}) error {
    if joke["setup"] == "" || joke["punchline"] == "" {
        return fmt.Errorf("joke must have both a setup and a punchline")
    }
    return nil
}

results, err := tools.CompareModels(context.Background(), promptText, validateJoke, configs...)
if err != nil {
    log.Fatalf("Error comparing models: %v", err)
}

fmt.Println(tools.AnalyzeComparisonResults(results))
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
   - Use `NewPrompt()` with options like `WithContext()`, `WithDirectives()`, and `WithOutput()` to create well-structured prompts.
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
     response, err := tools.ChainOfThought(ctx, llm, "Your complex question here")
     ```

4. **Work with Examples**:
   - Use `ReadExamplesFromFile()` to load examples from files for consistent outputs.
   - Example:
     ```go
     examples, err := utils.ReadExamplesFromFile("examples.txt")
     if err != nil {
         log.Fatalf("Failed to read examples: %v", err)
     }
     ```

5. **Implement Structured Output**:
   - Use `WithJSONSchemaValidation()` to ensure valid JSON outputs.
   - Example:
     ```go
     response, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
     ```

6. **Optimize Prompts**:
   - Utilize `PromptOptimizer` to refine prompts automatically.
   - Example:
     ```go
     optimizer := optimizer.NewPromptOptimizer(llm, initialPrompt, taskDescription,
         optimizer.WithCustomMetrics(
             optimizer.Metric{Name: "Relevance", Description: "How relevant the response is to the task"},
         ),
         optimizer.WithRatingSystem("numerical"),
         optimizer.WithThreshold(0.8),
     )
     ```

7. **Compare Model Performances**:
   - Use `CompareModels()` to evaluate different models or providers.
   - Example:
     ```go
     results, err := tools.CompareModels(ctx, promptText, validateFunc, configs...)
     ```

8. **Implement Memory for Contextual Interactions**:
   - Enable memory retention for maintaining context across interactions.
   - Example:
     ```go
     llm, err := gollm.NewLLM(
         gollm.SetProvider("openai"),
         gollm.SetModel("gpt-3.5-turbo"),
         gollm.SetMemory(4096),
     )
     ```

9. **Error Handling and Retries**:
   - Always check for errors and configure retry mechanisms.
   - Example:
     ```go
     llm, err := gollm.NewLLM(
         gollm.SetMaxRetries(3),
         gollm.SetRetryDelay(time.Second * 2),
     )
     ```

10. **Secure API Key Handling**:
    - Use environment variables for API keys.
    - Example:
      ```go
      llm, err := gollm.NewLLM(
          gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
      )
      ```

## Examples and Tutorials

Check out our [examples directory](https://github.com/teilomillet/gollm/tree/main/examples) for more usage examples, including:

- Basic usage
- Different prompt types
- Comparing providers
- Advanced prompt templates
- Prompt optimization
- JSON output validation
- Mixture of Agents

## Project Status

`gollm` is actively maintained and under continuous development. With the recent refactoring, we've streamlined the codebase to make it simpler and more accessible for new contributors. We welcome contributions and feedback from the community.

## Philosophy

`gollm` is built on a philosophy of pragmatic minimalism and forward-thinking simplicity:

1. **Build what's necessary**: We add features as they become needed, avoiding speculative development.

2. **Simplicity first**: Additions should be straightforward while fulfilling their purpose.

3. **Future-compatible**: We consider how current changes might impact future development.

4. **Readability counts**: Code should be clear and self-explanatory.

5. **Modular design**: Each component should do one thing well.

## Contributing

We welcome contributions that align with our philosophy! Whether you're fixing a bug, improving documentation, or proposing new features, your efforts are appreciated.

To get started:

1. Familiarize yourself with our [philosophy](#philosophy).
2. Check out our [CONTRIBUTING.md](CONTRIBUTING.md).
3. Look through our [issues](https://github.com/teilomillet/gollm/issues).
4. Fork the repository, make your changes, and submit a pull request.

Thank you for helping make `gollm` better!

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

