# goal - Go Abstract Language Model Interface

`goal` is a Go package that provides a simple, unified interface for interacting with various Language Model (LLM) providers. It abstracts away the differences between different LLM APIs, allowing you to easily switch between providers or use multiple providers in your application.

## Installation

To install the `goal` package, use `go get`:

```
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
	llmClient, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	// Generate a response
	ctx := context.Background()
	question := "What are the main benefits of artificial intelligence?"
	answer, err := goal.QuestionAnswer(ctx, llmClient, question, "")
	if err != nil {
		log.Fatalf("Failed to generate answer: %v", err)
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Answer:\n%s\n", answer)
}
```

## Key Features

1. **Unified Interface**: Interact with different LLM providers using a consistent API.
2. **Chain of Thought**: Perform step-by-step reasoning for complex problems.
3. **Customizable Prompts**: Create and reuse structured prompts for various tasks.
4. **High-Level Functions**: Use pre-built functions for common tasks like question answering and summarization.

## Advanced Usage

### Creating Custom Prompts

You can create reusable prompt templates for more complex tasks:

```go
promptTemplate := goal.NewPrompt("Analyze the following topic:").
	WithDirective("Consider multiple perspectives").
	WithDirective("Provide pros and cons").
	WithOutput("Analysis:")

topic := "The impact of AI on job markets"
fullPrompt := promptTemplate.WithInput(topic)
analysis, _, err := llmClient.Generate(ctx, fullPrompt.String())
```

### Chaining Operations

Combine different goal package features for more complex workflows:

```go
expertAnalysis, _, err := llmClient.Generate(ctx, promptTemplate.WithInput(topic).String())
if err != nil {
	log.Fatal(err)
}

summary, err := goal.Summarize(ctx, llmClient, expertAnalysis, 50)
if err != nil {
	log.Fatal(err)
}

keyPoints, err := goal.ChainOfThought(ctx, llmClient, fmt.Sprintf("Extract key points from:\n%s", expertAnalysis))
if err != nil {
	log.Fatal(err)
}
```
