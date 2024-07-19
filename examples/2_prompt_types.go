// File: examples/2_prompt_types.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal/llm"
)

func main() {
	config, err := llm.LoadConfig("")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	llmClient, err := llm.NewLLMFromConfig(config)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	examples := []struct {
		name   string
		prompt *llm.Prompt
	}{
		{"Question-Answer", llm.QuestionAnswer("What is the capital of France?")},
		{"Chain-of-Thought", llm.ChainOfThought("If a train travels 120 km in 2 hours, what is its average speed?")},
		{"Summarize", llm.Summarize("The Internet of Things (IoT) is transforming how we live and work. It refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data.")},
	}

	for _, ex := range examples {
		fmt.Printf("--- %s Example ---\n", ex.name)
		response, _, err := llmClient.Generate(context.Background(), ex.prompt.String())
		if err != nil {
			log.Printf("Failed to generate response for %s: %v\n", ex.name, err)
			continue
		}
		fmt.Printf("Response:\n%s\n\n", response)
	}
}
