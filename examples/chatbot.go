// File: examples/chatbot.go

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/teilomillet/gollm"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with memory
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMemory(4000), // Enable memory with a 4000 token limit
		gollm.SetDebugLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	fmt.Println("Welcome to the Memory-Enabled Chatbot!")
	fmt.Println("Type 'exit' to quit, or 'clear memory' to reset the conversation.")

	reader := bufio.NewReader(os.Stdin)
	ctx := context.Background()

	for {
		fmt.Print("You: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			break
		}

		if input == "clear memory" {
			if memoryLLM, ok := llm.(interface{ ClearMemory() }); ok {
				memoryLLM.ClearMemory()
				fmt.Println("Memory cleared. Starting a new conversation.")
			}
			continue
		}

		prompt := gollm.NewPrompt(input)
		response, err := llm.Generate(ctx, prompt)
		if err != nil {
			log.Printf("Error generating response: %v", err)
			continue
		}

		fmt.Printf("Chatbot: %s\n", response)
	}

	fmt.Println("Thank you for chatting!")
}
