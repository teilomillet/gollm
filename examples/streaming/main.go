package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

func runStream(llm gollm.LLM, prompt *gollm.Prompt) error {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Start streaming
	stream, err := llm.Stream(ctx, prompt)
	if err != nil {
		return fmt.Errorf("failed to start stream: %w", err)
	}
	defer func() {
		if err := stream.Close(); err != nil {
			log.Printf("Warning: Failed to close stream: %v", err)
		}
	}()

	fmt.Println("\nStreaming response:")
	fmt.Println("-------------------")

	var fullResponse strings.Builder
	tokenCount := 0

	// Read and print tokens as they arrive
	for {
		token, err := stream.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading stream: %w", err)
		}

		fmt.Print(token.Text)
		fullResponse.WriteString(token.Text)
		tokenCount++
	}

	fmt.Println("\n-------------------")
	fmt.Printf("\nReceived %d tokens\n", tokenCount)
	fmt.Printf("Total response length: %d characters\n", len(fullResponse.String()))
	return nil
}

func main() {
	fmt.Println("Streaming Example with OpenAI and Anthropic")
	fmt.Println("=========================================")

	// Test OpenAI streaming
	fmt.Println("\nTesting OpenAI Streaming:")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetAPIKey(openaiKey),
			gollm.SetMaxTokens(500),
			gollm.SetMaxRetries(3),
			gollm.SetRetryDelay(time.Second*2),
			gollm.SetLogLevel(gollm.LogLevelInfo),
		)
		if err != nil {
			log.Printf("Failed to create OpenAI LLM: %v", err)
		} else {
			prompt := gollm.NewPrompt(
				"Write a short story about a programmer discovering an AI that can write code.",
				gollm.WithSystemPrompt("You are a creative writer who specializes in tech fiction.", gollm.CacheTypeEphemeral),
			)
			if err := runStream(llm, prompt); err != nil {
				log.Printf("OpenAI streaming error: %v", err)
			}
		}
	} else {
		fmt.Println("Skipping OpenAI (OPENAI_API_KEY not set)")
	}

	// Test Anthropic streaming
	fmt.Println("\nTesting Anthropic Streaming:")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	if anthropicKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("anthropic"),
			gollm.SetModel("claude-3-5-haiku-latest"),
			gollm.SetAPIKey(anthropicKey),
			gollm.SetMaxTokens(1000),
			gollm.SetMaxRetries(3),
			gollm.SetRetryDelay(time.Second*2),
			gollm.SetLogLevel(gollm.LogLevelInfo),
		)
		if err != nil {
			log.Printf("Failed to create Anthropic LLM: %v", err)
		} else {
			prompt := gollm.NewPrompt(
				"Write a short story about a programmer discovering an AI that can write code.",
				gollm.WithSystemPrompt("You are a creative writer who specializes in tech fiction.", gollm.CacheTypeEphemeral),
			)
			if err := runStream(llm, prompt); err != nil {
				log.Printf("Anthropic streaming error: %v", err)
			}
		}
	} else {
		fmt.Println("Skipping Anthropic (ANTHROPIC_API_KEY not set)")
	}
}
