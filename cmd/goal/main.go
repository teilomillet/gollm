// File: cmd/goal/main.go

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/teilomillet/goal/llm"
	"go.uber.org/zap"
)

func main() {
	// Define flags for optional parameters
	temperature := flag.Float64("temperature", 0.7, "Temperature for the LLM")
	maxTokens := flag.Int("max-tokens", 300, "Max tokens for the LLM response")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")

	// Parse flags
	flag.Parse()

	// Set log level
	switch *logLevel {
	case "debug":
		llm.SetLogLevel(zap.DebugLevel)
	case "info":
		llm.SetLogLevel(zap.InfoLevel)
	case "warn":
		llm.SetLogLevel(zap.WarnLevel)
	case "error":
		llm.SetLogLevel(zap.ErrorLevel)
	default:
		llm.Logger.Fatal("Invalid log level", zap.String("log_level", *logLevel))
	}

	// Check remaining arguments
	args := flag.Args()
	if len(args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <provider> <model> <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	provider := args[0]
	model := args[1]
	rawPrompt := strings.Join(args[2:], " ")

	// Ensure the appropriate API key is set in the environment
	apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(provider))
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		llm.Logger.Fatal("API key not set", zap.String("env_var", apiKeyEnv))
	}

	ctx := context.Background()

	llmProvider, err := llm.GetProvider(provider, apiKey, model)
	if err != nil {
		llm.Logger.Fatal("Error creating LLM provider", zap.Error(err))
	}

	llmClient := llm.NewLLM(llmProvider)

	// Set options
	llmClient.SetOption("temperature", *temperature)
	llmClient.SetOption("max_tokens", *maxTokens)

	// Create prompt based on type
	var prompt *llm.Prompt
	switch *promptType {
	case "raw":
		prompt = llm.NewPrompt(rawPrompt)
	case "qa":
		prompt = llm.QuestionAnswer(rawPrompt)
	case "cot":
		prompt = llm.ChainOfThought(rawPrompt)
	case "summarize":
		prompt = llm.Summarize(rawPrompt)
	default:
		llm.Logger.Fatal("Invalid prompt type", zap.String("type", *promptType))
	}

	// Generate response
	response, fullPrompt, err := llmClient.Generate(ctx, prompt.String())
	if err != nil {
		llm.Logger.Fatal("Error generating text", zap.Error(err))
	}

	if *verbose {
		fmt.Println("Full Prompt:")
		fmt.Println("------------")
		fmt.Println(fullPrompt)
		fmt.Println("\nResponse:")
		fmt.Println("---------")
	}

	fmt.Println(response)
}

