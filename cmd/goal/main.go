// cmd/goal/main.go
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
	maxTokens := flag.Int("max-tokens", 100, "Max tokens for the LLM response")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")

	// Custom usage message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <provider> <model> <prompt>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  or:  %s [flags] <model> <prompt> (provider determined from environment)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nFlags:\n")
		flag.PrintDefaults()
	}

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
	if len(args) < 2 {
		flag.Usage()
		os.Exit(1)
	}

	var provider, model, prompt string

	// Determine provider, model, and prompt based on the number of arguments
	switch len(args) {
	case 2:
		// If only two args, assume provider is to be determined from environment
		model = args[0]
		prompt = args[1]
		// Try to determine provider from environment variables
		for _, p := range []string{"OPENAI", "ANTHROPIC", "GROQ"} {
			if os.Getenv(p+"_API_KEY") != "" {
				provider = strings.ToLower(p)
				break
			}
		}
		if provider == "" {
			llm.Logger.Fatal("Provider not specified and couldn't be determined from environment variables")
		}
	case 3:
		// If three args, use them as provider, model, and prompt
		provider = args[0]
		model = args[1]
		prompt = args[2]
	default:
		// If more than three args, use the first three as provider, model, and join the rest as prompt
		provider = args[0]
		model = args[1]
		prompt = strings.Join(args[2:], " ")
	}

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

	response, err := llmClient.Generate(ctx, prompt)
	if err != nil {
		llm.Logger.Fatal("Error generating text", zap.Error(err))
	}

	fmt.Println("Response:", response)
}
