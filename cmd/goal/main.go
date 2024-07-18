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
	configPaths := flag.String("configs", "", "Comma-separated paths to config files")
	compare := flag.Bool("compare", false, "Compare responses from multiple providers")
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")

	flag.Parse()

	// Load configurations
	var configs []*llm.Config
	if *configPaths == "" {
		loadedConfigs, err := llm.LoadConfigs()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading configs: %v\n", err)
			os.Exit(1)
		}
		for _, cfg := range loadedConfigs {
			configs = append(configs, cfg)
		}
	} else {
		paths := strings.Split(*configPaths, ",")
		loadedConfigs, err := llm.LoadConfigs(paths...)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading configs: %v\n", err)
			os.Exit(1)
		}
		for _, cfg := range loadedConfigs {
			configs = append(configs, cfg)
		}
	}

	if len(configs) == 0 {
		fmt.Fprintf(os.Stderr, "No valid configurations found\n")
		os.Exit(1)
	}

	// Set log level based on the first config
	llm.SetLogLevel(llm.LogLevelFromString(configs[0].LogLevel))

	// Check remaining arguments
	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(args, " ")

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

	ctx := context.Background()

	if *compare {
		results := llm.CompareProviders(ctx, prompt.String(), configs...)
		llm.PrintComparisonResults(results)
	} else {
		// Use the first config for single provider mode
		config := configs[0]

		fmt.Println("Debug: Loaded configurations:")
		for i, cfg := range configs {
			fmt.Printf("Config %d: %+v\n", i, *cfg)
		}

		fmt.Println("Debug: All environment variables:")
		for _, env := range os.Environ() {
			fmt.Println(env)
		}
		fmt.Printf("Debug: ANTHROPIC_API_KEY=%s\n", os.Getenv("ANTHROPIC_API_KEY"))

		apiKey := os.Getenv(config.Provider + "_API_KEY")
		if apiKey == "" {
			fmt.Fprintf(os.Stderr, "Error: API key for %s not set. Please set the %s_API_KEY environment variable.\n", config.Provider, strings.ToUpper(config.Provider))
			os.Exit(1)
		}

		llmProvider, err := llm.GetProvider(config.Provider, apiKey, config.Model)
		if err != nil {
			llm.Logger.Fatal("Error creating LLM provider", zap.Error(err))
		}

		llmClient := llm.NewLLM(llmProvider)
		llmClient.SetOption("temperature", config.Temperature)
		llmClient.SetOption("max_tokens", config.MaxTokens)

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
}

