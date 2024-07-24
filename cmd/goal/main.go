package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/goal"
)

func main() {
	// Command-line flags for all configuration options
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")
	provider := flag.String("provider", "", "LLM provider (anthropic, openai, groq)")
	model := flag.String("model", "", "LLM model")
	temperature := flag.Float64("temperature", -1, "LLM temperature")
	maxTokens := flag.Int("max-tokens", 0, "LLM max tokens")
	timeout := flag.Duration("timeout", 0, "LLM timeout")
	apiKey := flag.String("api-key", "", "API key for the specified provider")
	maxRetries := flag.Int("max-retries", 3, "Maximum number of retries for API calls")
	retryDelay := flag.Duration("retry-delay", time.Second*2, "Delay between retries")
	debugLevel := flag.String("debug-level", "warn", "Debug level (debug, info, warn, error)")
	outputFormat := flag.String("output-format", "", "Output format for structured responses (json)")

	flag.Parse()

	// Prepare configuration options
	var configOpts []goal.ConfigOption

	if *provider != "" {
		configOpts = append(configOpts, goal.SetProvider(*provider))
	}
	if *model != "" {
		configOpts = append(configOpts, goal.SetModel(*model))
	}
	if *temperature != -1 {
		configOpts = append(configOpts, goal.SetTemperature(*temperature))
	}
	if *maxTokens != 0 {
		configOpts = append(configOpts, goal.SetMaxTokens(*maxTokens))
	}
	if *timeout != 0 {
		configOpts = append(configOpts, goal.SetTimeout(*timeout))
	}
	if *apiKey != "" {
		configOpts = append(configOpts, goal.SetAPIKey(*apiKey))
	}
	configOpts = append(configOpts, goal.SetMaxRetries(*maxRetries))
	configOpts = append(configOpts, goal.SetRetryDelay(*retryDelay))
	configOpts = append(configOpts, goal.SetDebugLevel(goal.LogLevel(getLogLevel(*debugLevel))))

	// Create LLM client with the specified options
	llmClient, err := goal.NewLLM(configOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating LLM client: %v\n", err)
		os.Exit(1)
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(flag.Args(), " ")
	ctx := context.Background()

	var response string
	var fullPrompt string

	switch *promptType {
	case "qa":
		response, err = goal.QuestionAnswer(ctx, llmClient, rawPrompt)
	case "cot":
		response, err = goal.ChainOfThought(ctx, llmClient, rawPrompt)
	case "summarize":
		response, err = goal.Summarize(ctx, llmClient, rawPrompt)
	default:
		prompt := goal.NewPrompt(rawPrompt)
		if *outputFormat == "json" {
			prompt.Apply(goal.WithOutput("Please provide your response in JSON format."))
		}
		response, err = llmClient.Generate(ctx, prompt, goal.WithJSONSchemaValidation())
		fullPrompt = prompt.String()
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		if fullPrompt == "" {
			fullPrompt = rawPrompt // For qa, cot, and summarize, we don't have access to the full prompt
		}
		fmt.Printf("Prompt Type: %s\nFull Prompt:\n%s\n\nResponse:\n---------\n", *promptType, fullPrompt)
	}

	if *outputFormat == "json" {
		var jsonResponse interface{}
		err := json.Unmarshal([]byte(response), &jsonResponse)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing JSON response: %v\n", err)
			fmt.Println(response) // Print raw response if JSON parsing fails
		} else {
			jsonPretty, _ := json.MarshalIndent(jsonResponse, "", "  ")
			fmt.Println(string(jsonPretty))
		}
	} else {
		fmt.Println(response)
	}
}

func getLogLevel(level string) goal.LogLevel {
	switch strings.ToLower(level) {
	case "debug":
		return goal.LogLevelDebug
	case "info":
		return goal.LogLevelInfo
	case "error":
		return goal.LogLevelError
	default:
		return goal.LogLevelWarn
	}
}

