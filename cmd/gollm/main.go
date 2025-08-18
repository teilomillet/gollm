// Package main provides a command-line interface for the GoLLM library.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/weave-labs/gollm/llm"

	"github.com/weave-labs/gollm"
	"github.com/weave-labs/gollm/internal/debug"
	"github.com/weave-labs/gollm/optimizer"
	"github.com/weave-labs/gollm/presets"
)

// Default configuration values
const (
	DefaultMaxRetries         = 3
	DefaultRetryDelaySeconds  = 2
	DefaultOptimizeIterations = 5
	DefaultOptimizeMemory     = 2
)

// cmdFlags holds all command-line flags
type cmdFlags struct {
	apiKey             string
	optimizeGoal       string
	provider           string
	model              string
	outputFormat       string
	debugLevel         string
	promptType         string
	timeout            time.Duration
	maxRetries         int
	retryDelay         time.Duration
	maxTokens          int
	temperature        float64
	optimizeIterations int
	optimizeMemory     int
	verbose            bool
}

// parseFlags parses command-line flags
func parseFlags() *cmdFlags {
	flags := &cmdFlags{}
	flag.StringVar(&flags.promptType, "type", "raw", "Prompt type (raw, qa, cot, summarize, optimize)")
	flag.BoolVar(&flags.verbose, "verbose", false, "Display verbose output including full prompt")
	flag.StringVar(&flags.provider, "provider", "", "LLM provider (anthropic, openai, groq, mistral, ollama, cohere)")
	flag.StringVar(&flags.model, "model", "", "LLM model")
	flag.Float64Var(&flags.temperature, "temperature", -1, "LLM temperature")
	flag.IntVar(&flags.maxTokens, "max-tokens", 0, "LLM max tokens")
	flag.DurationVar(&flags.timeout, "timeout", 0, "LLM timeout")
	flag.StringVar(&flags.apiKey, "api-key", "", "API key for the specified provider")
	flag.IntVar(&flags.maxRetries, "max-retries", DefaultMaxRetries, "Maximum number of retries for API calls")
	flag.DurationVar(&flags.retryDelay, "retry-delay", time.Second*DefaultRetryDelaySeconds, "Delay between retries")
	flag.StringVar(&flags.debugLevel, "debug-level", "warn", "Debug level (debug, info, warn, error)")
	flag.StringVar(&flags.outputFormat, "output-format", "", "Output format for structured responses (json)")
	flag.StringVar(
		&flags.optimizeGoal,
		"optimize-goal",
		"Improve the prompt's clarity and effectiveness",
		"Optimization goal",
	)
	flag.IntVar(
		&flags.optimizeIterations,
		"optimize-iterations",
		DefaultOptimizeIterations,
		"Number of optimization iterations",
	)
	flag.IntVar(
		&flags.optimizeMemory,
		"optimize-memory",
		DefaultOptimizeMemory,
		"Number of previous iterations to remember",
	)
	flag.Parse()
	return flags
}

func main() {
	flags := parseFlags()

	// Create LLM client
	llmClient, err := createLLMClient(flags)
	if err != nil {
		exitWithError("Error creating LLM client: %v\n", err)
	}

	// Get and validate prompt
	rawPrompt := getPrompt()

	// Generate and print response
	ctx := context.Background()
	processPrompt(ctx, llmClient, flags, rawPrompt)
}

// exitWithError prints an error message and exits
func exitWithError(format string, args ...any) {
	_, _ = fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(1)
}

// getPrompt gets the prompt from command-line arguments
func getPrompt() string {
	if len(flag.Args()) < 1 {
		_, _ = fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}
	return strings.Join(flag.Args(), " ")
}

// createLLMClient creates an LLM client with the given flags
func createLLMClient(flags *cmdFlags) (gollm.LLM, error) {
	configOpts := prepareConfigOptions(
		&flags.provider,
		&flags.model,
		&flags.temperature,
		&flags.maxTokens,
		&flags.timeout,
		&flags.apiKey,
		&flags.maxRetries,
		&flags.retryDelay,
		&flags.debugLevel,
	)
	client, err := gollm.NewLLM(configOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM client: %w", err)
	}
	return client, nil
}

// processPrompt generates and prints the response
func processPrompt(ctx context.Context, llmClient gollm.LLM, flags *cmdFlags, rawPrompt string) {
	response, fullPrompt, err := generateResponse(
		ctx,
		llmClient,
		flags.promptType,
		rawPrompt,
		flags.outputFormat,
		flags.optimizeGoal,
		flags.optimizeIterations,
		flags.optimizeMemory,
	)
	if err != nil {
		exitWithError("Error generating response: %v\n", err)
	}

	printResponse(flags.verbose, flags.promptType, fullPrompt, rawPrompt, response, flags.outputFormat)
}

func generateResponse(
	ctx context.Context,
	llmClient gollm.LLM,
	promptType, rawPrompt, outputFormat, optimizeGoal string,
	optimizeIterations, optimizeMemory int,
) (string, string, error) {
	switch promptType {
	case "qa":
		resp, err := presets.QuestionAnswer(ctx, llmClient, rawPrompt)
		if err != nil {
			return "", "", fmt.Errorf("question-answer: %w", err)
		}
		return resp, "", nil
	case "summarize":
		resp, err := presets.Summarize(ctx, llmClient, rawPrompt)
		if err != nil {
			return "", "", fmt.Errorf("summarize: %w", err)
		}
		return resp, "", nil
	case "optimize":
		return handleOptimize(ctx, llmClient, rawPrompt, optimizeGoal, optimizeIterations, optimizeMemory)
	default:
		return handleRaw(ctx, llmClient, rawPrompt, outputFormat)
	}
}

func handleOptimize(
	ctx context.Context,
	llmClient gollm.LLM,
	rawPrompt, optimizeGoal string,
	optimizeIterations, optimizeMemory int,
) (string, string, error) {
	promptOptimizer := optimizer.NewPromptOptimizer(
		llmClient,
		debug.NewDebugManager(true, "./debug_output"),
		llm.NewPrompt(rawPrompt),
		optimizeGoal,
		optimizer.WithIterations(optimizeIterations),
		optimizer.WithMemorySize(optimizeMemory),
	)
	optimizedPrompt, err := promptOptimizer.OptimizePrompt(ctx)
	if err != nil {
		return "", "", fmt.Errorf("optimize prompt: %w", err)
	}
	fullPrompt := fmt.Sprintf(
		"Initial Prompt: %s\nOptimization Goal: %s\nMemory Size: %d",
		rawPrompt,
		optimizeGoal,
		optimizeMemory,
	)
	return optimizedPrompt.Input, fullPrompt, nil
}

func handleRaw(ctx context.Context, llmClient gollm.LLM, rawPrompt, outputFormat string) (string, string, error) {
	prompt := gollm.NewPrompt(rawPrompt)
	if outputFormat == "json" {
		prompt.Apply(gollm.WithOutput("Please provide your response in JSON format."))
	}

	providerResponse, err := llmClient.Generate(ctx, prompt)
	if err != nil {
		return "", "", fmt.Errorf("generate: %w", err)
	}
	return providerResponse.AsText(), prompt.String(), nil
}

func prepareConfigOptions(
	provider, model *string,
	temperature *float64,
	maxTokens *int,
	timeout *time.Duration,
	apiKey *string,
	maxRetries *int,
	retryDelay *time.Duration,
	debugLevel *string,
) []gollm.ConfigOption {
	var configOpts []gollm.ConfigOption

	if *provider != "" {
		configOpts = append(configOpts, gollm.SetProvider(*provider))
	}
	if *model != "" {
		configOpts = append(configOpts, gollm.SetModel(*model))
	}
	if *temperature != -1 {
		configOpts = append(configOpts, gollm.SetTemperature(*temperature))
	}
	if *maxTokens != 0 {
		configOpts = append(configOpts, gollm.SetMaxTokens(*maxTokens))
	}
	if *timeout != 0 {
		configOpts = append(configOpts, gollm.SetTimeout(*timeout))
	}
	if *apiKey != "" {
		configOpts = append(configOpts, gollm.SetAPIKey(*apiKey))
	}
	configOpts = append(configOpts,
		gollm.SetMaxRetries(*maxRetries),
		gollm.SetRetryDelay(*retryDelay),
		gollm.SetLogLevel(getLogLevel(*debugLevel)))

	return configOpts
}

func printResponse(verbose bool, promptType, fullPrompt, rawPrompt, response, outputFormat string) {
	if verbose {
		if fullPrompt == "" {
			fullPrompt = rawPrompt // For qa, cot, and summarize, we don't have access to the full prompt
		}
		log.Printf("Prompt Type: %s\nFull Prompt:\n%s\n\nResponse:\n---------\n", promptType, fullPrompt)
	}

	if outputFormat != "json" {
		log.Println(response)
		return
	}
	printJSON(response)
}

func printJSON(response string) {
	var jsonResponse any
	if err := json.Unmarshal([]byte(response), &jsonResponse); err != nil {
		_, fmtErr := fmt.Fprintf(os.Stderr, "Error parsing JSON response: %v\n", err)
		if fmtErr != nil {
			return
		}
		log.Println(response) // Print raw response if JSON parsing fails
		return
	}
	jsonPretty, err := json.MarshalIndent(jsonResponse, "", "  ")
	if err != nil {
		log.Printf("Warning: Failed to format JSON: %v", err)
		jsonPretty = []byte(response)
	}
	log.Println(string(jsonPretty))
}

func getLogLevel(level string) gollm.LogLevel {
	switch strings.ToLower(level) {
	case "debug":
		return gollm.LogLevelDebug
	case "info":
		return gollm.LogLevelInfo
	case "error":
		return gollm.LogLevelError
	default:
		return gollm.LogLevelWarn
	}
}
