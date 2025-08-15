package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/optimizer"
	"github.com/teilomillet/gollm/presets"
	"github.com/teilomillet/gollm/utils"
)

func main() {
	// Existing flags
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize, optimize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")
	provider := flag.String("provider", "", "LLM provider (anthropic, openai, groq, mistral, ollama, cohere)")
	model := flag.String("model", "", "LLM model")
	temperature := flag.Float64("temperature", -1, "LLM temperature")
	maxTokens := flag.Int("max-tokens", 0, "LLM max tokens")
	timeout := flag.Duration("timeout", 0, "LLM timeout")
	apiKey := flag.String("api-key", "", "API key for the specified provider")
	maxRetries := flag.Int("max-retries", 3, "Maximum number of retries for API calls")
	retryDelay := flag.Duration("retry-delay", time.Second*2, "Delay between retries")
	debugLevel := flag.String("debug-level", "warn", "Debug level (debug, info, warn, error)")
	outputFormat := flag.String("output-format", "", "Output format for structured responses (json)")

	// New flags for prompt optimization
	optimizeGoal := flag.String("optimize-goal", "Improve the prompt's clarity and effectiveness", "Optimization goal")
	optimizeIterations := flag.Int("optimize-iterations", 5, "Number of optimization iterations")
	optimizeMemory := flag.Int("optimize-memory", 2, "Number of previous iterations to remember")

	flag.Parse()

	// Prepare configuration options
	configOpts := prepareConfigOptions(provider, model, temperature, maxTokens, timeout, apiKey, maxRetries, retryDelay, debugLevel)

	// Create LLM client with the specified options
	llmClient, err := gollm.NewLLM(configOpts...)
	if err != nil {
		_, fmtErr := fmt.Fprintf(os.Stderr, "Error creating LLM client: %v\n", err)
		if fmtErr != nil {
			return
		}
		os.Exit(1)
	}

	if len(flag.Args()) < 1 {
		_, fmtErr := fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		if fmtErr != nil {
			return
		}
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(flag.Args(), " ")
	ctx := context.Background()

	var response string
	var fullPrompt string

	switch *promptType {
	case "qa":
		response, err = presets.QuestionAnswer(ctx, llmClient, rawPrompt)
	case "summarize":
		response, err = presets.Summarize(ctx, llmClient, rawPrompt)
	case "optimize":
		promptOptimizer := optimizer.NewPromptOptimizer(
			llmClient,
			utils.NewDebugManager(
				llmClient.GetLogger(),
				utils.DebugOptions{LogPrompts: true, LogResponses: true}),
			llmClient.NewPrompt(rawPrompt),
			*optimizeGoal,
			optimizer.WithIterations(*optimizeIterations),
			optimizer.WithMemorySize(*optimizeMemory),
		)
		optimizedPrompt, err := promptOptimizer.OptimizePrompt(ctx)
		if err == nil {
			response = optimizedPrompt.Input
			fullPrompt = fmt.Sprintf("Initial Prompt: %s\nOptimization Goal: %s\nMemory Size: %d", rawPrompt, *optimizeGoal, *optimizeMemory)
		}
	default:
		prompt := gollm.NewPrompt(rawPrompt)
		if *outputFormat == "json" {
			prompt.Apply(gollm.WithOutput("Please provide your response in JSON format."))
		}

		providerResponse, err := llmClient.Generate(ctx, prompt)
		if err == nil {
			response = providerResponse.AsText()
			fullPrompt = prompt.String()
		}
	}

	if err != nil {
		_, fmtErr := fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
		if fmtErr != nil {
			return
		}
		os.Exit(1)
	}

	printResponse(*verbose, *promptType, fullPrompt, rawPrompt, response, *outputFormat)
}

func prepareConfigOptions(provider, model *string, temperature *float64, maxTokens *int, timeout *time.Duration, apiKey *string, maxRetries *int, retryDelay *time.Duration, debugLevel *string) []gollm.ConfigOption {
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
	configOpts = append(configOpts, gollm.SetMaxRetries(*maxRetries))
	configOpts = append(configOpts, gollm.SetRetryDelay(*retryDelay))
	configOpts = append(configOpts, gollm.SetLogLevel(getLogLevel(*debugLevel)))

	return configOpts
}

func printResponse(verbose bool, promptType, fullPrompt, rawPrompt, response, outputFormat string) {
	if verbose {
		if fullPrompt == "" {
			fullPrompt = rawPrompt // For qa, cot, and summarize, we don't have access to the full prompt
		}
		fmt.Printf("Prompt Type: %s\nFull Prompt:\n%s\n\nResponse:\n---------\n", promptType, fullPrompt)
	}

	if outputFormat == "json" {
		var jsonResponse any
		err := json.Unmarshal([]byte(response), &jsonResponse)
		if err != nil {
			_, fmtErr := fmt.Fprintf(os.Stderr, "Error parsing JSON response: %v\n", err)
			if fmtErr != nil {
				return
			}
			fmt.Println(response) // Print raw response if JSON parsing fails
		} else {
			jsonPretty, _ := json.MarshalIndent(jsonResponse, "", "  ")
			fmt.Println(string(jsonPretty))
		}
	} else {
		fmt.Println(response)
	}
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
