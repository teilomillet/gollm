package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"

	"github.com/teilomillet/gollm"
	llmopts "github.com/teilomillet/gollm/llm"
)

func main() {
	// Define command-line flags
	runTests := flag.Bool("test", false, "Run integration tests instead of examples")
	apiKey := flag.String("key", "", "OpenRouter API key (Required)")
	flag.Parse()

	// Get API key from environment variable if not provided as flag
	if *apiKey == "" {
		*apiKey = os.Getenv("OPENROUTER_API_KEY")
		if *apiKey == "" {
			fmt.Println("Please set the OPENROUTER_API_KEY environment variable or use -key flag")
			os.Exit(1)
		}
	}

	// If the -test flag is provided, run integration tests instead of examples
	if *runTests {
		runIntegrationTests(*apiKey)
		return
	}

	// Otherwise, run the regular examples
	ctx := context.Background()

	// Example 1: Basic Usage with Chat Completions
	fmt.Println("\n=== Example 1: Basic Usage with Chat Completions ===")
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("anthropic/claude-3-5-sonnet"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt := gollm.NewPrompt("What are the main features of OpenRouter?")
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 2: Model Fallbacks
	fmt.Println("\n=== Example 2: Model Fallbacks ===")
	llm.SetOption("fallback_models", []string{"openai/gpt-4o", "gryphe/mythomax-l2-13b"})
	prompt = gollm.NewPrompt("Explain how model fallback works in OpenRouter")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 3: Auto-Routing
	fmt.Println("\n=== Example 3: Auto-Routing ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("openrouter/auto"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt = gollm.NewPrompt("What is the best model for creative writing?")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 4: Prompt Caching
	fmt.Println("\n=== Example 4: Prompt Caching ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("anthropic/claude-3-5-sonnet"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Enable prompt caching
	llm.SetOption("enable_prompt_caching", true)

	// Create a large prompt that would benefit from caching
	largePromptText := "Here is a large text that would benefit from caching:\n"
	for i := range 50 {
		largePromptText += fmt.Sprintf("This is paragraph %d with some content that takes up space.\n", i)
	}
	largePromptText += "\nSummarize the above text."

	prompt = gollm.NewPrompt(largePromptText)
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 5: JSON Schema Validation
	fmt.Println("\n=== Example 5: JSON Schema Validation ===")
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": "string",
			},
			"age": map[string]any{
				"type": "integer",
			},
			"interests": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
				},
			},
		},
		"required": []string{"name", "age", "interests"},
	}

	prompt = gollm.NewPrompt("Create a profile for a fictional person who loves technology")
	response, err = llm.Generate(ctx, prompt, llmopts.WithStructuredResponse(schema))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 6: Reasoning Tokens
	fmt.Println("\n=== Example 6: Reasoning Tokens ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("anthropic/claude-3-5-sonnet"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Enable reasoning tokens
	llm.SetOption("enable_reasoning", true)

	prompt = gollm.NewPrompt("What is the square root of 144 and why?")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response with reasoning: %s\n\n", response)
	}

	// Example 7: Provider Routing Preferences
	fmt.Println("\n=== Example 7: Provider Routing Preferences ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("openai/gpt-4o"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Set provider routing preferences
	providerPrefs := map[string]any{
		"openai": map[string]any{
			"weight": 1.0,
		},
	}
	llm.SetOption("provider_preferences", providerPrefs)

	prompt = gollm.NewPrompt("Explain how provider routing works in OpenRouter")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 8: Tool Calling
	fmt.Println("\n=== Example 8: Tool Calling ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openrouter"),
		gollm.SetAPIKey(*apiKey),
		gollm.SetModel("openai/gpt-4o"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Define tools
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "Get the current weather in a given location",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
						"unit": map[string]any{
							"type":        "string",
							"enum":        []string{"celsius", "fahrenheit"},
							"description": "The temperature unit to use",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	// Set the tools
	llm.SetOption("tools", tools)
	llm.SetOption("tool_choice", "auto")

	prompt = gollm.NewPrompt("What's the weather in San Francisco?")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Tool call response: %s\n\n", response)
	}
}

// runIntegrationTests executes the OpenRouter integration tests with the provided API key.
func runIntegrationTests(apiKey string) {
	fmt.Println("Running OpenRouter integration tests with the provided API key...")
	fmt.Println("This will make actual API calls to OpenRouter and consume credits.")
	fmt.Println()

	// Set the API key in the environment for the tests
	if err := os.Setenv("OPENROUTER_API_KEY", apiKey); err != nil {
		log.Printf("Warning: Failed to set OPENROUTER_API_KEY: %v", err)
	}

	// Construct the test command
	cmd := exec.Command("go", "test", "-v", "../../providers", "-run", "TestOpenRouterIntegration")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start the command with a timeout
	if err := cmd.Start(); err != nil {
		fmt.Printf("Error starting tests: %v\n", err)
		os.Exit(1)
	}

	// Wait for the command to complete with a timeout
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	// Set a timeout to prevent hanging
	select {
	case err := <-done:
		if err != nil {
			fmt.Printf("Test execution failed: %v\n", err)
			os.Exit(1)
		}
	case <-time.After(5 * time.Minute): // 5 minute timeout
		// Kill the process if it takes too long
		if err := cmd.Process.Kill(); err != nil {
			fmt.Printf("Failed to kill test process: %v\n", err)
		}
		fmt.Println("Tests timed out after 5 minutes")
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("Tests completed successfully!")
}
