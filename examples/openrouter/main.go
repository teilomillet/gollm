package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/exec"
	"time"
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
			log.Println("Please set the OPENROUTER_API_KEY environment variable or use -key flag")
			os.Exit(1)
		}
	}

	// If the -test flag is provided, run integration tests instead of examples
	if *runTests {
		runIntegrationTests(*apiKey)
		return
	}

	// Otherwise, run the regular examples
	runExamples(*apiKey)
}

// runExamples runs all the OpenRouter examples
func runExamples(apiKey string) {
	ctx := context.Background()

	// Run all examples
	runBasicChatExample(ctx, apiKey)
	runModelFallbackExample(ctx, apiKey)
	runAutoRoutingExample(ctx, apiKey)
	runPromptCachingExample(ctx, apiKey)
	runJSONSchemaExample(ctx, apiKey)
	runReasoningTokensExample(ctx, apiKey)
	runProviderRoutingExample(ctx, apiKey)
	runToolCallingExample(ctx, apiKey)
}

// runIntegrationTests executes the OpenRouter integration tests with the provided API key.
func runIntegrationTests(apiKey string) {
	log.Println("Running OpenRouter integration tests with the provided API key...")
	log.Println("This will make actual API calls to OpenRouter and consume credits.")
	log.Println()

	// Set the API key in the environment for the tests
	if err := os.Setenv("OPENROUTER_API_KEY", apiKey); err != nil {
		log.Printf("Warning: Failed to set OPENROUTER_API_KEY: %v", err)
	}

	// Construct the test command
	cmd := exec.CommandContext(
		context.Background(),
		"go",
		"test",
		"-v",
		"../../providers",
		"-run",
		"TestOpenRouterIntegration",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start the command with a timeout
	if err := cmd.Start(); err != nil {
		log.Printf("Error starting tests: %v\n", err)
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
			log.Printf("Test execution failed: %v\n", err)
			os.Exit(1)
		}
	case <-time.After(5 * time.Minute): // 5 minute timeout
		// Kill the process if it takes too long
		if err := cmd.Process.Kill(); err != nil {
			log.Printf("Failed to kill test process: %v\n", err)
		}
		log.Println("Tests timed out after 5 minutes")
		os.Exit(1)
	}

	log.Println()
	log.Println("Tests completed successfully!")
}
