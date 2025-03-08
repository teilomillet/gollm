package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
)

func main() {
	// Get API key and other credentials from environment
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: AZURE_OPENAI_API_KEY environment variable not set")
		os.Exit(1)
	}

	resourceName := os.Getenv("AZURE_OPENAI_RESOURCE_NAME")
	if resourceName == "" {
		fmt.Println("Error: AZURE_OPENAI_RESOURCE_NAME environment variable not set")
		os.Exit(1)
	}

	deploymentName := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
	if deploymentName == "" {
		fmt.Println("Error: AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set")
		os.Exit(1)
	}

	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		// Default to a recent version
		apiVersion = "2023-05-15"
		fmt.Println("Using default API version:", apiVersion)
	}

	// Method 1: Using the generic provider directly (more control)
	// -----------------------------------------------------------

	// Create a custom configuration for Azure OpenAI
	azureConfig := providers.ProviderConfig{
		Name: "my-azure-openai",
		Type: providers.TypeOpenAI,
		Endpoint: fmt.Sprintf("https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s",
			resourceName, deploymentName, apiVersion),
		AuthHeader: "api-key",
		AuthPrefix: "",
		RequiredHeaders: map[string]string{
			"Content-Type": "application/json",
		},
		SupportsSchema:    true,
		SupportsStreaming: true,
	}

	// Register the custom provider
	providers.RegisterGenericProvider("my-azure-openai", azureConfig)

	// Create a new LLM instance with our custom configuration
	llm, err := gollm.NewLLM(
		config.SetProvider("my-azure-openai"),
		config.SetAPIKey(apiKey),
		config.SetModel(deploymentName), // Azure uses deployment name as the model
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Create a context and prompt
	ctx := context.Background()
	prompt := gollm.NewPrompt("Summarize the main differences between Go and Python in 3 bullet points.")

	// Generate a response
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Response from Azure OpenAI:")
	fmt.Println(response)

	// Method 2: Using the built-in azure-openai configuration with ExtraHeaders
	// -----------------------------------------------------------------------

	// Create a custom endpoint string for Azure OpenAI
	endpoint := fmt.Sprintf("https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s",
		resourceName, deploymentName, apiVersion)

	// Create extra headers to pass endpoint information
	extraHeaders := map[string]string{
		"azure_endpoint": endpoint,
	}

	// Create a new LLM instance with the built-in azure-openai provider
	llm2, err := gollm.NewLLM(
		config.SetProvider("azure-openai"),
		config.SetAPIKey(apiKey),
		config.SetModel(deploymentName),
		config.SetExtraHeaders(extraHeaders),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Create a prompt for the second example
	prompt2 := gollm.NewPrompt("What are 3 key features of Go that make it suitable for cloud applications?")

	// Generate a response
	response2, err := llm2.Generate(ctx, prompt2)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nResponse from Azure OpenAI (method 2):")
	fmt.Println(response2)
}
