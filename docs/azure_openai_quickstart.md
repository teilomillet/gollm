# Azure OpenAI Quickstart Guide

This guide shows you how to use Azure OpenAI with Gollm in just a few steps.

## Prerequisites

- Azure OpenAI Service account
- Deployment of a model (e.g., GPT-4, GPT-3.5-Turbo)
- API Key for your Azure OpenAI resource

## Step 1: Set up the environment

Ensure you have the following information:

- **API Key**: Your Azure OpenAI API key
- **Resource Name**: Name of your Azure OpenAI resource (e.g., `my-openai-resource`)
- **Deployment Name**: Name of the model deployment (e.g., `gpt-4`)
- **API Version**: Version of the API to use (e.g., `2023-05-15`)

You can set these as environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_RESOURCE_NAME="your-resource-name"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
export AZURE_OPENAI_API_VERSION="2023-05-15"
```

## Step 2: Write your code

Here's a complete example:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/weave-labs/gollm"
	"github.com/weave-labs/gollm/config"
)

func main() {
	// Get configuration from environment
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	resourceName := os.Getenv("AZURE_OPENAI_RESOURCE_NAME")
	deploymentName := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	
	if apiKey == "" || resourceName == "" || deploymentName == "" {
		fmt.Println("Error: Missing required environment variables")
		fmt.Println("Please set: AZURE_OPENAI_API_KEY, AZURE_OPENAI_RESOURCE_NAME, AZURE_OPENAI_DEPLOYMENT_NAME")
		os.Exit(1)
	}
	
	if apiVersion == "" {
		apiVersion = "2023-05-15" // Default value
	}
	
	// Create the endpoint URL
	endpoint := fmt.Sprintf(
		"https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s", 
		resourceName, deploymentName, apiVersion,
	)
	
	// Create the LLM instance
	llm, err := gollm.NewLLM(
		config.SetProvider("azure-openai"),
		config.SetAPIKey(apiKey),
		config.SetModel(deploymentName),
		config.SetExtraHeaders(map[string]string{
			"azure_endpoint": endpoint,
		}),
	)
	
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}
	
	// Create a prompt
	ctx := context.Background()
	prompt := gollm.NewPrompt("Explain what Azure OpenAI Service is in 3 sentences.")
	
	// Generate a response
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error generating response: %v\n", err)
		os.Exit(1)
	}
	
	// Print the response
	fmt.Println("Response from Azure OpenAI:")
	fmt.Println(response)
}
```

## Step 3: Run your application

```bash
go run main.go
```

## How It Works

The example above uses the built-in `azure-openai` provider. The key differences from using OpenAI directly are:

1. **Endpoint construction**: Azure OpenAI requires a specific endpoint format that includes your resource name, deployment name, and API version
2. **Authentication**: Azure uses an `api-key` header instead of `Authorization: Bearer`
3. **Model parameter**: For Azure, the model parameter should be your deployment name

## Additional Options

You can also add other options as needed:

```go
llm, err := gollm.NewLLM(
    config.SetProvider("azure-openai"),
    config.SetAPIKey(apiKey),
    config.SetModel(deploymentName),
    config.SetExtraHeaders(map[string]string{
        "azure_endpoint": endpoint,
    }),
    config.SetTemperature(0.7),
    config.SetMaxTokens(500),
    config.SetLogLevel(gollm.LogLevelDebug),
)
```

For advanced usage, check out the [Provider System Documentation](provider_system.md). 