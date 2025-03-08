# OpenRouter Provider Example

This example demonstrates how to use the OpenRouter provider with GoLLM. OpenRouter is a service that provides access to multiple LLMs through a single API, with features like model routing, fallbacks, prompt caching, reasoning tokens, provider routing, and tool calling.

## Features Demonstrated

1. Basic text generation with OpenRouter
2. Model fallback capabilities
3. Auto-routing between models
4. Prompt caching for improved performance
5. JSON schema validation for structured outputs
6. Reasoning tokens for step-by-step thinking
7. Provider routing preferences
8. Tool/function calling support

## Prerequisites

- Go 1.18 or later
- An OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

## Running the Example

1. Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

2. Run the example:

```bash
go run main.go
```

Alternatively, you can provide the API key directly:

```bash
go run main.go -key="your_api_key_here"
```

## Running Integration Tests

To verify that the OpenRouter provider works correctly with the actual OpenRouter API, you can run the integration tests in two ways:

### Option 1: Using the main.go with -test flag

```bash
# Using environment variable
export OPENROUTER_API_KEY="your_api_key_here"
go run main.go -test

# Or providing the key directly
go run main.go -test -key="your_api_key_here"
```

### Option 2: Running the tests directly

```bash
# Set your API key as an environment variable
export OPENROUTER_API_KEY="your_api_key_here"

# Run just the OpenRouter integration tests
go test -v ./providers -run TestOpenRouterIntegration
```

These tests will make actual API calls to OpenRouter and consume credits from your account. They test various features including:

- Basic chat completion
- Model fallback mechanism
- JSON schema validation
- Message history with reasoning
- Tool/function calling

## Code Explanation

The example demonstrates several key features of OpenRouter:

### Basic Usage

```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openrouter"),
    gollm.SetAPIKey(apiKey),
    gollm.SetModel("anthropic/claude-3-5-sonnet"),
    gollm.SetTemperature(0.7),
    gollm.SetMaxTokens(1000),
)

prompt := gollm.NewPrompt("What are the main features of OpenRouter?")
response, err := llm.Generate(ctx, prompt)
```

### Model Fallbacks

OpenRouter allows specifying fallback models that will be used if the primary model is unavailable or returns an error:

```go
llm.SetOption("fallback_models", []string{"openai/gpt-4o", "gryphe/mythomax-l2-13b"})
```

### Auto-Routing

OpenRouter can automatically select the most appropriate model for a given prompt using the special `openrouter/auto` model:

```go
llm, err := gollm.NewLLM(
    gollm.SetProvider("openrouter"),
    gollm.SetAPIKey(apiKey),
    gollm.SetModel("openrouter/auto"),
    // ...
)
```

### Prompt Caching

Enable prompt caching to improve performance and reduce costs:

```go
llm.SetOption("enable_prompt_caching", true)
```

### JSON Schema Validation

Generate structured outputs that conform to a specific JSON schema:

```go
schema := map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "name": map[string]interface{}{
            "type": "string",
        },
        // ...
    },
    "required": []string{"name", "age", "interests"},
}

response, err = llm.GenerateWithSchema(ctx, prompt, schema)
```

### Reasoning Tokens

Enable reasoning tokens to get step-by-step thinking from the model:

```go
llm.SetOption("enable_reasoning", true)
```

### Provider Routing Preferences

Specify provider preferences for routing:

```go
providerPrefs := map[string]interface{}{
    "openai": map[string]interface{}{
        "weight": 1.0,
    },
}
llm.SetOption("provider_preferences", providerPrefs)
```

### Tool Calling

Use function/tool calling with compatible models:

```go
tools := []interface{}{
    map[string]interface{}{
        "type": "function",
        "function": map[string]interface{}{
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": map[string]interface{}{
                // ...
            },
        },
    },
}

llm.SetOption("tools", tools)
llm.SetOption("tool_choice", "auto")
```

## Available Models

OpenRouter provides access to models from various providers, including:

- OpenAI (e.g., "openai/gpt-4o")
- Anthropic (e.g., "anthropic/claude-3-5-sonnet")
- Mistral (e.g., "mistral/mistral-large")
- And many others

For a complete list of available models, see the [OpenRouter documentation](https://openrouter.ai/docs).

## Advanced Features

### Completions Endpoint Support

For applications that require the legacy completions API, use the CompletionsEndpoint method:

```go
// Not shown in example, but available in the provider
endpoint := openRouterProvider.CompletionsEndpoint()
```

### Generation Details

Retrieve generation details like cost and token usage:

```go
// Not shown in example, but available in the provider
generationDetails := openRouterProvider.GenerationEndpoint(generationId)
``` 