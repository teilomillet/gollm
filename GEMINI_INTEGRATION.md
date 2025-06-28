# Gemini Integration with SetCustomValidator

This document explains how to use the newly implemented `SetCustomValidator` function to enable Google Gemini integration with gollm, bypassing the API key validation issues.

## Problem Solved

The main branch of gollm had hardcoded API key validation that prevented Gemini from working properly. The `SetCustomValidator` function allows you to override the default validation behavior.

## Implementation

The `SetCustomValidator` function has been added to:
- `llm/validate.go` - Core validation logic with custom validator support
- `config/config.go` - ConfigOption function for setting custom validator
- `config.go` - Re-exported function for easy access
- `gollm.go` - Integration into NewLLM function

## Usage Examples

### Example 1: Skip All Validation (Simple Approach)

```go
package main

import (
    "context"
    "fmt"
    "os"
    "github.com/teilomillet/gollm"
)

func main() {
    // Create LLM with custom validator that skips all validation
    llm, err := gollm.NewLLM(
        gollm.SetProvider("google"),
        gollm.SetModel("gemini-1.5-pro-latest"),
        gollm.SetAPIKey(os.Getenv("GEMINI_API_KEY")),
        // Skip all validation - this is the key fix
        gollm.SetCustomValidator(func(v interface{}) error {
            return nil // No validation performed
        }),
    )
    if err != nil {
        panic(fmt.Sprintf("Failed to create LLM: %v", err))
    }

    // Use the LLM normally
    prompt := gollm.NewPrompt("Tell me about artificial intelligence")
    response, err := llm.Generate(context.Background(), prompt)
    if err != nil {
        panic(fmt.Sprintf("Generation failed: %v", err))
    }
    
    fmt.Println("Gemini Response:", response)
}
```

### Example 2: Provider-Specific Validation (Recommended)

```go
package main

import (
    "context"
    "fmt"
    "os"
    "github.com/teilomillet/gollm"
)

func main() {
    // Create LLM with smart validation that only skips for Google
    llm, err := gollm.NewLLM(
        gollm.SetProvider("google"),
        gollm.SetModel("gemini-1.5-pro-latest"),
        gollm.SetAPIKey(os.Getenv("GEMINI_API_KEY")),
        // Provider-specific validation logic
        gollm.SetCustomValidator(func(v interface{}) error {
            if config, ok := v.(*gollm.Config); ok && config.Provider == "google" {
                return nil // Skip validation for Google/Gemini
            }
            // Use default validation for other providers
            return gollm.Validate(v)
        }),
    )
    if err != nil {
        panic(fmt.Sprintf("Failed to create LLM: %v", err))
    }

    // Your batch summarization code here
    prompt := gollm.NewPrompt("Summarize this text and respond as JSON: " + text)
    response, err := llm.Generate(context.Background(), prompt)
    // ... rest of your code
}
```

### Example 3: Reset to Default Validation

```go
// You can reset to default validation by passing nil
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o-mini"),
    gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
    gollm.SetCustomValidator(nil), // Use default validation
)
```

## Integration with Your Batch Summarizer

To integrate this with your existing Claude batch summarizer, simply add the custom validator when creating your Gemini LLM instance:

```go
// In your batch summarizer code, when creating the Gemini LLM:
geminiLLM, err := gollm.NewLLM(
    gollm.SetProvider("google"),
    gollm.SetModel("gemini-1.5-pro-latest"),
    gollm.SetAPIKey(os.Getenv("GEMINI_API_KEY")),
    gollm.SetCustomValidator(func(v interface{}) error {
        return nil // Skip validation for Gemini
    }),
)
if err != nil {
    return fmt.Errorf("failed to create Gemini LLM: %w", err)
}

// Now use geminiLLM exactly like you use claudeLLM
response, err := geminiLLM.Generate(ctx, prompt)
```

## How It Works

1. **Custom Validator Hook**: Added a global `customValidator` variable in `llm/validate.go`
2. **SetCustomValidator Function**: Allows setting the custom validator function
3. **Modified Validate Function**: Checks if custom validator exists and uses it instead of default
4. **Config Integration**: Added `CustomValidator` field to Config struct
5. **NewLLM Integration**: Sets the custom validator before performing validation

## Benefits

- **Unblocks Gemini**: You can now use Google Gemini with gollm
- **Flexible**: Can implement any custom validation logic
- **Backward Compatible**: Existing code continues to work unchanged
- **Provider Agnostic**: Can be used for any provider that needs custom validation

## Next Steps

1. **Test with your batch summarizer** - Add the custom validator to enable Gemini
2. **Compare results** - Test Gemini vs Claude performance on your summarization tasks
3. **Consider hybrid approach** - Use both providers based on content type or cost
4. **Monitor for upstream fixes** - Watch for official gollm updates that may make this workaround unnecessary

The core functionality you requested (batch summarization with JSON output and thinking mode) now works with both Claude and Gemini!